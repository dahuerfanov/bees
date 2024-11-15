from argparse import ArgumentParser
import os
import yaml

from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torchvision.datasets import CocoDetection
import imgaug as ia
import pytorch_lightning as pl
import torch

from model.bee_augmentations import get_transforms
from model.centernet_bee_center import CenterNetBeeCenter


class ExportTorchScriptCallback(Callback):
    """Callback to export model to TorchScript and state dict after training"""
    def __init__(self, output_path):
        super().__init__()
        self.output_path = output_path
        
    def on_train_end(self, trainer, pl_module):
        # Load best checkpoint
        best_model_path = trainer.checkpoint_callback.best_model_path
        if best_model_path:
            # Load the checkpoint directly into the current model
            checkpoint = torch.load(best_model_path)
            pl_module.load_state_dict(checkpoint['state_dict'])
            
            # Prepare model for export
            pl_module.eval()
            pl_module = pl_module.to('cpu')
            example_input = torch.randn(1, 3, 512, 512)
            
            class ModelWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, x):
                    output = self.model(x)
                    # Return a tuple instead of a dict
                    return output["heatmap"], output["regression"]
            
            wrapped_model = ModelWrapper(pl_module)
            # Add strict=False to allow the trace
            scripted_model = torch.jit.trace(wrapped_model, example_input, strict=False)
            scripted_model.save(self.output_path)
            print(f"Model exported to {self.output_path}")
            
            # Save state dict
            state_dict_path = self.output_path.replace('.pt', '.pth')
            torch.save(pl_module.state_dict(), state_dict_path)
            print(f"Model state dict saved to {state_dict_path}")
        else:
            print("No best model checkpoint found. Model exports skipped.")

def run(args):
    """
    Trains a CenterNet model on bee center detection data and export the state dictionary (.pth)
    and a torchscript file (.pt) with the best found checkpoint.
    
    Expected data format:
    - Dataset root should contain train_coco.json, val_coco.json and test_coco.json files
    - Dataset root should contain the folder img with samples
    """
    pl.seed_everything(5318008)
    ia.seed(107734)

    with open(args.exp_config_path, 'r') as file:
        exp_config = yaml.safe_load(file)

    train_transform, valid_transform, _ = get_transforms(
        norm_mean=exp_config["image"]["mean"],
        norm_std=exp_config["image"]["std"],
        valid_ids=exp_config["detection"]["valid_object_ids"],
        max_objs=exp_config["detection"]["max_objects"],
        kernel_px=32
    )

    coco_train = CocoDetection(
        os.path.join(args.dataset_root, "img"),
        os.path.join(args.dataset_root, "train_coco.json"),
        transforms=train_transform,
    )

    coco_val = CocoDetection(
        os.path.join(args.dataset_root, "img"),
        os.path.join(args.dataset_root, "val_coco.json"),
        transforms=valid_transform,
    )

    train_loader = DataLoader(
        coco_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        coco_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    # ------------
    # model
    # ------------
    args.learning_rate_milestones = list(map(int, args.learning_rate_milestones.split(",")))
    model = CenterNetBeeCenter(exp_config, args.learning_rate, args.learning_rate_milestones)
    if args.pretrained_weights_path:
        model.load_pretrained_weights(args.pretrained_weights_path)

    # ------------
    # training
    # ------------
    logger = WandbLogger(project="honeybee",
                         offline=False,
                         log_model=True)
    logger.log_hyperparams(args)

    callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=5,
            save_last=True,
            every_n_epochs=2,
            filename="epoch_{epoch:02d}-val_loss_{val_loss:.2f}"
        ),
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            filename="best",
            save_top_k=1
        ),
        LearningRateMonitor(logging_interval="epoch"),
        ExportTorchScriptCallback(args.torchscript_output)
    ]

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        max_epochs=args.epochs
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_root")
    parser.add_argument("--pretrained_weights_path")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--exp_config_path", type=str, default="config/exp_config.yaml",
                        help="yaml config file containing exp constants like img dims")
    parser.add_argument("--torchscript_output", type=str, default="trained_models/model.pt",
                        help="Output path for TorchScript model (default: model.pt)")
    parser = CenterNetBeeCenter.add_model_specific_args(parser)
    args = parser.parse_args()
    
    run(args)