from argparse import ArgumentParser
import os

from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torchvision.datasets import CocoDetection
import imgaug as ia
import pytorch_lightning as pl

from model.bee_augmentations import get_transforms
from model.centernet_bee_center import CenterNetBeeCenter


def run():
    """
    Trains a CenterNet model on bee center detection data.
    
    Expected data format:
    - Dataset root should contain train/, val/ and test/ subdirectories
    - Each subdirectory should have:
        - img/ folder containing the images
        - *_coco.json annotation file in COCO format with bounding boxes around individual bees
        - Annotations should have category_id=1 for bees
    """
    pl.seed_everything(5318008)
    ia.seed(107734)

    train_transform, valid_transform, _ = get_transforms(
        norm_mean=CenterNetBeeCenter.mean,
        norm_std=CenterNetBeeCenter.std,
        valid_ids=CenterNetBeeCenter.valid_ids,
        max_objs=CenterNetBeeCenter.max_objs,
        kernel_px=32
    )

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument("--dataset_root")

    parser.add_argument("--pretrained_weights_path")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_workers", default=1, type=int)
    parser = CenterNetBeeCenter.add_model_specific_args(parser)
    args = parser.parse_args()

    coco_train = CocoDetection(
        os.path.join(args.dataset_root, "train", "img"),
        os.path.join(args.dataset_root, "train", "train_coco.json"),
        transforms=train_transform,
    )

    coco_val = CocoDetection(
        os.path.join(args.dataset_root, "val", "img"),
        os.path.join(args.dataset_root, "val", "val_coco.json"),
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
    model = CenterNetBeeCenter(
        "res_18", args.learning_rate,
        args.learning_rate_milestones,
    )
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
    ]

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        max_epochs=20
    )
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    #trainer.test(dataloaders=test_loader, ckpt_path="best")


#os.environ['WANDB_DISABLED'] = 'true'

if __name__ == "__main__":
    run()