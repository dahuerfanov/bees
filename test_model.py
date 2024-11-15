from argparse import ArgumentParser
import os
import yaml

from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
import pytorch_lightning as pl
import torch

from model.bee_augmentations import get_transforms
from model.centernet_bee_center import CenterNetBeeCenter
from tools.prepare_data_from_file_lists import create_coco_annotations


def test(args):
    """
    Tests a trained CenterNet model on bee center detection test data.
    
    Expected data format:
    - Dataset root should contain test_imgs.txt file listing test images
    - Dataset root should contain the folders img/ and gt-dots/ with test samples
    """
    pl.seed_everything(5318008)

    # Create test COCO annotations if they don't exist
    test_coco_path = os.path.join(args.dataset_root, "test_coco.json")
    if not os.path.exists(test_coco_path):
        test_file_list = os.path.join(args.dataset_root, "test_imgs.txt")
        if not os.path.exists(test_file_list):
            raise FileNotFoundError(f"Test file list not found at {test_file_list}")
            
        create_coco_annotations(
            args.dataset_root,
            test_file_list,
            test_coco_path,
            ['png', 'jpg', 'jpeg', 'bmp'],
            args.image_prefix,
            args.gt_prefix
        )
        print(f"Created test COCO annotations at {test_coco_path}")

    # Load experiment config
    with open(args.exp_config_path, 'r') as file:
        exp_config = yaml.safe_load(file)

    # Get transforms
    _, _, test_transform = get_transforms(
        norm_mean=exp_config["image"]["mean"],
        norm_std=exp_config["image"]["std"],
        valid_ids=exp_config["detection"]["valid_object_ids"],
        max_objs=exp_config["detection"]["max_objects"],
        kernel_px=32
    )

    # Load test dataset
    coco_test = CocoDetection(
        os.path.join(args.dataset_root, "img"),
        test_coco_path,
        transforms=test_transform,
    )

    test_loader = DataLoader(
        coco_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Initialize model and load weights
    args.learning_rate_milestones = list(map(int, args.learning_rate_milestones.split(",")))
    model = CenterNetBeeCenter(exp_config, args.learning_rate, args.learning_rate_milestones)
    
    if not args.model_weights_path:
        raise ValueError("Model weights path must be provided for testing")
    
    checkpoint = torch.load(args.model_weights_path)
    model.load_state_dict(checkpoint)

    # Test
    trainer = pl.Trainer()
    trainer.test(model, test_loader)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_root", required=True,
                      help="Root directory containing img/ and gt-dots/ folders")
    parser.add_argument("--model_weights_path", required=True,
                      help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--exp_config_path", type=str, default="config/exp_config.yaml",
                      help="yaml config file containing exp constants like img dims")
    parser.add_argument("--image_prefix", default='beeType1_',
                      help="Prefix to add to image filenames")
    parser.add_argument("--gt_prefix", default='dots',
                      help="Prefix to add to ground truth filenames")
    parser = CenterNetBeeCenter.add_model_specific_args(parser)
    args = parser.parse_args()
    
    test(args)
