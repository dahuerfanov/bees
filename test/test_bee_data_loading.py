"""
Unit tests for bee data loading and transformations.

This module tests the data loading pipeline for bee detection, including:
- Creation and validation of COCO format annotations from image/ground truth pairs
- Loading and transforming images and annotations through the training pipeline
- Consistency checks for train/val/test data splits
- Verification of transform output shapes and formats

The tests ensure:
- COCO annotation files are created correctly for each data split
- Each split contains the expected number of unique images
- Images can be loaded and transformed with correct output formats
- All transforms produce consistent output shapes
"""

import json
import os
import unittest
import yaml

from PIL import Image
import numpy as np

from model.bee_augmentations import get_transforms
from tools.prepare_data_from_file_lists import create_coco_annotations


class TestBeeDataLoading(unittest.TestCase):
    def setUp(self):
        # Load config
        self.config_path = "config/exp_config.yaml"
        self.data_root = "test/test_data"
        
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
            
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        # Create COCO annotations for each split
        splits = ['train', 'val', 'test']
        extensions = ['png', 'jpg', 'jpeg', 'bmp']
        for split in splits:
            file_list = os.path.join(self.data_root, f"{split}_imgs.txt")
            output_path = os.path.join(self.data_root, f"{split}_coco.json")
            create_coco_annotations(
                self.data_root,
                file_list,
                output_path,
                extensions,
                image_prefix="beeType1_",
                gt_prefix="dots"
            )

        # Get transforms
        self.train_transform, self.val_transform, self.test_transform = get_transforms(
            norm_mean=self.config["image"]["mean"],
            norm_std=self.config["image"]["std"],
            valid_ids=self.config["detection"]["valid_object_ids"],
            max_objs=self.config["detection"]["max_objects"],
            kernel_px=32
        )

    def test_data_splits_exist(self):
        """Test that train/val/test splits were created"""
        splits = ['train', 'val', 'test']
        
        for split in splits:
            coco_path = os.path.join(self.data_root, f"{split}_coco.json")
            self.assertTrue(os.path.exists(coco_path), 
                          f"COCO annotation file missing for {split} split")
            
            # Verify COCO file structure
            with open(coco_path) as f:
                coco_data = json.load(f)
                
            self.assertIn("images", coco_data)
            self.assertIn("annotations", coco_data)
            self.assertIn("categories", coco_data)

    def test_json_image_count(self):
        """Test that each JSON file contains exactly 2 images and that image filenames are unique across splits"""
        splits = ['train', 'val', 'test']
        all_filenames = set()
        
        for split in splits:
            coco_path = os.path.join(self.data_root, f"{split}_coco.json")
            with open(coco_path) as f:
                coco_data = json.load(f)
            
            # Check image count
            self.assertEqual(len(coco_data["images"]), 2,
                           f"{split} split should contain exactly 2 images")
            
            # Check for unique filenames
            split_filenames = {img["file_name"] for img in coco_data["images"]}
            overlap = split_filenames & all_filenames
            self.assertEqual(len(overlap), 0,
                           f"Found duplicate filenames between {split} and other splits: {overlap}")
            all_filenames.update(split_filenames)

    def test_image_loading(self):
        """Test that images can be loaded and transformed"""
        
        # Load first image from train split
        with open(os.path.join(self.data_root, "train_coco.json")) as f:
            train_data = json.load(f)
            
        first_img = train_data["images"][0]
        img_path = os.path.join(self.data_root, "img", first_img["file_name"])
        
        # Test image exists
        self.assertTrue(os.path.exists(img_path))
        
        # Test image can be loaded
        img = Image.open(img_path)
        img = np.array(img)
        
        # Get annotations for this image
        img_id = first_img["id"]
        annots = []
        for ann in train_data["annotations"]:
            if ann["image_id"] == img_id:
                # Convert points to expected format
                annots.append({
                    'bbox': ann["bbox"],
                    'category_id': ann["category_id"]
                })
                
        # Test transform
        img_tensor, target = self.train_transform(img, annots)
        
        # Check tensor format
        self.assertEqual(len(img_tensor.shape), 3)  # C,H,W
        self.assertEqual(img_tensor.shape[0], 3)  # RGB channels
        
        # Check target format
        self.assertIn("heatmap", target)
        self.assertIn("regression", target)
        self.assertIn("regression_mask", target)
        self.assertIn("indices", target)
        self.assertIn("original_pts", target)

    def test_transforms_consistency(self):
        """Test that transforms produce consistent output shapes"""
        dummy_img = np.zeros((512, 512, 3), dtype=np.uint8)
        dummy_points = [{
            'bbox': [250, 250, 0, 0],
            'category_id': 1
        }] 
        
        # Test all transforms
        transforms = [self.train_transform, self.val_transform, self.test_transform]
        for transform in transforms:
            img_tensor, target = transform(dummy_img, dummy_points)
            
            # Check image dimensions match input sizes
            self.assertEqual(img_tensor.shape[1], 512)
            self.assertEqual(img_tensor.shape[2], 512)


if __name__ == '__main__':
    unittest.main()
