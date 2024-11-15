"""Tests for the bee detector model.

This module contains unit tests for the bee detector model, testing:
- Model loading and configuration
- Output shape validation 
- Detection extraction from heatmaps
- Score thresholding and coordinate validation

The tests verify that:
1. The model loads correctly with the expected config
2. Model outputs have the correct shapes and formats
3. Detection extraction produces valid detections
4. Score thresholding filters detections appropriately
"""

import unittest
import os
import yaml

import numpy as np
import torch

from model.bee_augmentations import get_transforms
from model.utils.decode import extract_detections


class TestBeeDetector(unittest.TestCase):
    def setUp(self):
        # Load config and model
        self.config_path = "config/exp_config.yaml"
        self.model_path = "trained_models/model.pt"
        
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
            
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
            
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.model = torch.jit.load(self.model_path)
        self.model.eval()
        
        # Get transforms
        _, _, self.test_transform = get_transforms(
            norm_mean=self.config["image"]["mean"],
            norm_std=self.config["image"]["std"],
            valid_ids=self.config["detection"]["valid_object_ids"],
            max_objs=self.config["detection"]["max_objects"],
            kernel_px=32
        )

    def test_model_output_shape(self):
        # Create dummy input
        dummy_img = np.zeros((500, 500, 3), dtype=np.uint8)
        img_tensor, _ = self.test_transform(dummy_img, [])
        img_tensor = img_tensor.unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(img_tensor)
            
        # Check output shapes
        self.assertEqual(len(outputs), 2)  # Should output heatmap and regression
        self.assertEqual(outputs[0].shape[1], 1)  # Single class detection
        self.assertEqual(outputs[0].shape[2:], outputs[1].shape[2:])  # Spatial dims should match
        
    def test_detection_extraction(self):
        # Create dummy predictions
        batch_size = 1
        h, w = 128, 128
        dummy_heatmap = torch.zeros((batch_size, 1, h, w))
        dummy_regression = torch.zeros((batch_size, 2, h, w))
        
        # Add a fake detection
        dummy_heatmap[0, 0, 64, 64] = 5.0  # High confidence at center
        
        detections = extract_detections(
            {"heatmap": dummy_heatmap, "regression": dummy_regression},
            max_objs=self.config["detection"]["max_objects"],
            down_ratio=4
        )
        
        # Check detection format
        self.assertEqual(detections.shape[0], batch_size)
        self.assertGreater(len(detections[0]), 0)  # Should detect at least one object
        self.assertEqual(detections[0].shape[1], 4)  # x, y, confidence, class
        
    def test_score_thresholding(self):
        # Create test image with known detections
        dummy_img = np.zeros((500, 500, 3), dtype=np.uint8)
        img_tensor, _ = self.test_transform(dummy_img, [])
        img_tensor = img_tensor.unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            
        detections = extract_detections(
            {"heatmap": outputs[0], "regression": outputs[1]},
            max_objs=self.config["detection"]["max_objects"],
            down_ratio=4
        )[0].cpu()
        
        # Check that all detections above threshold have valid coordinates
        for det in detections:
            score = 1 / (1 + np.exp(-det[2].item()))  # Apply sigmoid
            if score > self.config["detection"]["score_threshold"]:
                self.assertGreaterEqual(det[0].item(), 0)  # x coordinate
                self.assertGreaterEqual(det[1].item(), 0)  # y coordinate
                self.assertLess(det[0].item(), 512)  # x within bounds
                self.assertLess(det[1].item(), 512)  # y within bounds


if __name__ == '__main__':
    unittest.main()
