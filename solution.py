import argparse
import yaml
import os

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from model.bee_augmentations import get_transforms
from model.utils.decode import extract_detections


def read_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def main():
    parser = argparse.ArgumentParser(description='Run inference with exported CenterNetBeeCenter model')
    parser.add_argument('--model', type=str, default="trained_models/model.pt",
                        help='Path to TorchScript model file (.pt)')
    parser.add_argument('--config_path', type=str, default="config/exp_config.yaml",
                        help='Path to TorchScript model file (.pt)')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image file')
    args = parser.parse_args()

    # Check if config file exists
    if not os.path.exists(args.config_path):
        print(f"Error: Config file not found at {args.config_path}")
        return
    config = read_config(args.config_path)

    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        return
    model = torch.jit.load(args.model)
    model.eval()

    # Check if image file exists
    if not os.path.exists(args.image):
        print(f"Error: Image file not found at {args.image}")
        return
    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: Could not read image at {args.image}")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Get test transform
    _, _, test_transform = get_transforms(
        norm_mean=config["image"]["mean"],
        norm_std=config["image"]["std"],
        valid_ids=config["detection"]["valid_object_ids"],
        max_objs=config["detection"]["max_objects"],
        kernel_px=32
    )

    # Apply transform and add batch dimension
    img_tensor, _ = test_transform(img, [])
    img_tensor = img_tensor.unsqueeze(0)

    # Run inference
    with torch.no_grad():
        outputs = model(img_tensor)

    # Get image dimensions
    tensor_h, tensor_w = img_tensor.shape[2: ]
    roi_h = int(config["image"]["longer_side"] * config["image"]["aspect_ratio"])
    roi_w = config["image"]["longer_side"]
    roi_x = (tensor_w - roi_w) // 2
    roi_y = (tensor_h - roi_h) // 2

    # Convert tensor image back to numpy for plotting
    img_tensor_denorm = img_tensor[0].clone()
    img_tensor_denorm = img_tensor_denorm * torch.tensor(config["image"]["std"]).view(3,1,1)
    img_tensor_denorm = img_tensor_denorm + torch.tensor(config["image"]["mean"]).view(3,1,1)
    img_tensor_denorm = torch.clamp(img_tensor_denorm, 0, 1)
    img_tensor_np = img_tensor_denorm.permute(1,2,0).cpu().numpy()

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot transformed image with detections
    ax1.imshow(img_tensor_np[:, :, ::-1])
    ax1.set_title('Detections')

    # Get heatmap and plot
    heatmap = outputs[0][0, 0].cpu().numpy()
    heatmap = 1 / (1 + np.exp(-heatmap)) # apply sigmoid
    heatmap = cv2.resize(heatmap, (tensor_w, tensor_h))
    heatmap_masked = np.zeros_like(heatmap)
    heatmap_masked[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] = heatmap[roi_y:roi_y+roi_h,
                                                                    roi_x:roi_x+roi_w]
    heatmap_masked = 0.55 - np.clip(heatmap_masked, 0, 0.55)
    ax2.imshow(heatmap_masked, cmap='Reds_r', vmin=0, vmax=0.55)
    ax2.set_title('Heatmap')

    # Get detections
    detections = extract_detections({"heatmap": outputs[0], "regression": outputs[1]},
                                     config["detection"]["max_objects"], 4)
    detections = detections[0].cpu()  # Get first sample in batch

    # Plot high confidence detections within ROI
    bee_count = 0
    for det in detections:
        x, y = det[0].item(), det[1].item()
        score = det[2].item()
        score = 1 / (1 + np.exp(-score)) # apply sigmoid
        if (score > config["detection"]["score_threshold"] and
            roi_x <= x < roi_x + roi_w and
            roi_y <= y < roi_y + roi_h):
            ax1.plot(x, y, 'ro', markersize=4)
            bee_count += 1

    # Add bee count to legend
    ax1.plot([], [], 'ro', markersize=4, label=f'Bees detected: {bee_count}')
    ax1.legend()

    plt.tight_layout()
    plt.savefig('bee_img.png')
    plt.close()


if __name__ == '__main__':
    main()