import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from model.bee_augmentations import get_transforms
from model.centernet_bee_center import CenterNetBeeCenter, extract_detections


def main():
    parser = argparse.ArgumentParser(description='Run inference with exported CenterNetBeeCenter model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to TorchScript model file (.pt)')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image file')
    args = parser.parse_args()

    # Load model
    model = torch.jit.load(args.model)
    model.eval()

    # Read and preprocess image
    img = cv2.imread(args.image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get test transform
    _, _, test_transform = get_transforms(
        norm_mean=CenterNetBeeCenter.mean,
        norm_std=CenterNetBeeCenter.std,
        valid_ids=CenterNetBeeCenter.valid_ids,
        max_objs=CenterNetBeeCenter.max_objs,
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
    roi_h = int(CenterNetBeeCenter.img_longer_side * CenterNetBeeCenter.img_aspect_ratio)
    roi_w = CenterNetBeeCenter.img_longer_side
    roi_x = (tensor_w - roi_w) // 2
    roi_y = (tensor_h - roi_h) // 2

    # Convert tensor image back to numpy for plotting
    img_tensor_denorm = img_tensor[0].clone()
    img_tensor_denorm = img_tensor_denorm * torch.tensor(CenterNetBeeCenter.std).view(3,1,1)
    img_tensor_denorm = img_tensor_denorm + torch.tensor(CenterNetBeeCenter.mean).view(3,1,1)
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
                                     CenterNetBeeCenter.max_objs, 4)
    detections = detections[0].cpu()  # Get first sample in batch

    # Plot high confidence detections within ROI
    bee_count = 0
    for det in detections:
        x, y = det[0].item(), det[1].item()
        score = det[2].item()
        score = 1 / (1 + np.exp(-score)) # apply sigmoid
        if (score > CenterNetBeeCenter.score_threshold and
            roi_x <= x < roi_x + roi_w and
            roi_y <= y < roi_y + roi_h):
            ax1.plot(x, y, 'ro', markersize=4)
            bee_count += 1

    # Add bee count to legend
    ax1.plot([], [], 'ro', markersize=4, label=f'Bees detected: {bee_count}')
    ax1.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
