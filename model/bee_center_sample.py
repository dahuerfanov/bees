import math

import numpy as np
import torch

from model.utils.gaussian import draw_umich_gaussian, draw_msra_gaussian, gaussian_radius


class BeeCenterSample:
    """Class for sampling augmentations on images and targets, adapted from bboxes to only points.

    This class processes bee center annotations to generate the required training targets for CenterNet,
    including heatmaps and regression values. It handles scaling coordinates between input and output
    resolutions and applies Gaussian kernels around center points.

    Args:
        down_ratio (int, optional): Factor by which the input is downsampled. Defaults to 4.
        num_classes (int, optional): Number of object classes. Defaults to 1.
        max_objects (int, optional): Maximum number of objects per image. Defaults to 128.
        kernel_px (int, optional): Size of Gaussian kernel in pixels. Defaults to 30.
        gaussian_type (str, optional): Type of Gaussian kernel ("umich" or "msra"). Defaults to "umich".

    The class generates the following targets:
        - heatmap: Gaussian peaks centered on bee locations
        - regression: Sub-pixel offset from quantized center points
        - regression_mask: Mask indicating valid regression targets
        - indices: Linear indices of center points in flattened output
        - width_height: Placeholder for object sizes (unused for points)
        - original_pts: Original center points normalized to [0,1]
    """

    def __init__(
        self,
        down_ratio=4,
        num_classes=1,
        max_objects=128,
        kernel_px=30,
        gaussian_type="umich",
    ):

        self.down_ratio = down_ratio
        self.kernel_px = kernel_px
        self.num_classes = num_classes
        self.max_objects = max_objects
        self.gaussian_type = gaussian_type

    @staticmethod
    def _coco_box_to_bbox(box):
        return np.array(
            [box[0], box[1], box[0] + box[2], box[1] + box[3]], dtype=np.float32
        )

    def scale_point(self, point, output_size):
        x, y = point / self.down_ratio
        output_h, output_w = output_size

        x = np.clip(x, 0, output_w - 1)
        y = np.clip(y, 0, output_h - 1)

        return [x, y]

    def __call__(self, img, target):
        _, input_w, input_h = img.shape

        output_h = input_h // self.down_ratio
        output_w = input_w // self.down_ratio

        heatmap = torch.zeros(
            (self.num_classes, output_h, output_w), dtype=torch.float32
        )
        width_height = torch.zeros((self.max_objects, 2), dtype=torch.float32)
        regression = torch.zeros((self.max_objects, 2), dtype=torch.float32)
        regression_mask = torch.zeros(self.max_objects, dtype=torch.bool)
        indices = torch.zeros(self.max_objects, dtype=torch.int64)
        original_pts = torch.zeros((self.max_objects, 2), dtype=torch.float32)

        draw_gaussian = (
            draw_msra_gaussian if self.gaussian_type == "msra" else draw_umich_gaussian
        )

        num_objects = min(len(target), self.max_objects)
        for k in range(num_objects):
            ann = target[k]
            bbox = self._coco_box_to_bbox(ann["bbox"])
            cls_id = ann["class_id"] if "class_id" in ann else int(ann["category_id"]) - 1

            # Scale to output size
            bbox[:2] = self.scale_point(bbox[:2], (output_h, output_w))
            bbox[2:] = self.scale_point(bbox[2:], (output_h, output_w))

            radius = gaussian_radius((math.ceil(self.kernel_px // self.down_ratio),
                                      math.ceil(self.kernel_px // self.down_ratio)))
            radius = max(0, int(radius))
            ct = torch.FloatTensor(
                [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            )
            ct_int = ct.to(torch.int32)

            draw_gaussian(heatmap[cls_id], ct_int, radius)
            indices[k] = ct_int[1] * output_w + ct_int[0]
            regression[k] = ct - ct_int
            regression_mask[k] = 1
            original_pts[k] = ct * self.down_ratio
            original_pts[k, 0] /= input_w
            original_pts[k, 1] /= input_h

        ret = {
            "heatmap": heatmap,
            "regression_mask": regression_mask,
            "indices": indices,
            "width_height": width_height,
            "regression": regression,
            "original_pts": original_pts,
        }

        return img, ret
