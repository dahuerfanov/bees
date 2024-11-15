import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torchvision
import wandb

from CenterNet import CenterNet
from CenterNet.models.heads import CenterHead
from model.utils.decode import extract_detections, sigmoid_clamped
from model.utils.losses import RegL1Loss, FocalLoss


class CenterNetBeeCenter(CenterNet):
    """CenterNet model adapted for bee center detection.

    This model is a variant of CenterNet specifically designed for detecting bee centers in images.
    It uses a single class (bee) and outputs both heatmaps and regression values to predict bee locations.

    Args:
        config (dict): Configuration dictionary containing model parameters:
            image:
                mean (list): Mean values for image normalization
                std (list): Standard deviation values for image normalization
                aspect_ratio (float): Expected aspect ratio of input images
                longer_side (int): Length in pixels of longer image side
            detection:
                score_threshold (float): Confidence threshold for detections
                max_objects (int): Maximum number of objects to detect per image
                valid_object_ids (list): List of valid class IDs
            post_process:
                epsilon_correct_detection (int): Maximum pixel distance for a detection to be considered correct
        arch (str, optional): Backbone architecture name. Defaults to "res_18".
        learning_rate (float, optional): Initial learning rate. Defaults to 1e-4.
        learning_rate_milestones (list, optional): Epochs at which to decay learning rate. Defaults to None.
        hm_weight (float, optional): Weight for heatmap loss. Defaults to 1.
        off_weight (float, optional): Weight for offset regression loss. Defaults to 1.
    """

    def __init__(
        self,
        config,
        learning_rate=1e-4,
        learning_rate_milestones=None,
        hm_weight=1,
        off_weight=1,
    ):
        super().__init__(config["model"]["backbone"])

        # Load config values
        self.config = config
        self.mean = config["image"]["mean"]
        self.std = config["image"]["std"]
        self.img_aspect_ratio = config["image"]["aspect_ratio"]
        self.img_longer_side = config["image"]["longer_side"]
        self.score_threshold = config["detection"]["score_threshold"]
        self.max_objs = config["detection"]["max_objects"]
        self.valid_ids = config["detection"]["valid_object_ids"]
        self.eps_correct_det = config["post_process"]["epsilon_correct_detection"]

        self.num_classes = 1 # bee class only
        assert self.num_stacks == 1 # to ensure forward function specialization

        heads = {"heatmap": self.num_classes, "regression": 2}
        self.heads = torch.nn.ModuleList(
            [
                CenterHead(heads, self.backbone.out_channels, self.head_conv)
                for _ in range(self.num_stacks)
            ]
        )

        self.hm_weight = hm_weight
        self.off_weight = off_weight
        self.learning_rate_milestones = (
            learning_rate_milestones
            if learning_rate_milestones is not None
            else []
        )

        # Loss
        self.criterion = FocalLoss()
        self.criterion_regression = RegL1Loss()

        self.save_hyperparameters()

    def forward(self, x):
        outputs = self.backbone(x)
        # For the given backbone, we're sure there is only one stack
        # and can avoid issues when generating Torchscript:
        result = self.heads[0](outputs[0])
        return result

    @torch.jit.ignore
    def loss(self, outputs, target):
        hm_loss, off_loss = 0, 0
        num_stacks = len(outputs)

        for s in range(num_stacks):
            output = outputs[s]
            output["heatmap"] = sigmoid_clamped(output["heatmap"])

            hm_loss += self.criterion(output["heatmap"], target["heatmap"])
            off_loss += self.criterion_regression(
                output["regression"],
                target["regression_mask"],
                target["indices"],
                target["regression"],
            )

        loss = (
            self.hparams.hm_weight * hm_loss
            + self.hparams.off_weight * off_loss
        ) / num_stacks
        loss_stats = {
            "loss": loss,
            "hm_loss": hm_loss,
            "off_loss": off_loss,
        }
        return loss, loss_stats

    @torch.jit.ignore
    def training_step(self, batch, batch_idx):
        img, target = batch
        outputs = self(img)
        outputs_list = [outputs]
        loss, loss_stats = self.loss(outputs_list, target)

        self.log(f"train_loss", loss, on_epoch=True)

        for key, value in loss_stats.items():
            self.log(f"train/{key}", value)

        return loss

    @torch.jit.ignore
    def validation_step(self, batch, batch_idx):
        img, target = batch
        outputs = self(img)
        outputs_list = [outputs]
        loss, loss_stats = self.loss(outputs_list, target)
    
        detections = extract_detections(outputs_list[0], self.max_objs, self.down_ratio)
        self.report_metrics(img, detections, outputs_list[0]["heatmap"], 
                            target["original_pts"], target["regression_mask"])

        self.log(f"val_loss", loss, on_epoch=True, sync_dist=True)

        for name, value in loss_stats.items():
            self.log(f"val/{name}", value, on_epoch=True, sync_dist=True)

        return {"loss": loss, "loss_stats": loss_stats}
    
    @torch.jit.ignore
    def test_step(self, batch, batch_idx):
        img, target = batch
        outputs = self(img)
    
        detections = extract_detections(outputs, self.max_objs, self.down_ratio)
        self.report_metrics(img, detections, outputs["heatmap"], 
                            target["original_pts"], target["regression_mask"])

    @torch.jit.ignore
    def report_metrics(self, img, detections, heatmap, target_regression, target_regression_mask):
        """Calculate and log detection metrics for bee center detection.
        
        Computes precision, recall, AUC, localization error and bee count error metrics by comparing
        detected bee centers against ground truth annotations. Only considers detections within a 
        defined ROI and above a confidence threshold. Also generates visualization plots.
        
        Args:
            img (torch.Tensor): Input image batch with shape [B,C,H,W]
            detections (torch.Tensor): Detection results with shape [B,K,5], where K is max detections
                                     and format is [x,y,score,class,_]
            heatmap (torch.Tensor): Predicted center point heatmaps with shape [B,C,H,W]
            target_regression (torch.Tensor): Ground truth center point coordinates normalized to [0,1]
            target_regression_mask (torch.Tensor): Binary mask indicating valid ground truth points
            
        The function:
        1. Filters detections by confidence and ROI boundaries
        2. Matches detections to ground truth using distance threshold
        3. Computes metrics:
            - Precision: TP / (TP + FP)
            - Recall: TP / (TP + FN) 
            - AUC: Precision * Recall
            - Localization error: Mean distance between matched det-GT pairs
            - Bee count error: Absolute difference between det and GT counts
        4. Logs metrics to wandb and generates visualization plots
        """

        batch_precision = []
        batch_recall = []
        batch_auc = []
        batch_loc_error = []
        batch_beecount_error = []
        roi_x, roi_y = 0, 0
        roi_w, roi_h = 0, 0
        
        detections = detections.cpu()
        target_regression = target_regression.cpu()
        target_regression_mask = target_regression_mask.cpu()

        # Calculate metrics for each sample in batch
        for batch_idx in range(len(img)):
            # Get ROI boundaries for current image
            img_h, img_w = img[batch_idx].shape[1:]
            roi_h = int(self.img_longer_side * self.img_aspect_ratio)
            roi_w = self.img_longer_side
            roi_x = (img_w - roi_w) // 2
            roi_y = (img_h - roi_h) // 2
            
            # Get high confidence detections within ROI
            high_conf_dets = []
            for det in detections[batch_idx]:
                x, y = det[0].item(), det[1].item()
                score = det[2].item()
                
                if (score > self.score_threshold and
                    roi_x <= x < roi_x + roi_w and
                    roi_y <= y < roi_y + roi_h):
                    high_conf_dets.append((x, y))
            
            # Get ground truth points
            gt_points = []
            curr_target_regression = target_regression[batch_idx].numpy()
            curr_target_mask = target_regression_mask[batch_idx].numpy()
            
            for i in range(len(curr_target_mask)):
                if curr_target_mask[i]:
                    x, y = curr_target_regression[i]
                    x = x * img_w
                    y = y * img_h
                    gt_points.append((x, y))
            
            # Calculate metrics
            tp = 0
            loc_error, num_loc_error = 0, 0
            matched_dets = set()
            
            for gt_point in gt_points:
                min_dist = float('inf')
                best_det_idx = None
                
                for det_idx, det_point in enumerate(high_conf_dets):
                    if det_idx not in matched_dets:
                        dist = np.sqrt((gt_point[0] - det_point[0])**2 + 
                                       (gt_point[1] - det_point[1])**2)
                        if dist <= self.eps_correct_det and dist < min_dist:
                            min_dist = dist
                            best_det_idx = det_idx

                if min_dist <= self.eps_correct_det:
                    loc_error += min_dist
                    num_loc_error += 1

                if best_det_idx is not None:
                    tp += 1
                    matched_dets.add(best_det_idx)

            if num_loc_error > 0:
                loc_error /= num_loc_error
            else:
                loc_error = self.eps_correct_det
                
            # Calculate metrics for current sample
            precision = tp / len(high_conf_dets) if len(high_conf_dets) > 0 else 0
            recall = tp / len(gt_points) if len(gt_points) > 0 else 0
            auc = precision * recall
            beecount_error = np.abs(np.sum(curr_target_mask) - len(high_conf_dets))
            
            # Store metrics
            batch_precision.append(precision)
            batch_recall.append(recall)
            batch_auc.append(auc)
            batch_loc_error.append(loc_error)
            batch_beecount_error.append(beecount_error)
        
        # Log mean metrics across batch
        self.log("val/precision", np.mean(batch_precision), on_epoch=True, sync_dist=True)
        self.log("val/recall", np.mean(batch_recall), on_epoch=True, sync_dist=True)
        self.log("val/auc", np.mean(batch_auc), on_epoch=True, sync_dist=True)
        self.log("val/loc_error", np.mean(batch_loc_error), on_epoch=True, sync_dist=True)
        self.log("val/beecount_error", np.mean(batch_beecount_error), on_epoch=True, sync_dist=True)

        # Plot results for first image
        self.plot_detection_results(img, detections, heatmap, target_regression,
                                    target_regression_mask, 0)

    @torch.jit.ignore
    def plot_detection_results(self, img, detections, heatmap, target_regression,
                               target_regression_mask, img_idx):
        """Plot detection results for a single image
        
        Args:
            img (torch.Tensor): Input image batch [B,C,H,W]
            detections (torch.Tensor): Detection results [B,K,5]
            heatmap (torch.Tensor): Heatmap predictions [B,C,H,W]
            target_regression (torch.Tensor): Ground truth regression targets
            target_regression_mask (torch.Tensor): Mask indicating valid regression targets
            img_idx (int): Index of image to plot from batch
        """

        img_h, img_w = img[img_idx].shape[1:]
        roi_h = int(self.img_longer_side * self.img_aspect_ratio)
        roi_w = self.img_longer_side
        roi_x = (img_w - roi_w) // 2
        roi_y = (img_h - roi_h) // 2

        # Prepare image for plotting
        plot_img = img[img_idx].cpu()
        # Denormalize image
        for t, m, s in zip(plot_img, self.mean, self.std):
            t.mul_(s).add_(m)
        plot_img = (plot_img * 255).byte()
        plot_img = torchvision.transforms.ToPILImage()(plot_img)
        plot_img = np.array(plot_img)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot image with detections
        ax1.imshow(plot_img[:, :, ::-1])
        
        # Plot detections
        for det in detections[img_idx]:
            x, y = det[0], det[1]
            score = det[2]
            if (score > self.score_threshold and
                roi_x <= x < roi_x + roi_w and
                roi_y <= y < roi_y + roi_h):
                ax1.plot(x, y, 'ro', markersize=4)
                
        # Plot ground truth points
        target_regression_plot = target_regression[img_idx].cpu().numpy()
        target_mask_plot = target_regression_mask[img_idx].cpu().numpy()
        for i in range(len(target_mask_plot)):
            if target_mask_plot[i]:
                x, y = target_regression_plot[i]
                x = x * img_w
                y = y * img_h
                ax2.plot(x, y, 'go', markersize=4)
        
        # Plot heatmap
        plot_heatmap = heatmap[img_idx, 0].cpu().numpy()
        plot_heatmap = cv2.resize(plot_heatmap, (img_w, img_h))
        heatmap_masked = np.zeros_like(plot_heatmap)
        heatmap_masked[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] = plot_heatmap[roi_y:roi_y+roi_h,
                                                                            roi_x:roi_x+roi_w]
        
        # Clamp heatmap values
        max_confidence = self.config["visualization"]["heatmap_max"]
        heatmap_masked = max_confidence - np.clip(heatmap_masked, 0, max_confidence)
        ax2.imshow(heatmap_masked, cmap='hot', vmin=0, vmax=max_confidence)
        
        # Add legends and finalize plot
        ax1.legend()
        plt.tight_layout()
        
        # Log to wandb if logger supports it (for testing phase)
        if hasattr(self.logger.experiment, 'log'):
            self.logger.experiment.log({
                f"val_detections/val_img": wandb.Image(fig)
            })
        plt.close()