from argparse import ArgumentParser
import os

from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torchvision.datasets import CocoDetection
import cv2
import numpy as np
import imgaug as ia
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torchvision
import wandb

from bee_augmentations import get_transforms
from CenterNet import CenterNet
from CenterNet.models.heads import CenterHead
from CenterNet.utils.decode import _nms, _topk, _transpose_and_gather_feat, sigmoid_clamped
from CenterNet.utils.losses import RegL1Loss, FocalLoss


#os.environ['WANDB_DISABLED'] = 'true'

def ctdet_decode(heat, reg, K=100):
    batch, _, _, _ = heat.size()

    # perform nms on heatmaps
    heat = _nms(heat)

    scores, inds, clses, ys, xs = _topk(heat, K=K)
    reg = _transpose_and_gather_feat(reg, inds)
    reg = reg.view(batch, K, 2)
    xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
    ys = ys.view(batch, K, 1) + reg[:, :, 1:2]

    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    centers = torch.cat([xs, ys], dim=2)
    detections = torch.cat([centers, scores, clses], dim=2)

    return detections


class CenterNetBeeCenter(CenterNet):
    mean = [0.408, 0.447, 0.470]
    std = [0.289, 0.274, 0.278]
    img_aspect_ratio = 3. / 4
    img_longer_side = 500
    score_threshold = 0.51
    max_objs = 100
    valid_ids = [1]
    eps_correct_det = 8 # in pxs

    def __init__(
        self,
        arch="res_18",
        learning_rate=1e-4,
        learning_rate_milestones=None,
        hm_weight=1,
        off_weight=1,
        test_coco=None,
        test_coco_ids=None,
        test_scales=None,
        test_flip=False,
    ):
        super().__init__(arch)

        self.num_classes = 1 # bee class only
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

        # Test
        self.test_coco = test_coco
        self.test_coco_ids = test_coco_ids
        self.test_max_per_image = 100
        self.test_scales = [1] if test_scales is None else test_scales
        self.test_flip = test_flip

        # Loss
        self.criterion = FocalLoss()
        self.criterion_regression = RegL1Loss()

        self.save_hyperparameters()

    def forward(self, x):
        outputs = self.backbone(x)

        rets = []
        for head, output in zip(self.heads, outputs):
            rets.append(head(output))

        return rets

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

    def validation_step(self, batch, batch_idx):
        img, target = batch
        outputs = self(img)
        loss, loss_stats = self.loss(outputs, target)
    
        detections = self.calculate_dets(outputs[0])
        self.report_metrics(img, detections, outputs[0]["heatmap"], 
                            target["original_pts"], target["regression_mask"])

        self.log(f"val_loss", loss, on_epoch=True, sync_dist=True)

        for name, value in loss_stats.items():
            self.log(f"val/{name}", value, on_epoch=True, sync_dist=True)

        return {"loss": loss, "loss_stats": loss_stats}

    def calculate_dets(self, output):
        detections = ctdet_decode(
            output["heatmap"].sigmoid_(),
            output["regression"],
            K=self.max_objs,
        )
        detections = detections.cpu().detach().squeeze()

        detections[:, :, :2] *= self.down_ratio  # Scale to input

        return detections

    def report_metrics(self, img, detections, heatmap, target_regression, target_regression_mask):
        """Plot detected bee centers and heatmap on the input image and log to wandb
        
        Args:
            img (torch.Tensor): Input image batch [B,C,H,W]
            detections (torch.Tensor): Detection results [B,K,5] with format [x,y,score,class]
            heatmap (torch.Tensor): Heatmap predictions [B,C,H,W]
            target_regression (torch.Tensor): Ground truth regression targets
            target_regression_mask (torch.Tensor): Mask indicating valid regression targets
        """
        batch_precision = []
        batch_recall = []
        batch_auc = []
        batch_loc_error = []
        batch_beecount_error = []
        roi_x, roi_y = 0, 0
        roi_w, roi_h = 0, 0
        
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
                x, y = det[0], det[1]
                score = det[2]
                
                if (score > self.score_threshold and
                    roi_x <= x < roi_x + roi_w and
                    roi_y <= y < roi_y + roi_h):
                    high_conf_dets.append((x, y))
            
            # Get ground truth points
            gt_points = []
            curr_target_regression = target_regression[batch_idx].cpu().numpy()
            curr_target_mask = target_regression_mask[batch_idx].cpu().numpy()
            
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
            
            for gt_idx, gt_point in enumerate(gt_points):
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

        # Plot results for first image only
        first_img = img[0].cpu()
        # Denormalize image
        for t, m, s in zip(first_img, self.mean, self.std):
            t.mul_(s).add_(m)
        first_img = (first_img * 255).byte()
        first_img = torchvision.transforms.ToPILImage()(first_img)
        first_img = np.array(first_img)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot first image with detections
        ax1.imshow(first_img[:, :, ::-1])
        
        # Plot detections from first image
        for det in detections[0]:
            x, y = det[0], det[1]
            score = det[2]
            if (score > self.score_threshold and
                roi_x <= x < roi_x + roi_w and
                roi_y <= y < roi_y + roi_h):
                ax1.plot(x, y, 'ro', markersize=4)
                
        # Plot ground truth points from first image
        first_target_regression = target_regression[0].cpu().numpy()
        first_target_mask = target_regression_mask[0].cpu().numpy()
        for i in range(len(first_target_mask)):
            if first_target_mask[i]:
                x, y = first_target_regression[i]
                x = x * img_w
                y = y * img_h
                ax2.plot(x, y, 'go', markersize=4)
        
        # Plot heatmap for first image
        first_heatmap = heatmap[0, 0].cpu().numpy()
        first_heatmap = cv2.resize(first_heatmap, (img_w, img_h))
        heatmap_masked = np.zeros_like(first_heatmap)
        heatmap_masked[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] = first_heatmap[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        
        # Clamp heatmap values
        heatmap_masked = 0.55 - np.clip(heatmap_masked, 0, 0.55)
        ax2.imshow(heatmap_masked, cmap='hot', vmin=0, vmax=0.55)
        
        # Add legends and finalize plot
        ax1.legend()
        plt.tight_layout()
        
        # Log to wandb
        self.logger.experiment.log({
            f"val_detections/val_img": wandb.Image(fig)
        })
        plt.close()

def cli_main():
    pl.seed_everything(5318008)
    ia.seed(107734)

    train_transform, valid_transform, test_transform = get_transforms(
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

    coco_test = CocoDetection(
        os.path.join(args.dataset_root, "test", "img"),
        os.path.join(args.dataset_root, "test", "test_coco.json"),
        transforms=test_transform,
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
    test_loader = DataLoader(
        coco_test,
        batch_size=1, 
        num_workers=0,
        pin_memory=True,
    )

    # ------------
    # model
    # ------------
    args.learning_rate_milestones = list(map(int, args.learning_rate_milestones.split(",")))
    model = CenterNetBeeCenter(
        "res_18", args.learning_rate,
        args.learning_rate_milestones,
        test_coco=coco_test.coco,
        test_coco_ids=list(sorted(coco_test.coco.imgs.keys()))
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
            filename="epoch_{epoch:02d}-val_loss_{val_loss:.2f}"  # Added filename pattern            
        ),
        LearningRateMonitor(logging_interval="epoch"),
        #VisualizationCallback(),
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


if __name__ == "__main__":
    cli_main()
