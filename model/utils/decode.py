"""
Code taken from: https://github.com/tteepe/CenterNet-pytorch-lightning/tree/main
"""

import torch
import torch.nn as nn


def ctdet_decode(heat, reg, K=100):
    """Custom function: decodes heatmap and regression predictions into detections.

    Args:
        heat (torch.Tensor): Heatmap predictions of shape (batch, num_classes, height, width)
        reg (torch.Tensor): Regression predictions of shape (batch, 2, height, width)
        K (int): Maximum number of detections to return per image

    Returns:
        torch.Tensor: Detections tensor of shape (batch, K, 5) where each detection is
            [x, y, score, class] and K is the max number of objects per image
    """
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


def extract_detections(output, max_objs, down_ratio):
    detections = ctdet_decode(
        output["heatmap"].sigmoid_(),
        output["regression"],
        K=max_objs,
    )
    detections[:, :, :2] *= down_ratio  # Scale to input

    return detections


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def _topk_channel(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    return topk_scores, topk_inds, topk_ys, topk_xs


def sigmoid_clamped(x, clamp=1e-4):
    y = torch.clamp(x.sigmoid_(), min=clamp, max=1 - clamp)
    return y


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat