"""RRB augmentation (Rotation + adaptive Resizing + Blur), from OSFD.

Adapted from reference-repo/OSFD/attack/base/RRB.py for a single (C,H,W)
tensor (no batch dim) in raw pixel scale [0,255], operating at attack
resolution. `gt_boxes_xyxy` must already be scaled to that resolution.

Only used by E2 (OSFD + RRB); E1/E3/E4/E5 run with `use_rrb=False`.
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import rotate


def random_axis_rotation(
    img_chw: torch.Tensor, gt_boxes_xyxy: torch.Tensor, theta: float = 7.0, l_s: int = 10
) -> torch.Tensor:
    device = img_chw.device
    _, h, w = img_chw.shape
    centers = (gt_boxes_xyxy[:, :2] + gt_boxes_xyxy[:, 2:]) / 2
    centers = torch.cat(
        [centers, torch.tensor([[w // 2, h // 2]], device=device, dtype=centers.dtype)], dim=0
    )
    if l_s > 0:
        centers = centers + torch.randint_like(centers, low=-l_s, high=l_s)
    cx, cy = centers[random.randrange(len(centers))]
    angle = random.random() * 2 * theta - theta
    return rotate(img_chw.unsqueeze(0), angle, center=[int(cx), int(cy)]).squeeze(0)


def adaptive_random_resizing(
    img_chw: torch.Tensor,
    gt_boxes_xyxy: torch.Tensor,
    rho: float = 0.8,
    s_max: float = 1.1,
) -> torch.Tensor:
    _, ori_h, ori_w = img_chw.shape
    box = gt_boxes_xyxy[random.randrange(len(gt_boxes_xyxy))]
    box_w = float(box[2] - box[0])
    box_h = float(box[3] - box[1])

    scale_h = min(1 + rho * (box_h / ori_h), s_max)
    scale_w = min(1 + rho * (box_w / ori_w), s_max)
    new_h = random.randint(ori_h, int(scale_h * ori_h))
    new_w = random.randint(ori_w, int(scale_w * ori_w))

    rescaled = F.interpolate(
        img_chw.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=True
    )
    rem_h = int(scale_h * ori_h) - new_h
    rem_w = int(scale_w * ori_w) - new_w
    pad_left = random.randint(0, rem_w)
    pad_top = random.randint(0, rem_h)
    padded = F.pad(
        rescaled, (pad_left, rem_w - pad_left, pad_top, rem_h - pad_top), mode="constant", value=0.0
    )
    return F.interpolate(
        padded, size=(ori_h, ori_w), mode="bilinear", align_corners=True
    ).squeeze(0)


def gaussian_blur(img_chw: torch.Tensor, sigma: float = 6.0) -> torch.Tensor:
    return torch.clamp(img_chw + torch.randn_like(img_chw) * sigma, 0.0, 255.0)


def apply_rrb(
    img_chw: torch.Tensor,
    gt_boxes_xyxy: torch.Tensor,
    theta: float = 7.0,
    l_s: int = 10,
    rho: float = 0.8,
    s_max: float = 1.1,
    sigma: float = 6.0,
) -> torch.Tensor:
    """Rotation + resizing (both applied, order matches reference RRB intent) + blur.

    Falls back to blur-only if the image has no GT boxes.
    """
    if gt_boxes_xyxy.numel() == 0:
        return gaussian_blur(img_chw, sigma)
    out = random_axis_rotation(img_chw, gt_boxes_xyxy, theta=theta, l_s=l_s)
    out = adaptive_random_resizing(out, gt_boxes_xyxy, rho=rho, s_max=s_max)
    out = gaussian_blur(out, sigma=sigma)
    return out
