"""DIM augmentation (Diverse Input Method, Xie et al. CVPR 2019).

Classification-attack baseline adapted for detection: random resize within
`scale_range`, pad back to the original size, applied stochastically. See
new-plan.txt Sec 5.2.C. Operates on a single (C,H,W) tensor in raw pixel
scale [0,255], matching rrb.py's convention.
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F


def apply_dim(
    img_chw: torch.Tensor,
    p: float = 0.7,
    scale_range: tuple[float, float] = (0.9, 1.1),
) -> torch.Tensor:
    if random.random() > p:
        return img_chw

    _, ori_h, ori_w = img_chw.shape
    scale = random.uniform(*scale_range)
    new_h = max(1, round(ori_h * scale))
    new_w = max(1, round(ori_w * scale))

    resized = F.interpolate(
        img_chw.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=True
    ).squeeze(0)

    rem_h = ori_h - new_h
    rem_w = ori_w - new_w
    if rem_h >= 0 and rem_w >= 0:
        pad_top = random.randint(0, rem_h)
        pad_left = random.randint(0, rem_w)
        return F.pad(
            resized,
            (pad_left, rem_w - pad_left, pad_top, rem_h - pad_top),
            mode="constant",
            value=0.0,
        )

    # scale > 1: resized is larger than the original in at least one dim --
    # random-crop back down to (ori_h, ori_w) instead of padding.
    crop_top = random.randint(0, max(0, new_h - ori_h))
    crop_left = random.randint(0, max(0, new_w - ori_w))
    return resized[:, crop_top : crop_top + ori_h, crop_left : crop_left + ori_w]
