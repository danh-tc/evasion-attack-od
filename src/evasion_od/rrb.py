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


def apply_fixed_scale(img_chw: torch.Tensor, scale: float) -> torch.Tensor:
    """Phase S0 (plan.md "Scale Transfer Mechanism"): deterministic global
    scale, always content-preserving (shrink + letterbox-pad, never crop).

    A true content-preserving "zoom in" (scale>1) within a fixed canvas is
    geometrically impossible without cropping: enlarging past the canvas and
    not cropping is a contradiction (resizing back down to fit would just
    undo the enlargement). So both directions of the sweep are implemented
    as the same shrink-then-pad operation, using occupancy = min(scale,
    1/scale) as the actual shrink fraction -- e.g. scale=1.2 shrinks content
    to 1/1.2=83.3% of the canvas, scale=1.4 to 1/1.4=71.4%, matching
    scale=0.8/0.6's own shrink fractions in spirit (how much of the frame
    the original content occupies) without ever discarding a pixel. This
    replaces an earlier version that center-cropped for scale>1 (silently
    losing GT boxes near the border, confounding "scale" with "content
    loss" -- see plan.md Phase S0's rerun note).

    Unlike `adaptive_random_resizing` (scales relative to a GT box's own
    size, randomizes both magnitude and crop/pad offset every call), this is
    a single controlled variable -- no GT boxes, no randomness -- so a sweep
    over `scale` traces a transfer response curve per target instead of
    marginalizing over a random-resize distribution.
    """
    occupancy = scale if scale <= 1.0 else 1.0 / scale
    _, h, w = img_chw.shape
    new_h, new_w = max(1, round(h * occupancy)), max(1, round(w * occupancy))
    resized = F.interpolate(
        img_chw.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=True
    )
    pad_h, pad_w = h - new_h, w - new_w
    top, left = pad_h // 2, pad_w // 2
    out = F.pad(resized, (left, pad_w - left, top, pad_h - top), mode="constant", value=0.0)
    return out.squeeze(0)


def random_occupancy_resize(img_chw: torch.Tensor, occ_low: float, occ_high: float) -> torch.Tensor:
    """Phase S1 (plan.md "Bidirectional/Shrink-aware RRB"): samples a random
    content occupancy in [occ_low, occ_high] each call, shrinks to that
    fraction of the canvas, and pads at a random offset (same
    randomized-offset spirit as `adaptive_random_resizing`'s pad_left/
    pad_top) -- content-preserving, same math as `apply_fixed_scale`'s
    occupancy but randomized per call instead of one fixed value.

    Replaces `adaptive_random_resizing` as RRB's resize step: that function
    -- despite `s_max` suggesting "enlarge" -- nets out to shrink-then-pad
    too (it upscales to an intermediate canvas, pads that canvas to a fixed
    size, then downsamples the whole thing back to the original size), just
    over occupancy range [1/s_max, 1.0] ~ [0.91, 1.0] at s_max=1.1 -- too
    narrow/close to 1.0 to reach the occupancy~0.8 region Phase S0 found
    DINO-Swin-L specifically benefits from.
    """
    occ = random.uniform(occ_low, occ_high)
    _, h, w = img_chw.shape
    new_h, new_w = max(1, round(h * occ)), max(1, round(w * occ))
    resized = F.interpolate(
        img_chw.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=True
    )
    pad_h, pad_w = h - new_h, w - new_w
    top = random.randint(0, pad_h) if pad_h > 0 else 0
    left = random.randint(0, pad_w) if pad_w > 0 else 0
    out = F.pad(resized, (left, pad_w - left, top, pad_h - top), mode="constant", value=0.0)
    return out.squeeze(0)


def apply_rrb_occupancy(
    img_chw: torch.Tensor,
    gt_boxes_xyxy: torch.Tensor,
    occ_low: float,
    occ_high: float,
    theta: float = 7.0,
    l_s: int = 10,
    sigma: float = 6.0,
) -> torch.Tensor:
    """Phase S1: RRB with `random_occupancy_resize` in place of
    `adaptive_random_resizing` -- rotation and noise steps unchanged from
    `apply_rrb`, only the resize component's sampling range differs.
    """
    if gt_boxes_xyxy.numel() == 0:
        return additive_gaussian_noise(img_chw, sigma)
    out = random_axis_rotation(img_chw, gt_boxes_xyxy, theta=theta, l_s=l_s)
    out = random_occupancy_resize(out, occ_low, occ_high)
    out = additive_gaussian_noise(out, sigma=sigma)
    return out


def additive_gaussian_noise(img_chw: torch.Tensor, sigma: float = 6.0) -> torch.Tensor:
    """Despite this step's name in the original OSFD paper/code ("gaussian_blur",
    reference-repo/OSFD/attack/base/RRB.py:107), it is not a blur kernel --
    no convolution happens. It's i.i.d. per-pixel additive Gaussian noise,
    named accordingly here (plan.md Phase B1 ablates it as a standalone
    component, so calling it "blur" would misdescribe what's being ablated).
    """
    return torch.clamp(img_chw + torch.randn_like(img_chw) * sigma, 0.0, 255.0)


def apply_resize_only(
    img_chw: torch.Tensor, gt_boxes_xyxy: torch.Tensor, rho: float = 0.8, s_max: float = 1.1
) -> torch.Tensor:
    """Phase B1 ablation component: adaptive resize alone, no rotation/noise.
    Identity (not noise-fallback like `apply_rrb`) when there's no GT box to
    scale relative to, so this stays a clean "resize-only" data point instead
    of silently reintroducing a different component for that image.
    """
    if gt_boxes_xyxy.numel() == 0:
        return img_chw
    return adaptive_random_resizing(img_chw, gt_boxes_xyxy, rho=rho, s_max=s_max)


def apply_rotation_resize(
    img_chw: torch.Tensor,
    gt_boxes_xyxy: torch.Tensor,
    theta: float = 7.0,
    l_s: int = 10,
    rho: float = 0.8,
    s_max: float = 1.1,
) -> torch.Tensor:
    """Phase B1 ablation component: rotation + resize, no noise."""
    out = random_axis_rotation(img_chw, gt_boxes_xyxy, theta=theta, l_s=l_s)
    return apply_resize_only(out, gt_boxes_xyxy, rho=rho, s_max=s_max)


def apply_rrb(
    img_chw: torch.Tensor,
    gt_boxes_xyxy: torch.Tensor,
    theta: float = 7.0,
    l_s: int = 10,
    rho: float = 0.8,
    s_max: float = 1.1,
    sigma: float = 6.0,
) -> torch.Tensor:
    """Rotation + resizing (both applied, order matches reference RRB intent) + noise.

    Falls back to noise-only if the image has no GT boxes.
    """
    if gt_boxes_xyxy.numel() == 0:
        return additive_gaussian_noise(img_chw, sigma)
    out = random_axis_rotation(img_chw, gt_boxes_xyxy, theta=theta, l_s=l_s)
    out = adaptive_random_resizing(out, gt_boxes_xyxy, rho=rho, s_max=s_max)
    out = additive_gaussian_noise(out, sigma=sigma)
    return out
