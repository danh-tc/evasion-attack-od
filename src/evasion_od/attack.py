"""MI-FGSM + backbone-feature-distortion attack, with optional RaPA masking.

Implements the E1-E8/I4 rows of plan.md's experiment table: the loss (NRDM
k=1 / OSFD k=3), masking (RaPA DropConnect on backbone BN affine), and
input-level augmentation (none/RRB/DIM/SSA/SRS) axes are all independent
AttackConfig flags so any config is just a different set of flags through
this one loop.
"""

from __future__ import annotations

import numpy as np
import torch

from evasion_od.config import AttackConfig
from evasion_od.dim import apply_dim
from evasion_od.losses import backbone_feature_loss
from evasion_od.masking import add_dropconnect_hooks, remove_dropconnect_hooks
from evasion_od.models import backbone_features
from evasion_od.preprocessing import build_adversarial_image, build_resized_input, preprocess_batch
from evasion_od.rrb import apply_rrb
from evasion_od.srs import apply_rrb_spectral, apply_srs
from evasion_od.ssa import apply_ssa


def _apply_augmentation(
    kind: str, adv: torch.Tensor, gt_resized: torch.Tensor
) -> torch.Tensor:
    if kind == "none":
        return adv
    if kind == "rrb":
        return apply_rrb(adv, gt_resized)
    if kind == "dim":
        return apply_dim(adv)
    if kind == "ssa":
        return apply_ssa(adv)
    if kind == "srs":
        return apply_srs(adv, gt_resized)
    if kind == "rrb_spectral":
        return apply_rrb_spectral(adv, gt_resized)
    raise ValueError(f"unknown augmentation: {kind!r}")


def _scaled_gt_boxes(gt_boxes_xyxy: np.ndarray, data_sample, device) -> torch.Tensor:
    if len(gt_boxes_xyxy) == 0:
        return torch.zeros((0, 4), device=device)
    sx, sy = data_sample.metainfo["scale_factor"]
    scale = torch.tensor([sx, sy, sx, sy], device=device, dtype=torch.float32)
    return torch.as_tensor(gt_boxes_xyxy, device=device, dtype=torch.float32) * scale


def run_attack(
    model,
    img_bgr_uint8: np.ndarray,
    image_id: int,
    gt_boxes_xyxy: np.ndarray,
    cfg: AttackConfig,
    device: str = "cuda:0",
) -> np.ndarray:
    """Returns the full-resolution adversarial image (HWC BGR uint8)."""
    resized = build_resized_input(model, img_bgr_uint8, image_id, device)
    clean_chw = resized.clean_chw  # (C,H,W), no grad, raw pixel scale
    gt_resized = _scaled_gt_boxes(gt_boxes_xyxy, resized.data_sample, device)

    with torch.no_grad():
        clean_batch = preprocess_batch(model, clean_chw, resized.data_sample)
        feats_clean = tuple(
            f.detach() for f in backbone_features(model, clean_batch["inputs"])
        )

    mask_handles = None
    if cfg.mask_enabled:
        mask_handles = add_dropconnect_hooks(model.backbone, cfg.drop_prob, cfg.mask_layer_types)

    try:
        delta = torch.zeros_like(clean_chw)
        momentum = torch.zeros_like(clean_chw)

        for _ in range(cfg.max_iterations):
            delta = delta.clone().detach().requires_grad_(True)
            grads = []
            for _ in range(cfg.num_masks):
                adv = torch.clamp(clean_chw + delta, 0.0, 255.0)
                adv = _apply_augmentation(cfg.augmentation, adv, gt_resized)
                batch = preprocess_batch(model, adv, resized.data_sample)
                feats_adv = backbone_features(model, batch["inputs"])
                loss = backbone_feature_loss(feats_adv, feats_clean, cfg.k)
                (grad,) = torch.autograd.grad(loss, delta)
                grads.append(grad)

            grad_avg = torch.stack(grads, dim=0).mean(dim=0)
            momentum = cfg.momentum * momentum + grad_avg / (grad_avg.abs().mean() + 1e-12)

            delta = delta.detach() + cfg.alpha * momentum.sign()
            delta = torch.clamp(delta, -cfg.epsilon, cfg.epsilon)
            delta = torch.clamp(clean_chw + delta, 0.0, 255.0) - clean_chw
    finally:
        if mask_handles is not None:
            remove_dropconnect_hooks(mask_handles)

    return build_adversarial_image(img_bgr_uint8, delta.detach(), resized.ori_shape)
