"""MI-FGSM + backbone-feature-distortion attack, with optional RaPA masking.

Implements the E1-E8/I4 rows of plan.md's experiment table plus the
REL_*/HYBRID_*/SPATIAL_* rows (idea.txt's object-context hypothesis: Phase 1
in results/phase0_diagnostic.json, Phase 2 mean-pooled relational loss in
results/P2_*_go.json, Phase 1b/2-followup dense per-pixel spatial
misalignment loss in results/P1b_prototype_diagnostic.json): the loss (NRDM
k=1 / OSFD k=3 / relational contrast / OSFD+relational hybrid / spatial
misalignment), masking (RaPA DropConnect on backbone BN affine), and
input-level augmentation (none/RRB/DIM/SSA/SRS) axes are all independent
AttackConfig flags so any config is just a different set of flags through
this one loop.
"""

from __future__ import annotations

import numpy as np
import torch

from evasion_od.config import AttackConfig
from evasion_od.dim import apply_dim
from evasion_od.losses import (
    backbone_feature_loss,
    osfd_rel_hybrid_loss,
    relational_contrast_loss,
    spatial_misalignment_loss,
)
from evasion_od.masking import add_dropconnect_hooks, remove_dropconnect_hooks
from evasion_od.models import backbone_features
from evasion_od.preprocessing import (
    build_adversarial_image,
    build_resized_input,
    preprocess_batch,
    scale_gt_boxes,
)
from evasion_od.regions import precompute_relational_targets, precompute_spatial_targets
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
    gt_resized = scale_gt_boxes(gt_boxes_xyxy, resized.data_sample, device)

    with torch.no_grad():
        clean_batch = preprocess_batch(model, clean_chw, resized.data_sample)
        feats_clean = tuple(
            f.detach() for f in backbone_features(model, clean_batch["inputs"])
        )
        rel_targets = None
        spatial_targets = None
        if cfg.loss_type in ("rel", "osfd_rel_hybrid"):
            img_h, img_w = clean_batch["inputs"].shape[-2:]
            rel_targets = precompute_relational_targets(
                feats_clean, gt_resized, img_h, img_w, cfg.rel_r, cfg.rel_min_margin_cells
            )
        elif cfg.loss_type == "spatial":
            img_h, img_w = clean_batch["inputs"].shape[-2:]
            spatial_targets = precompute_spatial_targets(
                feats_clean, gt_resized, img_h, img_w, cfg.rel_r, cfg.rel_min_margin_cells
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
                if cfg.loss_type == "osfd":
                    loss = backbone_feature_loss(feats_adv, feats_clean, cfg.k)
                elif cfg.loss_type == "rel":
                    # Loop below does gradient ASCENT; relational_contrast_loss
                    # is meant to be MINIMIZED (push C_adv opposite C_clean),
                    # so ascend its negation -- see losses.py docstring.
                    loss = -relational_contrast_loss(feats_adv, rel_targets, cfg.rel_stage_weights)
                elif cfg.loss_type == "osfd_rel_hybrid":
                    # Already ascent-oriented as a whole -- see losses.py docstring.
                    loss = osfd_rel_hybrid_loss(
                        feats_adv, feats_clean, cfg.k, rel_targets, cfg.rel_stage_weights, cfg.rel_lambda
                    )
                elif cfg.loss_type == "spatial":
                    # Loop below does gradient ASCENT; spatial_misalignment_loss
                    # is meant to be MINIMIZED, so ascend its negation -- see
                    # losses.py docstring.
                    loss = -spatial_misalignment_loss(
                        feats_adv, spatial_targets, cfg.spatial_stage_weights
                    )
                else:
                    raise ValueError(f"unknown loss_type: {cfg.loss_type!r}")
                if loss.requires_grad:
                    (grad,) = torch.autograd.grad(loss, delta)
                else:
                    # "rel"/"spatial" loss with every weighted stage skipped
                    # (empty GT, or O/V mask empty at every weighted stage's
                    # resolution -- see regions.py:precompute_relational_targets
                    # /precompute_spatial_targets): no gradient signal exists
                    # for this image, so leave delta unperturbed rather than
                    # crash on a disconnected loss.
                    grad = torch.zeros_like(delta)
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
