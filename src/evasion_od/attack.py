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
this one loop. Also implements the RRB_K5_* rows (Phase G0 "RRB Gradient
Mechanism Analysis", plan.md): `grad_combine` picks how the K per-view
gradients get combined into one update direction each iteration -- see
gradient_diagnostics.py.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from evasion_od.config import AttackConfig
from evasion_od.dim import apply_dim
from evasion_od.gradient_diagnostics import CheckpointStats, combine_gradients, compute_checkpoint_stats
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
from evasion_od.rrb import (
    additive_gaussian_noise,
    apply_fixed_scale,
    apply_resize_only,
    apply_rotation_resize,
    apply_rrb,
    apply_rrb_occupancy,
    random_axis_rotation,
    random_occupancy_resize,
)
from evasion_od.srs import apply_rrb_spectral, apply_srs
from evasion_od.ssa import apply_ssa


def _apply_augmentation(
    kind: str,
    adv: torch.Tensor,
    gt_resized: torch.Tensor,
    fixed_scale: float = 1.0,
    occ_low: float = 0.7,
    occ_high: float = 1.0,
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
    # Phase B1 (plan.md "RRB Component Ablation"): isolated/partial RRB
    # components, to attribute E1->E2's transfer gain to rotation, resize,
    # noise, or their interaction.
    if kind == "rrb_rot":
        return random_axis_rotation(adv, gt_resized)
    if kind == "rrb_resize":
        return apply_resize_only(adv, gt_resized)
    if kind == "rrb_noise":
        return additive_gaussian_noise(adv)
    if kind == "rrb_rot_resize":
        return apply_rotation_resize(adv, gt_resized)
    # Phase S0 (plan.md "Scale Transfer Mechanism"): deterministic global
    # scale sweep, isolated from rotation/noise/GT-relative randomization.
    if kind == "fixed_scale":
        return apply_fixed_scale(adv, fixed_scale)
    # Phase S1 (plan.md "Bidirectional/Shrink-aware RRB"): full RRB
    # (rotation+resize+noise) with `random_occupancy_resize` in place of
    # `adaptive_random_resizing`.
    if kind == "rrb_occupancy":
        return apply_rrb_occupancy(adv, gt_resized, occ_low, occ_high)
    # Phase S2 (plan.md "Trajectory Consistency"): resize-only (no
    # rotation/noise, matching S0's isolation), reconciling S0 (one fixed
    # scale held for the whole trajectory) vs S1 (resampled every
    # iteration) -- isolates exactly that one variable.
    if kind == "random_shrink":
        return random_occupancy_resize(adv, occ_low, occ_high)
    if kind == "fixed_shrink_per_image":
        # `fixed_scale` here is the per-image value run_attack() drew once
        # before the iteration loop (see below), not AttackConfig.fixed_scale.
        return apply_fixed_scale(adv, fixed_scale)
    raise ValueError(f"unknown augmentation: {kind!r}")


def run_attack(
    model,
    img_bgr_uint8: np.ndarray,
    image_id: int,
    gt_boxes_xyxy: np.ndarray,
    cfg: AttackConfig,
    device: str = "cuda:0",
    diagnostic_iters: frozenset[int] = frozenset(),
    diagnostic_out: list[CheckpointStats] | None = None,
) -> np.ndarray:
    """Returns the full-resolution adversarial image (HWC BGR uint8).

    `diagnostic_iters`/`diagnostic_out`: Phase G0 only (plan.md). When the
    current iteration index is in `diagnostic_iters`, appends a
    `CheckpointStats` (gradient_diagnostics.py) computed from that
    iteration's K per-view gradients to `diagnostic_out`, before they're
    combined into the update direction -- measured on this exact trajectory,
    not a separate pass, so it stays consistent with this image's own final
    delta/ASR. No-op (default) for every non-G0 experiment.
    """
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

    # Phase S2 (plan.md "Trajectory Consistency"): "fixed_shrink_per_image"
    # draws one occupancy value per image (constant across that image's
    # whole T-iteration trajectory), unlike AttackConfig.fixed_scale (Phase
    # S0's "fixed_scale" kind, constant across every image in the run too).
    effective_fixed_scale = cfg.fixed_scale
    if cfg.augmentation == "fixed_shrink_per_image":
        effective_fixed_scale = random.uniform(cfg.occ_low, cfg.occ_high)

    try:
        delta = torch.zeros_like(clean_chw)
        momentum = torch.zeros_like(clean_chw)

        for it in range(cfg.max_iterations):
            delta = delta.clone().detach().requires_grad_(True)
            grads = []
            for k_idx in range(cfg.num_masks):
                if cfg.deterministic_augmentation:
                    view_seed = (
                        cfg.seed * 1_000_003 + image_id * 10_007 + it * 101 + k_idx
                    ) % 2_147_483_647
                    random.seed(view_seed)
                    torch.manual_seed(view_seed)
                adv = torch.clamp(clean_chw + delta, 0.0, 255.0)
                adv = _apply_augmentation(
                    cfg.augmentation, adv, gt_resized, effective_fixed_scale, cfg.occ_low, cfg.occ_high
                )
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

            if it in diagnostic_iters and diagnostic_out is not None:
                diagnostic_out.append(compute_checkpoint_stats(grads, it))

            grad_avg = combine_gradients(grads, cfg.grad_combine, cfg.grad_combine_gamma)
            momentum = cfg.momentum * momentum + grad_avg / (grad_avg.abs().mean() + 1e-12)

            delta = delta.detach() + cfg.alpha * momentum.sign()
            delta = torch.clamp(delta, -cfg.epsilon, cfg.epsilon)
            delta = torch.clamp(clean_chw + delta, 0.0, 255.0) - clean_chw
    finally:
        if mask_handles is not None:
            remove_dropconnect_hooks(mask_handles)

    return build_adversarial_image(img_bgr_uint8, delta.detach(), resized.ori_shape)
