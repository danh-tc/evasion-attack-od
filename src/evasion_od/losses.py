"""Backbone-feature distortion loss (OSFD), with NRDM as the k=1 special case.

    L = sum_i (1/N_i) * sum_j ( F_adv[i][j] - k * F_clean[i][j] )^2

Maximizing this loss suppresses the significant (object) features and, for
k>1, amplifies the vicinal (boundary/background) features -- see OSFD paper
Eq. 2 and Fig. 3. k=1 reduces to NRDM (plain feature-distance maximization,
no suppress/amplify asymmetry).

Attack loops should compute gradients as `grad(loss, delta)` and step in the
`+sign(grad)` direction (gradient *ascent* on this loss).
"""

from __future__ import annotations

import torch

from evasion_od.regions import (
    pixelwise_cosine_margin,
    region_score_normalized_energy,
    relational_contrast,
    relational_diff,
)


def backbone_feature_loss(
    feats_adv: tuple[torch.Tensor, ...], feats_clean: tuple[torch.Tensor, ...], k: float
) -> torch.Tensor:
    if len(feats_adv) != len(feats_clean):
        raise ValueError(
            f"stage count mismatch: adv={len(feats_adv)} clean={len(feats_clean)}"
        )
    loss = feats_adv[0].new_zeros(())
    for f_adv, f_clean in zip(feats_adv, feats_clean):
        loss = loss + torch.mean((f_adv - k * f_clean) ** 2)
    return loss


def relational_contrast_loss(
    feats_adv: tuple[torch.Tensor, ...],
    targets: list[dict | None],
    stage_weights: tuple[float, ...],
) -> torch.Tensor:
    """L = sum_l w_l * C_l_clean * C_l_adv (idea.txt's object-context
    relational hypothesis, Phase 0-validated in results/phase0_diagnostic.json
    -- see regions.py:precompute_relational_targets for C_clean/mask/mu/sigma).

    This is meant to be MINIMIZED: whatever sign C_clean has on a clean
    image, pushing C_adv to the opposite sign drives the product negative.
    attack.py's MI-FGSM loop does gradient *ascent*, so callers must pass
    `-relational_contrast_loss(...)` -- ascending the negation is descending
    this loss. Stages with w_l==0 or targets[l] is None (empty GT/O/V at
    that stage) are skipped.
    """
    loss = feats_adv[0].new_zeros(())
    for feat, target, w in zip(feats_adv, targets, stage_weights):
        if target is None or w == 0:
            continue
        f = feat[0]
        s_o = region_score_normalized_energy(f, target["o_mask"], target["mu"], target["sigma"])
        s_v = region_score_normalized_energy(f, target["v_mask"], target["mu"], target["sigma"])
        c_adv = relational_contrast(s_o, s_v)
        loss = loss + w * target["c_clean"] * c_adv
    return loss


def relational_diff_loss(
    feats_adv: tuple[torch.Tensor, ...],
    targets: list[dict | None],
    stage_weights: tuple[float, ...],
) -> torch.Tensor:
    """Same as relational_contrast_loss but on the UNBOUNDED D = S_O - S_V
    instead of the bounded contrast C -- Phase 2 (results/P2_rel_*_go.json)
    showed the bounded C saturates within ~20/100 MI-FGSM iterations, wasting
    most of the attack budget; D keeps producing gradient throughout. Used
    standalone or as the regularizer term in osfd_rel_hybrid_loss. Also
    meant to be MINIMIZED (negate before ascending)."""
    loss = feats_adv[0].new_zeros(())
    for feat, target, w in zip(feats_adv, targets, stage_weights):
        if target is None or w == 0:
            continue
        f = feat[0]
        s_o = region_score_normalized_energy(f, target["o_mask"], target["mu"], target["sigma"])
        s_v = region_score_normalized_energy(f, target["v_mask"], target["mu"], target["sigma"])
        d_adv = relational_diff(s_o, s_v)
        loss = loss + w * target["d_clean"] * d_adv
    return loss


def osfd_rel_hybrid_loss(
    feats_adv: tuple[torch.Tensor, ...],
    feats_clean: tuple[torch.Tensor, ...],
    k: float,
    targets: list[dict | None],
    stage_weights: tuple[float, ...],
    lam: float,
) -> torch.Tensor:
    """L_OSFD - lambda * L_relD, meant to be ASCENDED directly (no extra
    negation at the call site): ascending increases L_OSFD (OSFD's own
    convention) and decreases L_relD (relational_diff_loss's MINIMIZE
    convention), matching both sub-losses' intents in one pass.

    lambda must be calibrated against L_OSFD's scale, not assumed O(1) --
    measured on a sample image, |L_OSFD| / |L_relD| was ~50-85x, so e.g.
    lambda=1.0 makes the relational term negligible; use lambda~10-100 for
    it to meaningfully influence the gradient.
    """
    return backbone_feature_loss(feats_adv, feats_clean, k) - lam * relational_diff_loss(
        feats_adv, targets, stage_weights
    )


def spatial_misalignment_loss(
    feats_adv: tuple[torch.Tensor, ...],
    targets: list[dict | None],
    stage_weights: tuple[float, ...],
) -> torch.Tensor:
    """L = sum_l w_l * mean_{p in O_l} [cos(F_adv(p), mu_O_l) - cos(F_adv(p), mu_V_l)]

    Phase 1b's object-to-context feature MISALIGNMENT loss (Phase 2's
    mean-pooled S_O/S_V, results/P2_rel_*_go.json + P2_hybrid_*_go.json, lost
    to plain OSFD on 7/7 models -- too coarse a gradient). This is dense per
    -pixel instead: every object-region pixel gets pushed away from its own
    region's clean prototype and toward the vicinity's, individually.
    Validated on clean images in results/P1b_prototype_diagnostic.json
    (P(margin>0)=1.000 on nearly every model/stage, including YOLOX stage 2
    where relational_contrast/relational_diff had reversed).

    Meant to be MINIMIZED. attack.py's MI-FGSM loop does gradient ASCENT, so
    callers must pass `-spatial_misalignment_loss(...)`. Stages with w_l==0
    or targets[l] is None (empty GT/O/V at that stage) are skipped.
    """
    loss = feats_adv[0].new_zeros(())
    for feat, target, w in zip(feats_adv, targets, stage_weights):
        if target is None or w == 0:
            continue
        f = feat[0]
        margins = pixelwise_cosine_margin(f, target["o_mask"], target["mu_o"], target["mu_v"])
        loss = loss + w * margins.mean()
    return loss
