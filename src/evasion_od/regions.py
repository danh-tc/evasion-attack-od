"""Object / vicinity region masks on backbone feature grids, for the
object-context relational hypothesis in idea.txt.

Margins are defined relative to each box's own size on the feature grid
(same convention as rrb.py's `adaptive_random_resizing`, which scales by
`box_size/img_size` rather than a fixed pixel amount) so a tiny COCO object
and a frame-filling one both get a proportionate vicinity ring.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-8


def boxes_to_feature_cells(
    gt_boxes_xyxy: torch.Tensor, img_h: int, img_w: int, feat_h: int, feat_w: int
) -> torch.Tensor:
    """Map boxes (resized-image pixel coords) to integer cell AABBs on the
    feat_h x feat_w grid, as [x1,y1,x2,y2) (x2/y2 exclusive), each >=1x1.
    """
    scale_x = feat_w / img_w
    scale_y = feat_h / img_h
    x1 = torch.clamp((gt_boxes_xyxy[:, 0] * scale_x).floor().long(), 0, feat_w - 1)
    y1 = torch.clamp((gt_boxes_xyxy[:, 1] * scale_y).floor().long(), 0, feat_h - 1)
    x2 = torch.clamp((gt_boxes_xyxy[:, 2] * scale_x).ceil().long(), min=1)
    y2 = torch.clamp((gt_boxes_xyxy[:, 3] * scale_y).ceil().long(), min=1)
    x2 = torch.clamp(torch.maximum(x2, x1 + 1), max=feat_w)
    y2 = torch.clamp(torch.maximum(y2, y1 + 1), max=feat_h)
    return torch.stack([x1, y1, x2, y2], dim=1)


def object_vicinity_masks(
    gt_boxes_xyxy: torch.Tensor,
    img_h: int,
    img_w: int,
    feat_h: int,
    feat_w: int,
    r: float,
    min_margin_cells: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (O_mask, V_mask), each a (feat_h, feat_w) bool tensor.

    O = union of GT boxes mapped to the feature grid.
    V = union of each box's own ring (expanded by `r * max(box_w, box_h)`
    cells, floor `min_margin_cells`), minus the global O (so one object's
    ring never counts a neighboring object's pixels as "vicinity").
    """
    device = gt_boxes_xyxy.device
    cells = boxes_to_feature_cells(gt_boxes_xyxy, img_h, img_w, feat_h, feat_w)

    O = torch.zeros((feat_h, feat_w), dtype=torch.bool, device=device)
    V_raw = torch.zeros((feat_h, feat_w), dtype=torch.bool, device=device)
    for x1, y1, x2, y2 in cells.tolist():
        O[y1:y2, x1:x2] = True
        margin = max(min_margin_cells, round(r * max(x2 - x1, y2 - y1)))
        ex1, ey1 = max(0, x1 - margin), max(0, y1 - margin)
        ex2, ey2 = min(feat_w, x2 + margin), min(feat_h, y2 + margin)
        V_raw[ey1:ey2, ex1:ex2] = True

    V = V_raw & ~O
    return O, V


def channel_stats(feat_chw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel mean/std over the whole (H,W) extent. feat_chw: (C,H,W)."""
    mu = feat_chw.mean(dim=(1, 2))
    sigma = feat_chw.std(dim=(1, 2))
    return mu, sigma


def region_score_magnitude(feat_chw: torch.Tensor, mask_hw: torch.Tensor) -> torch.Tensor:
    """M1: mean over the region of the per-position channel-L2 norm.

    Returns a 0-dim tensor (not `.item()`) so this stays usable inside a loss
    that needs gradients w.r.t. `feat_chw` (e.g. relational_contrast_loss);
    diagnostics that only want a float (run_phase0_diagnostic.py) call
    `.item()` themselves at the call site.
    """
    norm_hw = feat_chw.pow(2).sum(dim=0).sqrt()
    return norm_hw[mask_hw].mean()


def region_score_normalized_energy(
    feat_chw: torch.Tensor, mask_hw: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    """M2: mean over the region of per-channel-standardized squared activation.

    Returns a 0-dim tensor -- see `region_score_magnitude` docstring.
    """
    z = (feat_chw - mu[:, None, None]) / (sigma[:, None, None] + EPS)
    energy_hw = z.pow(2).mean(dim=0)
    return energy_hw[mask_hw].mean()


def relational_contrast(s_o: torch.Tensor, s_v: torch.Tensor) -> torch.Tensor:
    """C = (S_O - S_V) / (S_O + S_V + eps), bounded in (-1, 1).

    Bounded -> saturates once C_adv is driven near +/-1 (empirically within
    ~20/100 MI-FGSM iterations), after which gradient vanishes and the rest
    of the attack budget is wasted. Fine for Phase 0 diagnostics (bounded
    values are easy to compare across stages/architectures); NOT used for
    the attack loss -- see `relational_diff` for the unbounded version used
    in relational_contrast_loss/osfd_rel_hybrid_loss instead.
    """
    return (s_o - s_v) / (s_o + s_v + EPS)


def relational_diff(s_o: torch.Tensor, s_v: torch.Tensor) -> torch.Tensor:
    """D = S_O - S_V, unbounded -- keeps producing gradient for the full
    attack instead of saturating like `relational_contrast`."""
    return s_o - s_v


def region_mean_vector(feat_chw: torch.Tensor, mask_hw: torch.Tensor) -> torch.Tensor:
    """Mean feature vector over the masked spatial positions -> (C,).

    The "prototype" for a region (mu_O or mu_V), for the P1b/Phase-2-followup
    per-pixel object-to-context feature misalignment idea: instead of
    collapsing O and V to one scalar each (Phase 1/2's S_O, S_V -- which
    Phase 2 showed produces too coarse a gradient), keep a C-dim direction
    and compare every individual pixel's feature to it.
    """
    return feat_chw[:, mask_hw].mean(dim=1)


def pixelwise_cosine_margin(
    feat_chw: torch.Tensor, mask_hw: torch.Tensor, mu_self: torch.Tensor, mu_other: torch.Tensor
) -> torch.Tensor:
    """For every position in mask_hw: cos(F(p), mu_self) - cos(F(p), mu_other).

    Returns a 1D tensor, one value per masked position (dense, not
    mean-pooled to a single scalar -- unlike relational_contrast/diff).
    Positive means a pixel looks more like its own region's clean prototype
    than the other region's; e.g. mask_hw=O_mask, mu_self=mu_O, mu_other=mu_V
    tests whether object pixels stay object-like rather than drifting toward
    vicinity-like.
    """
    feats = feat_chw[:, mask_hw].T  # (N, C)
    cos_self = F.cosine_similarity(feats, mu_self.unsqueeze(0), dim=1)
    cos_other = F.cosine_similarity(feats, mu_other.unsqueeze(0), dim=1)
    return cos_self - cos_other


def precompute_relational_targets(
    feats_clean: tuple[torch.Tensor, ...],
    gt_boxes_xyxy: torch.Tensor,
    img_h: int,
    img_w: int,
    r: float = 1.0,
    min_margin_cells: int = 1,
) -> list[dict | None]:
    """Per surrogate backbone stage: O/V masks + clean per-channel stats +
    the clean relational contrast C_clean -- all fixed for the whole attack,
    computed once from feats_clean (same spirit as feats_clean itself).

    mu/sigma come from the CLEAN feature map only and are reused to score
    the adversarial feature map every iteration (attack.py), rather than
    recomputed from feats_adv each step -- otherwise the attack could shrink
    C_adv just by shifting the image's overall per-channel statistics
    (renormalizing away the signal) instead of actually disrupting the
    object/vicinity spatial relation, which is the thing Phase 0 validated.

    Returns one entry per stage; None where GT is empty or O/V ended up
    empty at that stage's resolution (relational_contrast_loss skips those).
    """
    targets: list[dict | None] = []
    for feat in feats_clean:
        f = feat[0]
        feat_h, feat_w = f.shape[-2:]
        if gt_boxes_xyxy.numel() == 0:
            targets.append(None)
            continue
        O_mask, V_mask = object_vicinity_masks(
            gt_boxes_xyxy, img_h, img_w, feat_h, feat_w, r, min_margin_cells
        )
        if not O_mask.any() or not V_mask.any():
            targets.append(None)
            continue
        mu, sigma = channel_stats(f)
        s_o = region_score_normalized_energy(f, O_mask, mu, sigma)
        s_v = region_score_normalized_energy(f, V_mask, mu, sigma)
        targets.append(
            {
                "o_mask": O_mask,
                "v_mask": V_mask,
                "mu": mu,
                "sigma": sigma,
                "c_clean": relational_contrast(s_o, s_v).detach(),
                "d_clean": relational_diff(s_o, s_v).detach(),
            }
        )
    return targets


def precompute_spatial_targets(
    feats_clean: tuple[torch.Tensor, ...],
    gt_boxes_xyxy: torch.Tensor,
    img_h: int,
    img_w: int,
    r: float = 1.0,
    min_margin_cells: int = 1,
) -> list[dict | None]:
    """Per surrogate backbone stage: O mask + clean prototypes mu_O, mu_V --
    fixed for the whole attack, for spatial_misalignment_loss (Phase 1b's
    object-to-context feature misalignment, validated in
    results/P1b_prototype_diagnostic.json: P(margin>0)=1.000 on essentially
    every model/stage, including YOLOX stage 2 where the mean-pooled S_O/S_V
    metric had reversed -- see relational_contrast/relational_diff).

    mu_O/mu_V come from the CLEAN feature map only, reused to score feats_adv
    every iteration -- same rationale as precompute_relational_targets's
    fixed mu/sigma (avoid the attack "winning" by shifting statistics rather
    than actually disrupting the object/vicinity spatial relation).

    Returns one entry per stage; None where GT is empty or O/V ended up
    empty at that stage's resolution.
    """
    targets: list[dict | None] = []
    for feat in feats_clean:
        f = feat[0]
        feat_h, feat_w = f.shape[-2:]
        if gt_boxes_xyxy.numel() == 0:
            targets.append(None)
            continue
        O_mask, V_mask = object_vicinity_masks(
            gt_boxes_xyxy, img_h, img_w, feat_h, feat_w, r, min_margin_cells
        )
        if not O_mask.any() or not V_mask.any():
            targets.append(None)
            continue
        targets.append(
            {
                "o_mask": O_mask,
                "mu_o": region_mean_vector(f, O_mask).detach(),
                "mu_v": region_mean_vector(f, V_mask).detach(),
            }
        )
    return targets
