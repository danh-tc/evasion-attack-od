"""Phase G0 -- RRB Gradient Mechanism Analysis (plan.md).

E1->E2 (adding RRB) is the single largest transferability jump measured in
this project, larger than any loss variant tried in Phase 1/2 (idea.txt) or
any augmentation variant tried in Sec 5.2 (new-plan.txt). Every prior
experiment asked "which loss/augmentation transfers best?"; this one asks a
mechanistic question instead: across the K independent RRB-augmented views
MI-FGSM already samples per iteration (`AttackConfig.num_masks`), how much do
the resulting gradients actually agree, and does steering the update toward
the agreeing coordinates change transfer?

Two uses:
  - `compute_checkpoint_stats`: correlational diagnostic, logged at a few
    iterations of an attack trajectory (see scripts/run_g0_diagnostic.py) --
    are the K gradients revealing a stable adversarial subspace (high
    sign-agreement) or is RRB acting as plain stochastic regularization (low
    agreement, still transfers)?
  - `combine_gradients`: the causal test. Replaces MI-FGSM's per-iteration
    `grads.mean()` with a consensus-weighted (or disagreement-weighted, or
    consensus-shuffled-control) combination, at identical compute budget (K
    forward/backward passes either way) -- see AttackConfig.grad_combine and
    the RRB_K5_* experiments in config.py.

All gradients here are per-image, shape (C, H, W) -- same as `delta` in
attack.py -- since RRB's rotation/resize are applied inside the
forward/backward graph (torchvision.transforms.functional.rotate,
F.interpolate/F.pad -- see rrb.py), `torch.autograd.grad(loss, delta)`
already chain-rules each view's gradient back to delta's own coordinate
system. No manual re-alignment across views is needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

EPS = 1e-12


def consensus_map(grads: torch.Tensor) -> torch.Tensor:
    """grads: (K, C, H, W) -> per-coordinate sign-agreement C in [0, 1].

    C_p = |mean_k sign(g_k,p)|: 1.0 where all K views agree on gradient sign
    at that coordinate, 0.0 where views split evenly.
    """
    return torch.sign(grads).mean(dim=0).abs()


def shuffle_spatial(c: torch.Tensor) -> torch.Tensor:
    """Permute a (C, H, W) consensus map's (H, W) positions, identically
    across channels -- the CONS_SHUFFLE control. Preserves C's value
    distribution and channel structure, destroys *where* high-consensus
    coordinates sit, so CONS > CONS_SHUFFLE isolates spatial localization of
    stable gradients as the source of any gain, rather than just the
    weighting's effect on overall gradient magnitude.
    """
    n_c, h, w = c.shape
    perm = torch.randperm(h * w, device=c.device)
    return c.reshape(n_c, h * w)[:, perm].reshape(n_c, h, w)


def combine_gradients(grads_list: list[torch.Tensor], mode: str, gamma: float = 1.0) -> torch.Tensor:
    """Combine K per-view gradients (same shape as delta) into one MI-FGSM
    update direction. "mean" reproduces the pre-G0 behavior used by every
    other experiment in plan.md; the other three modes are RRB_K5_* only.
    """
    grads = torch.stack(grads_list, dim=0)
    mean_g = grads.mean(dim=0)
    if mode == "mean":
        return mean_g
    c = consensus_map(grads)
    if mode == "consensus":
        return c.pow(gamma) * mean_g
    if mode == "disagree":
        return (1.0 - c).pow(gamma) * mean_g
    if mode == "consensus_shuffle":
        return shuffle_spatial(c).pow(gamma) * mean_g
    raise ValueError(f"unknown grad_combine: {mode!r}")


def pairwise_cosine_stats(grads: torch.Tensor) -> tuple[float, float]:
    """grads: (K, C, H, W) -> (mean, median) cosine similarity over all
    K*(K-1)/2 view pairs, each view flattened to one vector.
    """
    k = grads.shape[0]
    flat = grads.reshape(k, -1)
    flat = flat / (flat.norm(dim=1, keepdim=True) + EPS)
    sim = flat @ flat.T
    iu = torch.triu_indices(k, k, offset=1, device=grads.device)
    pairs = sim[iu[0], iu[1]]
    return float(pairs.mean()), float(pairs.median())


def normalized_variance(grads: torch.Tensor) -> float:
    """V = E_p[Var_k(g_k,p)] / (E_p[mean_k(g_k,p)^2] + eps) -- cross-view
    gradient variance normalized by the combined gradient's own magnitude,
    so it's comparable across images/iterations regardless of raw gradient
    scale.
    """
    mean_g = grads.mean(dim=0)
    var_g = grads.var(dim=0, unbiased=False)
    return float(var_g.mean() / (mean_g.pow(2).mean() + EPS))


@dataclass
class CheckpointStats:
    iteration: int
    mean_pairwise_cosine: float
    median_pairwise_cosine: float
    mean_consensus: float  # E_p[C_p]
    frac_c_gt_0_6: float  # P(C_p > 0.6)
    frac_c_gt_0_8: float  # P(C_p > 0.8)
    normalized_variance: float


def compute_checkpoint_stats(grads_list: list[torch.Tensor], iteration: int) -> CheckpointStats:
    with torch.no_grad():
        grads = torch.stack(grads_list, dim=0)
        c = consensus_map(grads)
        mean_cos, median_cos = pairwise_cosine_stats(grads)
        return CheckpointStats(
            iteration=iteration,
            mean_pairwise_cosine=mean_cos,
            median_pairwise_cosine=median_cos,
            mean_consensus=float(c.mean()),
            frac_c_gt_0_6=float((c > 0.6).float().mean()),
            frac_c_gt_0_8=float((c > 0.8).float().mean()),
            normalized_variance=normalized_variance(grads),
        )
