#!/usr/bin/env python3
"""Sanity check (no GPU/model, pure sampling geometry) for the proposed
History-Aware / Anti-Redundant RRB idea -- candidate Phase S3 in plan.md,
before spending GPU budget on the 3-arm pilot (RRB_IID / RRB_ANTI_REPEAT /
RRB_OVER_DIVERSE).

Question: `ANTI_REPEAT` rejects/resamples a draw if it's "too close" (within
threshold tau) to the last h=3 draws. For this to have any chance of changing
attack behavior, IID sampling must actually produce near-duplicate consecutive
draws often enough for the rejection to bite. This script estimates that
rejection rate directly from the RRB_ORIG-matching sampling distribution,
with no image/model/GPU involved -- if the answer is "rejection rate is near
0 at any reasonable tau", ANTI_REPEAT vs IID is unlikely to show a real effect
and the GPU pilot should be redesigned (e.g. drop ANTI_REPEAT, keep only
OVER_DIVERSE as a clean positive control against IID) rather than spend
compute on a policy that's statistically indistinguishable from what it's
compared against.

Parameterization (matches rrb.py + config.py's occ_low/occ_high defaults and
the "safe RRB region" the S3 proposal specifies -- theta<=7 deg, occupancy in
RRB_ORIG's own effective range [0.91,1.0], *not* S1's wider/aggressive
[0.7,1.0] shrink range that plan.md already found to underperform):

    z = [theta_norm, occ_norm, pad_top_frac, pad_left_frac]  in [0,1]^4

  - theta ~ Uniform(-7, 7) deg (random_axis_rotation), normalized to [0,1].
  - occ   ~ Uniform(0.91, 1.0) (random_occupancy_resize), normalized to [0,1].
  - pad_top_frac, pad_left_frac ~ Uniform(0,1) exactly, by construction of
    `random.randint(0, pad_h)` / `random.randint(0, pad_w)` in
    random_occupancy_resize -- the fraction of remaining pad slack used is
    uniform regardless of image resolution or occupancy value.

Noise (additive_gaussian_noise) is excluded: sigma is a fixed scalar (not
resampled per iteration), and its *realization* is a full per-pixel field --
not a low-dimensional parameter comparable to theta/occ/pad, so it isn't part
of the "transform parameter vector" a history-aware policy would condition
on.

Usage: python scripts/sanity_check_history_aware_rrb.py
"""

from __future__ import annotations

import numpy as np

THETA_RANGE = (-7.0, 7.0)
OCC_RANGE = (0.91, 1.0)
H = 3  # history window (matches the S3 proposal's h=3)
T = 50  # matches pilot max_iterations
N_TRAJECTORIES = 4000
N_PAIRS = 200_000
SEED = 42


def sample_vec(rng: np.random.Generator, n: int) -> np.ndarray:
    theta = rng.uniform(*THETA_RANGE, size=n)
    occ = rng.uniform(*OCC_RANGE, size=n)
    pad_top = rng.uniform(0.0, 1.0, size=n)
    pad_left = rng.uniform(0.0, 1.0, size=n)
    theta_n = (theta - THETA_RANGE[0]) / (THETA_RANGE[1] - THETA_RANGE[0])
    occ_n = (occ - OCC_RANGE[0]) / (OCC_RANGE[1] - OCC_RANGE[0])
    return np.stack([theta_n, occ_n, pad_top, pad_left], axis=1)  # (n, 4), each dim in [0,1]


def main() -> None:
    rng = np.random.default_rng(SEED)
    max_dist = float(np.sqrt(4.0))  # diagonal of the [0,1]^4 unit hypercube

    # Reference distribution: distance between two *independent* draws (i.e.
    # what "a random pair" looks like -- the null against which we judge
    # whether consecutive-in-trajectory draws are unusually close).
    pair_a = sample_vec(rng, N_PAIRS)
    pair_b = sample_vec(rng, N_PAIRS)
    pair_dist = np.linalg.norm(pair_a - pair_b, axis=1)

    # Simulated trajectories: for each t >= H, distance from draw t to each
    # of the previous H draws, then take the min (= what ANTI_REPEAT checks
    # before accepting a draw; also what OVER_DIVERSE tries to maximize).
    min_dists = []
    for _ in range(N_TRAJECTORIES):
        traj = sample_vec(rng, T)
        for t in range(H, T):
            window = traj[t - H : t]
            d = np.linalg.norm(window - traj[t], axis=1)
            min_dists.append(d.min())
    min_dists = np.array(min_dists)

    def pct(arr: np.ndarray, q: float) -> float:
        return float(np.percentile(arr, q))

    print("=== Reference: distance between two independent draws (the null) ===")
    print(f"  max possible distance (unit hypercube diagonal): {max_dist:.3f}")
    for q in (5, 25, 50, 75, 95):
        print(f"  p{q:<3d} pairwise distance: {pct(pair_dist, q):.3f}")
    print(f"  mean pairwise distance: {pair_dist.mean():.3f}")

    print(f"\n=== min-distance to previous h={H} draws, within {T}-step trajectories ===")
    print(f"  ({N_TRAJECTORIES} trajectories x {T - H} checked steps = {len(min_dists)} samples)")
    for q in (5, 25, 50, 75, 95):
        print(f"  p{q:<3d} min-distance: {pct(min_dists, q):.3f}")
    print(f"  mean min-distance: {min_dists.mean():.3f}")

    print("\n=== Implied ANTI_REPEAT rejection rate at candidate thresholds tau ===")
    print("  (tau expressed as a fraction of the null's own percentiles, so this")
    print("   reads as 'how often would a draw already be farther than the X-th")
    print("   percentile of a totally unrelated random pair')")
    for q in (5, 10, 25, 50):
        tau = pct(pair_dist, q)
        reject_rate = float((min_dists < tau).mean())
        print(f"  tau = p{q:<3d} of null ({tau:.3f}): reject_rate = {reject_rate:.1%}")

    print("\n=== Interpretation ===")
    ratio = min_dists.mean() / pair_dist.mean()
    print(f"  mean(min-of-{H}) / mean(random pair) = {ratio:.3f}")
    print(
        "  This ratio is an order-statistic artifact (min of h comparisons is\n"
        "  stochastically smaller than a single pairwise distance) even under pure\n"
        "  IID sampling with zero real redundancy structure -- it is NOT by itself\n"
        "  evidence that IID trajectories contain problematic near-duplicates.\n"
        "  What matters for the pilot is the reject_rate table above: if it stays\n"
        "  small (single digits or less) even at tau = p25 of the null, ANTI_REPEAT\n"
        "  will resample on only a small minority of iterations, so its trajectories\n"
        "  will look statistically close to RRB_IID -- a GPU pilot is unlikely to\n"
        "  show a clear separation between the two arms. In that case, prefer\n"
        "  running OVER_DIVERSE (farthest-of-K-candidates, a much larger and more\n"
        "  deterministic push away from IID) as the main comparison against IID,\n"
        "  and treat ANTI_REPEAT-with-a-small-tau as a weak/likely-null arm rather\n"
        "  than the primary bet."
    )


if __name__ == "__main__":
    main()
