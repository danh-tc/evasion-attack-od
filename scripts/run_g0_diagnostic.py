#!/usr/bin/env python3
"""Phase G0 -- RRB Gradient Mechanism Analysis (plan.md).

E1->E2 (adding RRB) is the largest transferability jump measured in this
project. This asks *why*, not "which loss/augmentation is best" again:

  1. Correlational: capture K=5 per-view RRB gradient agreement
     (gradient_diagnostics.py) at checkpoints t in {0,25,50,75,99} of a
     RRB_K5_MEAN attack trajectory, then correlate each image's
     trajectory-averaged agreement (primary) and final-checkpoint agreement
     (secondary) against that same image's ASR on every target model.
  2. Causal, matched compute budget (K=5 forward/backward passes/iteration
     in all four): RRB_K5_MEAN vs RRB_K5_CONS vs RRB_K5_DISAGREE vs
     RRB_K5_CONS_SHUFFLE -- does steering the update toward high-consensus
     coordinates change transfer, especially on Group B/C?

Cost note: the full design (K=5 x 4 variants x 50 images x 100 iterations,
~100k forward/backward passes) is the same order of magnitude as E7 (SSA,
num_masks=20), which had to be aborted mid-run for being too slow (see
plan.md). Cheap 2-stage screen before paying for that (plan.md "Phase G0"):

    # Stage 1 (~18% of full-design compute): does consensus exist at all?
    # No target evaluation -- pure gradient diagnostic on one MEAN trajectory.
    python scripts/run_g0_diagnostic.py --n-images 20 --n-iters 30 --k 3 \
        --checkpoints 0,15,29 --variants RRB_K5_MEAN --diagnostic-only \
        --out results/G0_stage1_diagnostic.json

    # Stage 2 (only if Stage 1 shows signal; ~15% of full-design compute):
    # MEAN vs CONS only (DISAGREE/CONS_SHUFFLE are mechanism controls for
    # after CONS is shown to win, not needed for this screen), 3 targets.
    python scripts/run_g0_diagnostic.py --n-images 30 --n-iters 50 --k 3 \
        --checkpoints 0,25,49 --variants RRB_K5_MEAN,RRB_K5_CONS \
        --targets fcos_r50,yolox_l,dino_swin_l \
        --out results/G0_stage2_cheap_causal.json

    # Full run (only if Stage 2 is GO):
    python scripts/run_g0_diagnostic.py --n-images 50 --n-iters 100 \
        --out results/G0_rrb_gradient_mechanism.json

Smoke-test: --n-images 3 --n-iters 5 --checkpoints 0,2,4.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from evasion_od.config import MODEL_ZOO, SURROGATE_NAME, TARGET_NAMES, make_experiments
from evasion_od.models import load_model
from evasion_od.runner import (
    evaluate_on_model_per_image,
    generate_adversarial,
    generate_adversarial_with_diagnostics,
    load_subset,
)

CAUSAL_VARIANTS = ["RRB_K5_MEAN", "RRB_K5_CONS", "RRB_K5_DISAGREE", "RRB_K5_CONS_SHUFFLE"]
DEFAULT_CHECKPOINTS = [0, 25, 50, 75, 99]
AGREEMENT_KEYS = ["A_traj", "A_final", "high_traj", "high_final"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-images", type=int, default=50)
    p.add_argument("--n-iters", type=int, default=100)
    p.add_argument("--manifest", default="dev_300")
    p.add_argument("--checkpoints", default=",".join(str(c) for c in DEFAULT_CHECKPOINTS))
    p.add_argument("--k", type=int, default=5, help="num_masks / K independent RRB views per iteration")
    p.add_argument("--gamma", type=float, default=1.0, help="exponent on the consensus map C")
    p.add_argument("--targets", default=",".join(TARGET_NAMES))
    p.add_argument(
        "--variants",
        default=",".join(CAUSAL_VARIANTS),
        help="comma list, subset of " + ",".join(CAUSAL_VARIANTS),
    )
    p.add_argument(
        "--diagnostic-only",
        action="store_true",
        help="skip all white-box/target evaluation -- just run the attack(s) and log "
        "gradient-agreement checkpoints (cheap Stage 1 screen: is there consensus at all?)",
    )
    p.add_argument(
        "--no-matched-views",
        dest="matched_views",
        action="store_false",
        default=True,
        help="disable common-random-numbers (AttackConfig.deterministic_augmentation): "
        "by default, all variants in one run see the same sampled K RRB views at each "
        "(image, iteration, k), so grad_combine is the only real difference between "
        "their trajectories -- pass this to fall back to independently-sampled views",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", required=True)
    return p.parse_args()


def _pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None
    x_arr, y_arr = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x_arr.std() == 0 or y_arr.std() == 0:
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-tie ranks, no scipy dependency needed for just this."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None
    x_arr, y_arr = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return _pearson(_rankdata(x_arr).tolist(), _rankdata(y_arr).tolist())


def _agreement_features(stats_list: list) -> dict:
    """stats_list: list[CheckpointStats] for one image, sorted by iteration
    -> trajectory-average (primary, plan.md) and final-checkpoint (secondary)
    agreement features.
    """
    if not stats_list:
        return {k: None for k in AGREEMENT_KEYS}
    by_iter = sorted(stats_list, key=lambda c: c.iteration)
    return {
        "A_traj": sum(c.mean_consensus for c in by_iter) / len(by_iter),
        "A_final": by_iter[-1].mean_consensus,
        "high_traj": sum(c.frac_c_gt_0_8 for c in by_iter) / len(by_iter),
        "high_final": by_iter[-1].frac_c_gt_0_8,
    }


def main() -> None:
    args = parse_args()
    checkpoints = frozenset(int(c) for c in args.checkpoints.split(",") if c)
    target_names = [t for t in args.targets.split(",") if t]
    variants_to_run = [v for v in args.variants.split(",") if v]
    for v in variants_to_run:
        if v not in CAUSAL_VARIANTS:
            raise ValueError(f"unknown variant {v!r}, must be one of {CAUSAL_VARIANTS}")

    # drop_prob/num_masks args to make_experiments are unused here: every
    # RRB_K5_* config has mask_enabled=False (RaPA masking is orthogonal to
    # Phase G0, not part of this test).
    experiments = make_experiments(0.05, 1)
    subset = load_subset(args.n_images, args.manifest)
    label_to_cat_id = subset.label_to_cat_id()
    print(
        f"[g0] {len(subset)} images from manifest={args.manifest}, variants={variants_to_run}, "
        f"K={args.k}, gamma={args.gamma}, checkpoints={sorted(checkpoints)}, "
        f"diagnostic_only={args.diagnostic_only}",
        flush=True,
    )

    causal_results: dict = {}
    per_image_asr_by_variant: dict = {}
    checkpoint_stats_by_image: dict = {}

    for variant in variants_to_run:
        cfg = dataclasses.replace(
            experiments[variant].attack,
            max_iterations=args.n_iters,
            num_masks=args.k,
            grad_combine_gamma=args.gamma,
            deterministic_augmentation=args.matched_views,
        )
        print(f"[g0] running {variant} cfg={cfg}", flush=True)
        t0 = time.time()

        surrogate = load_model(SURROGATE_NAME, args.device)
        if variant == "RRB_K5_MEAN":
            adv_images, gt_by_image, diagnostics = generate_adversarial_with_diagnostics(
                surrogate, subset, cfg, args.device, checkpoints
            )
            checkpoint_stats_by_image = diagnostics
        else:
            adv_images, gt_by_image = generate_adversarial(surrogate, subset, cfg, args.device)
        del surrogate
        print(f"[g0] {variant} adversarial generation done in {time.time() - t0:.1f}s", flush=True)

        if args.diagnostic_only:
            continue

        variant_results: dict = {}
        variant_per_image_asr: dict = {}
        for name in [SURROGATE_NAME] + target_names:
            print(f"[g0] {variant}: evaluating {name}", flush=True)
            agg, per_image = evaluate_on_model_per_image(
                name, args.device, subset, adv_images, gt_by_image, label_to_cat_id
            )
            variant_results[name] = dataclasses.asdict(agg)
            variant_per_image_asr[name] = per_image

        causal_results[variant] = variant_results
        per_image_asr_by_variant[variant] = variant_per_image_asr

    # Correlational analysis: agreement measured on RRB_K5_MEAN's own
    # trajectory, correlated against RRB_K5_MEAN's own per-image ASR -- same
    # trajectory for both (plan.md: avoids mismatched deltas across a
    # separate measurement pass). image_ids comes from the diagnostic
    # checkpoints themselves (not from per-image ASR) so per_image_diagnostics
    # is still populated in --diagnostic-only mode, where no ASR exists.
    image_ids = list(checkpoint_stats_by_image.keys())
    agreement = {
        image_id: _agreement_features(checkpoint_stats_by_image.get(image_id, []))
        for image_id in image_ids
    }

    per_image_diagnostics = {
        str(image_id): {
            "checkpoints": [dataclasses.asdict(c) for c in checkpoint_stats_by_image.get(image_id, [])],
            **agreement[image_id],
        }
        for image_id in image_ids
    }

    correlation: dict = {}
    if not args.diagnostic_only and "RRB_K5_MEAN" in per_image_asr_by_variant:
        mean_per_image_asr = per_image_asr_by_variant["RRB_K5_MEAN"]
        group_of = {name: MODEL_ZOO[name].group for name in target_names}
        groups = sorted(set(group_of.values()))
        pooled_by_group = {g: {**{k: [] for k in AGREEMENT_KEYS}, "asr": []} for g in groups}

        for model_name in [SURROGATE_NAME] + target_names:
            xs = {k: [] for k in AGREEMENT_KEYS}
            ys = []
            for image_id in image_ids:
                asr = mean_per_image_asr[model_name].get(image_id)
                feats = agreement[image_id]
                if asr is None or feats["A_traj"] is None:
                    continue
                for key in AGREEMENT_KEYS:
                    xs[key].append(feats[key])
                ys.append(asr)
                if model_name in group_of:
                    g = group_of[model_name]
                    for key in AGREEMENT_KEYS:
                        pooled_by_group[g][key].append(feats[key])
                    pooled_by_group[g]["asr"].append(asr)

            correlation[model_name] = {
                key: {"pearson": _pearson(xs[key], ys), "spearman": _spearman(xs[key], ys), "n": len(ys)}
                for key in AGREEMENT_KEYS
            }

        for g, pooled in pooled_by_group.items():
            n = len(pooled["asr"])
            correlation[f"group_{g}_pooled"] = {
                key: {
                    "pearson": _pearson(pooled[key], pooled["asr"]),
                    "spearman": _spearman(pooled[key], pooled["asr"]),
                    "n": n,
                }
                for key in AGREEMENT_KEYS
            }

    checkpoint_summary: dict = {}
    for cp in sorted(checkpoints):
        vals = [c for stats in checkpoint_stats_by_image.values() for c in stats if c.iteration == cp]
        if not vals:
            continue
        checkpoint_summary[str(cp)] = {
            "n_images": len(vals),
            "mean_pairwise_cosine": float(np.mean([v.mean_pairwise_cosine for v in vals])),
            "median_pairwise_cosine": float(np.mean([v.median_pairwise_cosine for v in vals])),
            "mean_consensus": float(np.mean([v.mean_consensus for v in vals])),
            "frac_c_gt_0_6": float(np.mean([v.frac_c_gt_0_6 for v in vals])),
            "frac_c_gt_0_8": float(np.mean([v.frac_c_gt_0_8 for v in vals])),
            "normalized_variance": float(np.mean([v.normalized_variance for v in vals])),
        }

    if checkpoint_summary:
        print("\n[g0] checkpoint summary (avg over images, RRB_K5_MEAN trajectory):")
        for cp, row in checkpoint_summary.items():
            print(f"  t={cp:>3}  mean_cos={row['mean_pairwise_cosine']:.3f}  "
                  f"E[C]={row['mean_consensus']:.3f}  P(C>0.8)={row['frac_c_gt_0_8']:.3f}  "
                  f"V={row['normalized_variance']:.3f}")

    if causal_results:
        print("\n[g0] causal comparison (white-box + avg black-box mAP-drop):")
        for variant in variants_to_run:
            wb = causal_results[variant][SURROGATE_NAME]["map_drop"]
            if target_names:
                drops = [causal_results[variant][t]["map_drop"] for t in target_names]
                bb = sum(drops) / len(drops)
                print(f"  {variant:<22} white_box_map_drop={wb:.3f}  avg_bb_map_drop={bb:.3f}")
            else:
                print(f"  {variant:<22} white_box_map_drop={wb:.3f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_images": len(subset),
        "manifest": args.manifest,
        "k": args.k,
        "gamma": args.gamma,
        "checkpoints": sorted(checkpoints),
        "n_iters": args.n_iters,
        "variants": variants_to_run,
        "diagnostic_only": args.diagnostic_only,
        "targets": target_names,
        "causal_results": causal_results,
        "checkpoint_summary": checkpoint_summary,
        "correlation": correlation,
        "per_image_diagnostics": per_image_diagnostics,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[g0] wrote {out_path}")


if __name__ == "__main__":
    main()
