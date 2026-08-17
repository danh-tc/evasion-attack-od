#!/usr/bin/env python3
"""Phase S2 -- Trajectory Consistency (plan.md).

S0 (one fixed global scale held for the whole T=50 trajectory) predicted
DINO-Swin-L should benefit from occupancy~0.8. S1 (that same occupancy
region wired into RRB, resampled every iteration, plus rotation+noise)
found the opposite -- every target, including DINO-Swin-L, got WORSE than
RRB_ORIG. S0 and S1 differ in two ways at once: (1) fixed-for-the-whole-
trajectory vs resampled-every-iteration, and (2) resize-only vs full RRB
(rotation+resize+noise). This isolates variable (1) alone -- resize-only in
both arms, matching S0's isolation, so any difference is attributable
specifically to trajectory consistency:

    FIXED_SHRINK  -- one occupancy ~ Uniform(0.7,0.9) drawn per image, held
                     constant across that image's whole T-iteration run
    RANDOM_SHRINK -- same range, resampled fresh every iteration

FIXED_0.8 (occupancy pinned exactly to S0's sweet spot, no per-image
randomness at all) is reused from S0's own results
(results/S0_scale_transfer_sweep_v2.json, scale=0.8 row) rather than rerun
-- identical config, same 30-image/T=50 pilot scale.

GO if FIXED_SHRINK clearly beats RANDOM_SHRINK, especially on DINO-Swin-L,
and the pattern moves back toward S0's (i.e. trajectory consistency, not the
occupancy value itself, was the missing variable in S1). NO-GO if fixed
still doesn't rescue DINO or avg BB still trails RRB_ORIG clearly -- close
the scale-mechanism branch.

    python scripts/run_s2_trajectory_consistency.py --n-images 30 --n-iters 50 \
        --out results/S2_trajectory_consistency.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

from evasion_od.config import SURROGATE_NAME, make_experiments
from evasion_od.models import load_model
from evasion_od.runner import evaluate_on_model, generate_adversarial, load_subset

S2_VARIANTS = ["FIXED_SHRINK", "RANDOM_SHRINK"]
S0_REFERENCE_FILE = "results/S0_scale_transfer_sweep_v2.json"
S0_REFERENCE_SCALE = "0.8"
B1_REFERENCE_FILE = "results/B1_rrb_component_ablation.json"
B1_REFERENCE_KEY = "OSFD_RRB_FULL"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-images", type=int, default=30)
    p.add_argument("--n-iters", type=int, default=50)
    p.add_argument("--manifest", default="dev_300")
    p.add_argument("--targets", default="fcos_r50,yolox_l,dino_swin_l")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", required=True)
    return p.parse_args()


def _load_reference(path_str: str, key: str, target_names: list[str], label: str) -> dict | None:
    path = Path(path_str)
    if not path.exists():
        print(f"[s2] WARNING: {path} not found -- {label} reference omitted", flush=True)
        return None
    data = json.loads(path.read_text())
    if data["n_images"] != 30 or data["n_iters"] != 50:
        print(
            f"[s2] WARNING: {path} scale (n_images={data['n_images']}, n_iters={data['n_iters']}) "
            f"doesn't match this pilot's default -- {label} reference may not be directly comparable",
            flush=True,
        )
    ref = data["results"][key]
    missing = [t for t in target_names if t not in ref]
    if missing:
        print(f"[s2] WARNING: {path} missing targets {missing} for {label}", flush=True)
    return ref


def main() -> None:
    args = parse_args()
    target_names = [t for t in args.targets.split(",") if t]

    experiments = make_experiments(0.05, 1)  # drop_prob/num_masks unused: mask_enabled=False for S2
    subset = load_subset(args.n_images, args.manifest)
    label_to_cat_id = subset.label_to_cat_id()
    print(
        f"[s2] {len(subset)} images from manifest={args.manifest}, "
        f"variants={S2_VARIANTS}, targets={target_names}",
        flush=True,
    )

    results: dict = {}
    fixed_08 = _load_reference(S0_REFERENCE_FILE, S0_REFERENCE_SCALE, target_names, "FIXED_0.8")
    if fixed_08 is not None:
        results["FIXED_0.8"] = fixed_08
    rrb_orig = _load_reference(B1_REFERENCE_FILE, B1_REFERENCE_KEY, target_names, "RRB_ORIG")
    if rrb_orig is not None:
        results["RRB_ORIG"] = rrb_orig

    for variant in S2_VARIANTS:
        cfg = dataclasses.replace(experiments[variant].attack, max_iterations=args.n_iters)
        print(f"[s2] running {variant} cfg={cfg}", flush=True)
        t0 = time.time()

        surrogate = load_model(SURROGATE_NAME, args.device)
        adv_images, gt_by_image = generate_adversarial(surrogate, subset, cfg, args.device)
        del surrogate
        print(f"[s2] {variant} adversarial generation done in {time.time() - t0:.1f}s", flush=True)

        variant_results: dict = {
            SURROGATE_NAME: dataclasses.asdict(
                evaluate_on_model(
                    SURROGATE_NAME, args.device, subset, adv_images, gt_by_image, label_to_cat_id
                )
            )
        }
        for name in target_names:
            print(f"[s2] {variant}: evaluating {name}", flush=True)
            variant_results[name] = dataclasses.asdict(
                evaluate_on_model(name, args.device, subset, adv_images, gt_by_image, label_to_cat_id)
            )
        results[variant] = variant_results

    order = ["RRB_ORIG", "FIXED_0.8"] + S2_VARIANTS
    print("\n[s2] summary (white-box + per-target + avg black-box mAP-drop):")
    header = f"{'variant':<14}{'white_box':>10}" + "".join(f"{t:>14}" for t in target_names) + f"{'avg_bb':>9}"
    print(header)
    for variant in order:
        if variant not in results:
            continue
        r = results[variant]
        wb = r[SURROGATE_NAME]["map_drop"]
        per_t = [r[t]["map_drop"] for t in target_names if t in r]
        avg = sum(per_t) / len(per_t) if per_t else float("nan")
        row = f"{variant:<14}{wb:>10.3f}" + "".join(f"{r[t]['map_drop']:>14.3f}" for t in target_names if t in r) + f"{avg:>9.3f}"
        print(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_images": len(subset),
        "manifest": args.manifest,
        "n_iters": args.n_iters,
        "variants": order,
        "targets": target_names,
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[s2] wrote {out_path}")


if __name__ == "__main__":
    main()
