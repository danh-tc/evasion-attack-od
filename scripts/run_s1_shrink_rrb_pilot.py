#!/usr/bin/env python3
"""Phase S1 -- Bidirectional/Shrink-aware RRB (plan.md).

S0 found DINO-Swin-L's transfer response to a fixed global scale peaks at
content-occupancy ~0.8 (pad_fraction ~0.2), not at the undistorted image
(occupancy=1.0, where it's actually worst) -- unlike white-box/FCOS, which
peak at occupancy=1.0 and monotonically degrade with more padding. RRB's own
`adaptive_random_resizing` nets out to occupancy range ~[0.91, 1.0] (from
s_max=1.1) -- too narrow to ever reach the 0.8 region. This is a small
causal attack pilot, not a new method yet: does swapping RRB's resize
sampling range to actually cover that region change transfer the way S0's
diagnostic predicts?

Compares 2 fresh variants against RRB_ORIG (= B1's already-computed
OSFD_RRB_FULL, same scale/manifest, reused rather than rerun):

    RRB_SHRINK  -- occupancy ~ Uniform(0.7, 0.9), centered on the sweet spot
    RRB_BIDIR   -- occupancy ~ Uniform(0.7, 1.0), covers both near-original
                   and moderate shrink, avoids the extreme (occupancy=0.6
                   was S0's worst point for white-box)

Expectation if S0 is causal: FCOS ~ unchanged; YOLOX BIDIR >= ORIG; DINO-Swin
SHRINK/BIDIR > ORIG clearly, gain increasing with architectural distance
from the surrogate.

    python scripts/run_s1_shrink_rrb_pilot.py --n-images 30 --n-iters 50 \
        --out results/S1_shrink_rrb_pilot.json
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

S1_VARIANTS = ["RRB_SHRINK", "RRB_BIDIR"]
ORIG_REFERENCE_FILE = "results/B1_rrb_component_ablation.json"
ORIG_REFERENCE_KEY = "OSFD_RRB_FULL"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-images", type=int, default=30)
    p.add_argument("--n-iters", type=int, default=50)
    p.add_argument("--manifest", default="dev_300")
    p.add_argument("--targets", default="fcos_r50,yolox_l,dino_swin_l")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", required=True)
    return p.parse_args()


def _load_orig_reference(target_names: list[str]) -> dict | None:
    """RRB_ORIG's numbers are B1's OSFD_RRB_FULL (same config, same 30
    images/T=50 pilot scale/manifest) -- reused instead of rerun."""
    path = Path(ORIG_REFERENCE_FILE)
    if not path.exists():
        print(f"[s1] WARNING: {path} not found -- RRB_ORIG reference omitted", flush=True)
        return None
    b1 = json.loads(path.read_text())
    if b1["n_images"] != 30 or b1["n_iters"] != 50:
        print(
            f"[s1] WARNING: {path} scale (n_images={b1['n_images']}, n_iters={b1['n_iters']}) "
            "doesn't match this pilot's default -- RRB_ORIG reference may not be directly comparable",
            flush=True,
        )
    orig = b1["results"][ORIG_REFERENCE_KEY]
    missing = [t for t in target_names if t not in orig]
    if missing:
        print(f"[s1] WARNING: {ORIG_REFERENCE_FILE} missing targets {missing} -- partial reference", flush=True)
    return orig


def main() -> None:
    args = parse_args()
    target_names = [t for t in args.targets.split(",") if t]

    experiments = make_experiments(0.05, 1)  # drop_prob/num_masks unused: mask_enabled=False for S1
    subset = load_subset(args.n_images, args.manifest)
    label_to_cat_id = subset.label_to_cat_id()
    print(
        f"[s1] {len(subset)} images from manifest={args.manifest}, "
        f"variants={S1_VARIANTS}, targets={target_names}",
        flush=True,
    )

    results: dict = {}
    orig_ref = _load_orig_reference(target_names)
    if orig_ref is not None:
        results["RRB_ORIG"] = orig_ref

    for variant in S1_VARIANTS:
        cfg = dataclasses.replace(experiments[variant].attack, max_iterations=args.n_iters)
        print(f"[s1] running {variant} cfg={cfg}", flush=True)
        t0 = time.time()

        surrogate = load_model(SURROGATE_NAME, args.device)
        adv_images, gt_by_image = generate_adversarial(surrogate, subset, cfg, args.device)
        del surrogate
        print(f"[s1] {variant} adversarial generation done in {time.time() - t0:.1f}s", flush=True)

        variant_results: dict = {
            SURROGATE_NAME: dataclasses.asdict(
                evaluate_on_model(
                    SURROGATE_NAME, args.device, subset, adv_images, gt_by_image, label_to_cat_id
                )
            )
        }
        for name in target_names:
            print(f"[s1] {variant}: evaluating {name}", flush=True)
            variant_results[name] = dataclasses.asdict(
                evaluate_on_model(name, args.device, subset, adv_images, gt_by_image, label_to_cat_id)
            )
        results[variant] = variant_results

    print("\n[s1] summary (white-box + per-target + avg black-box mAP-drop):")
    header = f"{'variant':<12}{'white_box':>10}" + "".join(f"{t:>14}" for t in target_names) + f"{'avg_bb':>9}"
    print(header)
    for variant in ["RRB_ORIG"] + S1_VARIANTS:
        if variant not in results:
            continue
        r = results[variant]
        wb = r[SURROGATE_NAME]["map_drop"]
        per_t = [r[t]["map_drop"] for t in target_names if t in r]
        avg = sum(per_t) / len(per_t) if per_t else float("nan")
        row = f"{variant:<12}{wb:>10.3f}" + "".join(f"{r[t]['map_drop']:>14.3f}" for t in target_names if t in r) + f"{avg:>9.3f}"
        print(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_images": len(subset),
        "manifest": args.manifest,
        "n_iters": args.n_iters,
        "variants": ["RRB_ORIG"] + S1_VARIANTS,
        "targets": target_names,
        "orig_reference_file": ORIG_REFERENCE_FILE if orig_ref is not None else None,
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[s1] wrote {out_path}")


if __name__ == "__main__":
    main()
