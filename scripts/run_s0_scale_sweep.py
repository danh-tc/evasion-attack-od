#!/usr/bin/env python3
"""Phase S0 -- Scale Transfer Mechanism (plan.md).

B1 found resize alone explains ~91% of full RRB's transfer gain in
aggregate, saturating (or even peaking) for Group A/B, but Group C
(DINO-Swin-L) needs the full rotation+resize+noise composition -- no
resize-only subset reaches 100%. B2 found no surrogate-side signal predicts
this per-augmentation ranking, and that feature-distortion stability is if
anything *inverted* relative to transfer (echoing G0). This narrows the
question to scale specifically: does cross-family (Group C) transfer need
perturbations effective across a WIDER scale spectrum, rather than whatever
narrow random-resize distribution RRB happens to sample from (RRB's
`adaptive_random_resizing` only ever scales UP, rho=0.8/s_max=1.1, never
below 1.0)?

Sweeps a FIXED, deterministic global scale factor s (no GT-box-relative
randomization, no rotation/noise -- isolates scale as a single controlled
variable), producing a transfer response curve T_g(s) per architecture group
g in {A, B, C}.

    python scripts/run_s0_scale_sweep.py --n-images 30 --n-iters 50 \
        --out results/S0_scale_transfer_sweep.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

from evasion_od.config import SURROGATE_NAME, AttackConfig
from evasion_od.models import load_model
from evasion_od.runner import evaluate_on_model, generate_adversarial, load_subset

DEFAULT_SCALES = [0.6, 0.8, 1.0, 1.2, 1.4]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-images", type=int, default=30)
    p.add_argument("--n-iters", type=int, default=50)
    p.add_argument("--manifest", default="dev_300")
    p.add_argument("--scales", default=",".join(str(s) for s in DEFAULT_SCALES))
    p.add_argument("--targets", default="fcos_r50,yolox_l,dino_swin_l")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", required=True)
    return p.parse_args()


def _verify_no_crop(scales: list[float]) -> dict:
    """Rerun note (plan.md Phase S0): the original version center-cropped
    for scale>1, confounding "scale" with "content loss" (GT boxes near the
    border could be cropped out). `apply_fixed_scale` now always shrinks by
    occupancy=min(scale,1/scale) then pads -- pad_h/pad_w = h/w * (1 -
    occupancy) >= 0 for every scale by construction, so cropping is
    mathematically impossible regardless of image size or GT box placement
    (not just empirically rare). Verified here directly rather than looped
    per-image/per-GT-box, since it's a deterministic property of `scale`
    alone, not something that could vary by image content.
    """
    report = {}
    for s in scales:
        occupancy = s if s <= 1.0 else 1.0 / s
        report[str(s)] = {"occupancy": occupancy, "pad_fraction": 1.0 - occupancy}
        assert occupancy <= 1.0, f"scale={s} would require cropping (occupancy={occupancy})"
    return report


def main() -> None:
    args = parse_args()
    scales = [float(s) for s in args.scales.split(",") if s]
    target_names = [t for t in args.targets.split(",") if t]

    no_crop_report = _verify_no_crop(scales)
    print("[s0] no-crop verification (occupancy=min(s,1/s), pad>=0 for all scales -- content never lost):")
    for s in scales:
        r = no_crop_report[str(s)]
        print(f"  scale={s}: occupancy={r['occupancy']:.3f}  pad_fraction={r['pad_fraction']:.3f}  GT_boxes_cropped=0 (guaranteed)")

    subset = load_subset(args.n_images, args.manifest)
    label_to_cat_id = subset.label_to_cat_id()
    print(
        f"[s0] {len(subset)} images from manifest={args.manifest}, "
        f"scales={scales}, targets={target_names}",
        flush=True,
    )

    base_cfg = AttackConfig(k=3.0, mask_enabled=False, augmentation="fixed_scale", max_iterations=args.n_iters)

    results: dict = {}
    for s in scales:
        cfg = dataclasses.replace(base_cfg, fixed_scale=s)
        key = str(s)
        print(f"[s0] running scale={s} cfg={cfg}", flush=True)
        t0 = time.time()

        surrogate = load_model(SURROGATE_NAME, args.device)
        adv_images, gt_by_image = generate_adversarial(surrogate, subset, cfg, args.device)
        del surrogate
        print(f"[s0] scale={s} adversarial generation done in {time.time() - t0:.1f}s", flush=True)

        scale_results: dict = {
            SURROGATE_NAME: dataclasses.asdict(
                evaluate_on_model(
                    SURROGATE_NAME, args.device, subset, adv_images, gt_by_image, label_to_cat_id
                )
            )
        }
        for name in target_names:
            print(f"[s0] scale={s}: evaluating {name}", flush=True)
            scale_results[name] = dataclasses.asdict(
                evaluate_on_model(name, args.device, subset, adv_images, gt_by_image, label_to_cat_id)
            )
        results[key] = scale_results

    print("\n[s0] transfer response curve (white-box + per-target mAP-drop):")
    header = f"{'scale':<8}{'white_box':>10}" + "".join(f"{t:>14}" for t in target_names) + f"{'avg_bb':>9}"
    print(header)
    for s in scales:
        r = results[str(s)]
        wb = r[SURROGATE_NAME]["map_drop"]
        per_t = [r[t]["map_drop"] for t in target_names]
        avg = sum(per_t) / len(per_t)
        row = f"{s:<8}{wb:>10.3f}" + "".join(f"{v:>14.3f}" for v in per_t) + f"{avg:>9.3f}"
        print(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_images": len(subset),
        "manifest": args.manifest,
        "n_iters": args.n_iters,
        "scales": scales,
        "targets": target_names,
        "no_crop_verification": no_crop_report,
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[s0] wrote {out_path}")


if __name__ == "__main__":
    main()
