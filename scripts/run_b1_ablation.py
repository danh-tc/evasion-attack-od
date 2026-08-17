#!/usr/bin/env python3
"""Phase B1 -- RRB Component Ablation (plan.md).

E1->E2 (adding full RRB) is the largest transferability jump measured in
this project (avg black-box mAP-drop 0.111 -> 0.335). Loss changes (Phase 2,
idea.txt) and gradient-consensus weighting (Phase G0) both failed to beat it.
This asks which part of RRB actually drives that gain: rotation, adaptive
resize, additive noise (mislabeled "blur" in the original OSFD code -- see
rrb.py's `additive_gaussian_noise`), or their interaction. Same surrogate,
loss (OSFD k=3), eps/alpha/iterations as E1/E2 -- only the augmentation axis
varies, via 5 new AttackConfig entries in config.py:

    OSFD_ROT         -- rotation only
    OSFD_RESIZE      -- adaptive resize only
    OSFD_NOISE       -- additive Gaussian noise only
    OSFD_ROT_RESIZE  -- rotation + resize, no noise
    OSFD_RRB_FULL    -- rotation + resize + noise (= E2's config, re-run at
                        this pilot's own scale for a fair same-scale baseline)

Cheap pilot (this script's defaults): 30 images, T=50, evaluate white-box +
FCOS/YOLOX/DINO-Swin-L (Group A/B/C representatives) -- not the full 6
targets, matching the scale already used for Phase G0's Stage 2 screen.

    python scripts/run_b1_ablation.py --n-images 30 --n-iters 50 \
        --out results/B1_rrb_component_ablation.json
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

B1_VARIANTS = ["OSFD_ROT", "OSFD_RESIZE", "OSFD_NOISE", "OSFD_ROT_RESIZE", "OSFD_RRB_FULL"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-images", type=int, default=30)
    p.add_argument("--n-iters", type=int, default=50)
    p.add_argument("--manifest", default="dev_300")
    p.add_argument("--targets", default="fcos_r50,yolox_l,dino_swin_l")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    target_names = [t for t in args.targets.split(",") if t]

    experiments = make_experiments(0.05, 1)  # drop_prob/num_masks unused: all B1 configs mask_enabled=False
    subset = load_subset(args.n_images, args.manifest)
    label_to_cat_id = subset.label_to_cat_id()
    print(
        f"[b1] {len(subset)} images from manifest={args.manifest}, "
        f"variants={B1_VARIANTS}, targets={target_names}",
        flush=True,
    )

    results: dict = {}
    for variant in B1_VARIANTS:
        cfg = dataclasses.replace(experiments[variant].attack, max_iterations=args.n_iters)
        print(f"[b1] running {variant} cfg={cfg}", flush=True)
        t0 = time.time()

        surrogate = load_model(SURROGATE_NAME, args.device)
        adv_images, gt_by_image = generate_adversarial(surrogate, subset, cfg, args.device)
        del surrogate
        print(f"[b1] {variant} adversarial generation done in {time.time() - t0:.1f}s", flush=True)

        variant_results: dict = {
            SURROGATE_NAME: dataclasses.asdict(
                evaluate_on_model(
                    SURROGATE_NAME, args.device, subset, adv_images, gt_by_image, label_to_cat_id
                )
            )
        }
        for name in target_names:
            print(f"[b1] {variant}: evaluating {name}", flush=True)
            variant_results[name] = dataclasses.asdict(
                evaluate_on_model(name, args.device, subset, adv_images, gt_by_image, label_to_cat_id)
            )
        results[variant] = variant_results

    print("\n[b1] summary (white-box + avg black-box mAP-drop):")
    for variant in B1_VARIANTS:
        wb = results[variant][SURROGATE_NAME]["map_drop"]
        drops = [results[variant][t]["map_drop"] for t in target_names]
        bb = sum(drops) / len(drops)
        print(f"  {variant:<18} white_box_map_drop={wb:.3f}  avg_bb_map_drop={bb:.3f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_images": len(subset),
        "manifest": args.manifest,
        "n_iters": args.n_iters,
        "variants": B1_VARIANTS,
        "targets": target_names,
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[b1] wrote {out_path}")


if __name__ == "__main__":
    main()
