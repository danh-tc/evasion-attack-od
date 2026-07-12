#!/usr/bin/env python3
"""Pilot Study: sweep RaPA DropConnect (p, S) on the backbone BN affine params.

Fixed: surrogate Faster R-CNN R50-FPN, loss NRDM (k=1), no RRB, MI-FGSM.
Evaluated on white-box (surrogate) + 2 representative targets (FCOS-R50 for
Group A, Mask R-CNN Swin-T for Group C). See plan.md "0. Pilot Study".

    python scripts/run_sweep.py --n-images 50 --rates 0.02,0.05,0.1,0.15,0.2 \
        --masks 1,3,5 --n-iters 100 --out results/pilot_sweep.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

from evasion_od.config import PILOT_TARGET_NAMES, SURROGATE_NAME, AttackConfig
from evasion_od.models import load_model
from evasion_od.runner import evaluate_on_model, generate_adversarial, load_subset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-images", type=int, default=50)
    p.add_argument("--n-iters", type=int, default=100)
    p.add_argument("--manifest", default="dev_300")
    p.add_argument("--rates", default="0.02,0.05,0.1,0.15,0.2", help="comma list of drop_prob (p)")
    p.add_argument("--masks", default="1,3,5", help="comma list of num_masks (S)")
    p.add_argument("--no-baseline", action="store_true", help="skip the no-mask control run")
    p.add_argument("--targets", default=",".join(PILOT_TARGET_NAMES))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", required=True)
    return p.parse_args()


def run_one(cfg, subset, label_to_cat_id, target_names, device) -> dict:
    surrogate = load_model(SURROGATE_NAME, device)
    adv_images, gt_by_image = generate_adversarial(surrogate, subset, cfg, device)

    out = {
        SURROGATE_NAME: dataclasses.asdict(
            evaluate_on_model(SURROGATE_NAME, device, subset, adv_images, gt_by_image, label_to_cat_id)
        )
    }
    del surrogate
    for name in target_names:
        out[name] = dataclasses.asdict(
            evaluate_on_model(name, device, subset, adv_images, gt_by_image, label_to_cat_id)
        )
    return out


def main() -> None:
    args = parse_args()
    rates = [float(x) for x in args.rates.split(",") if x]
    masks = [int(x) for x in args.masks.split(",") if x]
    target_names = [t for t in args.targets.split(",") if t]

    subset = load_subset(args.n_images, args.manifest)
    label_to_cat_id = subset.label_to_cat_id()
    print(f"[run_sweep] {len(subset)} images, rates={rates}, masks={masks}")

    grid: list[dict] = []

    if not args.no_baseline:
        print("[run_sweep] running no-mask baseline (NRDM control)")
        cfg = AttackConfig(k=1.0, mask_enabled=False, max_iterations=args.n_iters)
        t0 = time.time()
        results = run_one(cfg, subset, label_to_cat_id, target_names, args.device)
        grid.append({"p": None, "S": None, "results": results, "elapsed_s": time.time() - t0})
        print(f"  done in {time.time() - t0:.1f}s -> {results}")

    for p in rates:
        for s in masks:
            print(f"[run_sweep] p={p} S={s}")
            cfg = AttackConfig(
                k=1.0, mask_enabled=True, drop_prob=p, num_masks=s, max_iterations=args.n_iters
            )
            t0 = time.time()
            results = run_one(cfg, subset, label_to_cat_id, target_names, args.device)
            grid.append({"p": p, "S": s, "results": results, "elapsed_s": time.time() - t0})
            print(f"  done in {time.time() - t0:.1f}s -> {results}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {"n_images": len(subset), "manifest": args.manifest, "targets": target_names, "grid": grid},
            f,
            indent=2,
        )
    print(f"[run_sweep] wrote {out_path}")


if __name__ == "__main__":
    main()
