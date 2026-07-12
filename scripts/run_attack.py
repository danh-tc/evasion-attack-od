#!/usr/bin/env python3
"""Run one attack config (E1..E5, see plan.md) on the surrogate and evaluate
white-box + all cross-family targets. Also serves as the project smoke test:

    python scripts/run_attack.py --n-images 5 --n-iters 5 --out results/smoke.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

from evasion_od.config import SURROGATE_NAME, TARGET_NAMES, AttackConfig, make_experiments
from evasion_od.runner import evaluate_on_model, generate_adversarial, load_subset
from evasion_od.models import load_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-images", type=int, default=50)
    p.add_argument("--n-iters", type=int, default=100)
    p.add_argument("--manifest", default="dev_300")
    p.add_argument(
        "--experiment",
        default="E1_osfd_baseline",
        choices=list(make_experiments(0.05, 1).keys()),
    )
    p.add_argument("--drop-prob", type=float, default=0.05, help="p, used by E4/E5")
    p.add_argument("--masks", type=int, default=1, help="S, used by E4/E5")
    p.add_argument("--targets", default=",".join(TARGET_NAMES))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    experiments = make_experiments(args.drop_prob, args.masks)
    cfg: AttackConfig = experiments[args.experiment].attack
    cfg = dataclasses.replace(cfg, max_iterations=args.n_iters)

    subset = load_subset(args.n_images, args.manifest)
    label_to_cat_id = subset.label_to_cat_id()

    print(f"[run_attack] experiment={args.experiment} cfg={cfg}")
    print(f"[run_attack] {len(subset)} images from manifest={args.manifest}")

    t0 = time.time()
    surrogate = load_model(SURROGATE_NAME, args.device)
    adv_images, gt_by_image = generate_adversarial(surrogate, subset, cfg, args.device)
    print(f"[run_attack] adversarial generation done in {time.time() - t0:.1f}s")

    results = {}

    print(f"[run_attack] evaluating white-box ({SURROGATE_NAME})")
    results[SURROGATE_NAME] = dataclasses.asdict(
        evaluate_on_model(SURROGATE_NAME, args.device, subset, adv_images, gt_by_image, label_to_cat_id)
    )
    del surrogate

    target_names = [t for t in args.targets.split(",") if t]
    for name in target_names:
        print(f"[run_attack] evaluating black-box target: {name}")
        t1 = time.time()
        results[name] = dataclasses.asdict(
            evaluate_on_model(name, args.device, subset, adv_images, gt_by_image, label_to_cat_id)
        )
        print(f"  done in {time.time() - t1:.1f}s -> {results[name]}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": args.experiment,
        "attack_config": dataclasses.asdict(cfg),
        "n_images": len(subset),
        "manifest": args.manifest,
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[run_attack] wrote {out_path}")


if __name__ == "__main__":
    main()
