#!/usr/bin/env python3
"""Phase F0 -- Backbone Stage Ablation (plan.md).

After S3 closed the RRB-sampling-policy branch (NO-GO at N=100: history-aware
resampling added nothing over plain i.i.d.), this pivots from "how should RRB
randomize" to a different axis entirely: OSFD's loss sums its distortion term
over all 4 faster_rcnn_r50_fpn backbone stages unconditionally --

    L_OSFD = sum_{l=0}^{3} mean( (F_adv_l - k*F_clean_l)^2 )

Does every stage contribute transferable directions equally, or do some
stages pull the perturbation toward surrogate (ResNet-50)-specific
vulnerabilities while others carry signal that survives across architectures
(CSPNet/Darknet/Swin)? No augmentation here (RRB's per-iteration randomness
would confound which stage's *gradient direction* -- not just magnitude --
actually transfers; the S-phases already established augmentation and loss
structure are separable axes).

8 variants, `osfd_stage_weights` in AttackConfig (see losses.py:
backbone_feature_loss), 30 images / T=50 pilot, no RRB:

    OSFD_S0    -- stage 0 only (1,0,0,0)
    OSFD_S1    -- stage 1 only (0,1,0,0)
    OSFD_S2    -- stage 2 only (0,0,1,0)
    OSFD_S3    -- stage 3 (deepest) only (0,0,0,1)
    OSFD_S01   -- stages 0+1 (1,1,0,0)
    OSFD_S12   -- stages 1+2 (0,1,1,0)
    OSFD_S23   -- stages 2+3 (0,0,1,1)
    OSFD_ALL   -- every stage (= E1's config, re-run at this pilot's own
                  scale for a fair same-scale reference, not E1's original
                  50-image/T=100 numbers)

Reads as "transfer efficiency" per variant via BB/WB drop ratio (avg
black-box mAP-drop / white-box mAP-drop) in addition to the raw table --
a stage/subset with a *lower* white-box drop but a *higher* BB/WB ratio,
especially on YOLOX/DINO-Swin-L, is the signal this phase is looking for
(stage-specific surrogate overfitting: that stage buys white-box strength
that doesn't transfer).

GO: a stage/subset beats OSFD_ALL on Group B/C (YOLOX/DINO-Swin-L) mAP-drop,
or shows a clearly higher BB/WB ratio than OSFD_ALL. NO-GO: OSFD_ALL remains
best everywhere, or stage ranking is noisy/inconsistent across targets --
close feature-stage selection as a branch.

    python scripts/run_f0_stage_ablation.py --n-images 30 --n-iters 50 \
        --out results/F0_stage_ablation.json

Pilot (N=30/T=50) result: OSFD_S2 alone beat OSFD_ALL on avg BB (+5%),
YOLOX (+32%), and DINO-Swin-L (+71%), losing only slightly on FCOS (-15%)
and white-box (-5%) -- results/F0_stage_ablation.json, plan.md Phase F0.
Confirm at N=100 with only the two variants that matter for that question
via `--variants` (OSFD_ALL re-run fresh at N=100, not reused from the
pilot's N=30 numbers -- same reasoning as Phase S3's confirm):

    python scripts/run_f0_stage_ablation.py --n-images 100 --n-iters 50 \
        --variants OSFD_S2,OSFD_ALL \
        --out results/F0_confirm_n100.json
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

ALL_F0_VARIANTS = [
    "OSFD_S0",
    "OSFD_S1",
    "OSFD_S2",
    "OSFD_S3",
    "OSFD_S01",
    "OSFD_S12",
    "OSFD_S23",
    "OSFD_ALL",
]
DEFAULT_VARIANTS = ",".join(ALL_F0_VARIANTS)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-images", type=int, default=30)
    p.add_argument("--n-iters", type=int, default=50)
    p.add_argument("--manifest", default="dev_300")
    p.add_argument("--targets", default="fcos_r50,yolox_l,dino_swin_l")
    p.add_argument("--variants", default=DEFAULT_VARIANTS)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    target_names = [t for t in args.targets.split(",") if t]
    F0_VARIANTS = [v for v in args.variants.split(",") if v]
    unknown = [v for v in F0_VARIANTS if v not in ALL_F0_VARIANTS]
    if unknown:
        raise ValueError(f"unknown --variants entries: {unknown}, must be from {ALL_F0_VARIANTS}")

    experiments = make_experiments(0.05, 1)  # drop_prob/num_masks unused: all F0 configs mask_enabled=False
    subset = load_subset(args.n_images, args.manifest)
    label_to_cat_id = subset.label_to_cat_id()
    print(
        f"[f0] {len(subset)} images from manifest={args.manifest}, "
        f"variants={F0_VARIANTS}, targets={target_names}",
        flush=True,
    )

    results: dict = {}
    for variant in F0_VARIANTS:
        cfg = dataclasses.replace(experiments[variant].attack, max_iterations=args.n_iters)
        print(f"[f0] running {variant} cfg={cfg}", flush=True)
        t0 = time.time()

        surrogate = load_model(SURROGATE_NAME, args.device)
        adv_images, gt_by_image = generate_adversarial(surrogate, subset, cfg, args.device)
        del surrogate
        print(f"[f0] {variant} adversarial generation done in {time.time() - t0:.1f}s", flush=True)

        variant_results: dict = {
            SURROGATE_NAME: dataclasses.asdict(
                evaluate_on_model(
                    SURROGATE_NAME, args.device, subset, adv_images, gt_by_image, label_to_cat_id
                )
            )
        }
        for name in target_names:
            print(f"[f0] {variant}: evaluating {name}", flush=True)
            variant_results[name] = dataclasses.asdict(
                evaluate_on_model(name, args.device, subset, adv_images, gt_by_image, label_to_cat_id)
            )
        results[variant] = variant_results

    print("\n[f0] summary (white-box + per-target + avg black-box mAP-drop + BB/WB ratio):")
    header = (
        f"{'variant':<10}{'white_box':>10}"
        + "".join(f"{t:>14}" for t in target_names)
        + f"{'avg_bb':>9}{'bb/wb':>8}"
    )
    print(header)
    for variant in F0_VARIANTS:
        r = results[variant]
        wb = r[SURROGATE_NAME]["map_drop"]
        per_t = [r[t]["map_drop"] for t in target_names]
        avg = sum(per_t) / len(per_t)
        ratio = avg / wb if wb > 1e-9 else float("nan")
        row = (
            f"{variant:<10}{wb:>10.3f}"
            + "".join(f"{r[t]['map_drop']:>14.3f}" for t in target_names)
            + f"{avg:>9.3f}{ratio:>8.3f}"
        )
        print(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_images": len(subset),
        "manifest": args.manifest,
        "n_iters": args.n_iters,
        "variants": F0_VARIANTS,
        "targets": target_names,
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[f0] wrote {out_path}")


if __name__ == "__main__":
    main()
