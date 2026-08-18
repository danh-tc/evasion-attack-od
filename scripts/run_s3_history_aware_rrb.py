#!/usr/bin/env python3
"""Phase S3 -- History-Aware RRB (plan.md candidate).

G0->B1->B2->S0->S1->S2 all point the same way: RRB's transferability comes
from per-iteration randomness/diversity, not from converging on or holding a
single "right" operating point (S2: RANDOM_SHRINK beat FIXED_SHRINK on every
target, often by 2-3x). The open question this phase asks: is *all*
diversity equally good, or does trajectory-level structure -- specifically,
not repeating a transform too close to what was just used -- add anything on
top of plain i.i.d. resampling, as long as the sampling range stays within
RRB_ORIG's own safe region (theta<=7deg, occupancy in [0.91,1.0], matching
RRB_ORIG's own effective resize range, *not* S1's wider/aggressive [0.7,1.0]
shrink range that underperformed RRB_ORIG on every target)?

Three arms, same rotation+resize+noise structure as full RRB (not
resize-only like S0/S1/S2 -- this phase isn't isolating a component, it's
testing a *sampling policy* over the whole RRB parameter vector):

    RRB_IID          -- each draw i.i.d. Uniform over the safe range (the
                         null: same distribution as RRB_ORIG's effective
                         range, just parameterized directly instead of via
                         adaptive_random_resizing's GT-box-relative logic)
    RRB_ANTI_REPEAT  -- reject/resample a draw if it's within `history_tau`
                         of any of the last `history_window` draws (up to
                         `history_max_tries`, else keep the least-similar
                         attempt seen)
    RRB_OVER_DIVERSE -- positive control: draw `history_k_candidates`
                         candidates, keep whichever is farthest (by the same
                         distance metric) from the last `history_window`
                         draws

scripts/sanity_check_history_aware_rrb.py sized `history_tau` (0.44, this
pilot's AttackConfig default) *before* running this: at that threshold,
~26% of plain i.i.d. draws already land within tau of their own last 3 --
large enough that ANTI_REPEAT's rejection behavior actually differs from
IID, not a near-certain no-op.

GO if RRB_ANTI_REPEAT clearly beats RRB_IID on avg BB (>=2/3 of
FCOS/YOLOX/DINO-Swin-L), especially DINO-Swin-L, without falling below
RRB_ORIG (OSFD_RRB_FULL, reused from results/B1_rrb_component_ablation.json,
same 30-image/T=50 pilot scale -- not rerun). NO-GO if RRB_ANTI_REPEAT <=
RRB_IID, or gains require RRB_OVER_DIVERSE-level aggressiveness (which the
S1 wide-shrink result already suggests would hurt) -- close the branch,
RRB's plain i.i.d. resampling is already sufficient.

    python scripts/run_s3_history_aware_rrb.py --n-images 30 --n-iters 50 \
        --out results/S3_history_aware_rrb.json

Pilot (N=30/T=50) result: borderline -- ANTI_REPEAT beat IID on YOLOX/DINO-
Swin-L (DINO +9% vs IID, +14% vs RRB_ORIG) but lost on FCOS, avg BB only
+1.7% over IID (results/S3_history_aware_rrb.json, plan.md Phase S3).
OVER_DIVERSE (positive control) didn't consistently beat ANTI_REPEAT --
its role is done, no need to rerun it at confirm scale. Confirm run drops
it via `--variants` and reruns RRB_ORIG fresh instead of reusing B1's N=30
reference (scale mismatch otherwise):

    python scripts/run_s3_history_aware_rrb.py --n-images 100 --n-iters 50 \
        --variants RRB_ORIG,RRB_IID,RRB_ANTI_REPEAT \
        --out results/S3_confirm_n100.json
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

DEFAULT_VARIANTS = "RRB_IID,RRB_ANTI_REPEAT,RRB_OVER_DIVERSE"
# "RRB_ORIG" isn't its own AttackConfig -- it's the same config as B1's
# OSFD_RRB_FULL (plain augmentation="rrb"). Included in --variants so a
# confirm run at a scale that doesn't match B1_REFERENCE_FILE's 30/T=50 can
# generate a same-scale RRB_ORIG instead of reusing a mismatched reference.
VARIANT_TO_CONFIG_NAME = {
    "RRB_IID": "RRB_IID",
    "RRB_ANTI_REPEAT": "RRB_ANTI_REPEAT",
    "RRB_OVER_DIVERSE": "RRB_OVER_DIVERSE",
    "RRB_ORIG": "OSFD_RRB_FULL",
}
B1_REFERENCE_FILE = "results/B1_rrb_component_ablation.json"
B1_REFERENCE_KEY = "OSFD_RRB_FULL"


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


def _load_reference(path_str: str, key: str, target_names: list[str], label: str) -> dict | None:
    path = Path(path_str)
    if not path.exists():
        print(f"[s3] WARNING: {path} not found -- {label} reference omitted", flush=True)
        return None
    data = json.loads(path.read_text())
    if data["n_images"] != 30 or data["n_iters"] != 50:
        print(
            f"[s3] WARNING: {path} scale (n_images={data['n_images']}, n_iters={data['n_iters']}) "
            f"doesn't match this pilot's default -- {label} reference may not be directly comparable",
            flush=True,
        )
    ref = data["results"][key]
    missing = [t for t in target_names if t not in ref]
    if missing:
        print(f"[s3] WARNING: {path} missing targets {missing} for {label}", flush=True)
    return ref


def main() -> None:
    args = parse_args()
    target_names = [t for t in args.targets.split(",") if t]
    requested = [v for v in args.variants.split(",") if v]
    unknown = [v for v in requested if v not in VARIANT_TO_CONFIG_NAME]
    if unknown:
        raise ValueError(f"unknown --variants entries: {unknown}, must be from {list(VARIANT_TO_CONFIG_NAME)}")
    run_orig_fresh = "RRB_ORIG" in requested

    experiments = make_experiments(0.05, 1)  # drop_prob/num_masks unused: mask_enabled=False for S3
    subset = load_subset(args.n_images, args.manifest)
    label_to_cat_id = subset.label_to_cat_id()
    print(
        f"[s3] {len(subset)} images from manifest={args.manifest}, "
        f"variants={requested}, targets={target_names}",
        flush=True,
    )

    results: dict = {}
    if not run_orig_fresh:
        rrb_orig = _load_reference(B1_REFERENCE_FILE, B1_REFERENCE_KEY, target_names, "RRB_ORIG")
        if rrb_orig is not None:
            results["RRB_ORIG"] = rrb_orig

    for variant in requested:
        cfg_name = VARIANT_TO_CONFIG_NAME[variant]
        cfg = dataclasses.replace(experiments[cfg_name].attack, max_iterations=args.n_iters)
        print(f"[s3] running {variant} cfg={cfg}", flush=True)
        t0 = time.time()

        surrogate = load_model(SURROGATE_NAME, args.device)
        adv_images, gt_by_image = generate_adversarial(surrogate, subset, cfg, args.device)
        del surrogate
        print(f"[s3] {variant} adversarial generation done in {time.time() - t0:.1f}s", flush=True)

        variant_results: dict = {
            SURROGATE_NAME: dataclasses.asdict(
                evaluate_on_model(
                    SURROGATE_NAME, args.device, subset, adv_images, gt_by_image, label_to_cat_id
                )
            )
        }
        for name in target_names:
            print(f"[s3] {variant}: evaluating {name}", flush=True)
            variant_results[name] = dataclasses.asdict(
                evaluate_on_model(name, args.device, subset, adv_images, gt_by_image, label_to_cat_id)
            )
        results[variant] = variant_results

    order = (["RRB_ORIG"] if "RRB_ORIG" in results and not run_orig_fresh else []) + requested
    print("\n[s3] summary (white-box + per-target + avg black-box mAP-drop):")
    header = f"{'variant':<18}{'white_box':>10}" + "".join(f"{t:>14}" for t in target_names) + f"{'avg_bb':>9}"
    print(header)
    for variant in order:
        if variant not in results:
            continue
        r = results[variant]
        wb = r[SURROGATE_NAME]["map_drop"]
        per_t = [r[t]["map_drop"] for t in target_names if t in r]
        avg = sum(per_t) / len(per_t) if per_t else float("nan")
        row = (
            f"{variant:<18}{wb:>10.3f}"
            + "".join(f"{r[t]['map_drop']:>14.3f}" for t in target_names if t in r)
            + f"{avg:>9.3f}"
        )
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
    print(f"\n[s3] wrote {out_path}")


if __name__ == "__main__":
    main()
