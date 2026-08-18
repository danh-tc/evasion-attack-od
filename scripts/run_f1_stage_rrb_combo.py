#!/usr/bin/env python3
"""Phase F1 -- Best Stage + RRB vs Full-Stage + RRB (plan.md).

Phase F0 confirmed (N=30 pilot AND N=100 confirm, no augmentation): attacking
only backbone stage 2 (`OSFD_S2`) beats attacking all 4 stages (`OSFD_ALL`)
on avg black-box mAP-drop and especially DINO-Swin-L, without reversing on
any target between the two scales -- a real structural signal, unlike Phase
S3's history-aware-RRB pilot which evaporated at N=100.

Decisive question this phase asks: does stage-2-only's advantage *stack* on
top of RRB (the project's strongest single lever, E1->E2's jump is still the
largest measured in this project), or does RRB swamp it the way it swamped
RaPA-mask (I4 vs E2, plan.md section 1 -- I4≈E2, no compounding)? Two arms,
same loss/eps/alpha/iterations, only `osfd_stage_weights` differs:

    E2_ALL_RRB -- OSFD (all 4 stages) + RRB (= E2's own config, already
                  named "E2_osfd_rrb" in config.py)
    S2_RRB     -- OSFD stage 2 only + RRB (new "S2_RRB" config)

At the N=30/T=50 pilot scale, E2_ALL_RRB is reused from Phase B1's
`OSFD_RRB_FULL` (results/B1_rrb_component_ablation.json -- identical config,
same reasoning as every S-phase script's reference reuse) instead of
re-running it, so the pilot only spends compute on S2_RRB.

GO: S2_RRB beats E2_ALL_RRB on avg BB and/or DINO-Swin-L without falling
much below it elsewhere. NO-GO: E2_ALL_RRB stays best everywhere (RRB
swamps the stage-2 advantage, matching I4's fate) -- close Phase F on this
note, stage-2-only is a no-augmentation-only finding.

    python scripts/run_f1_stage_rrb_combo.py --n-images 30 --n-iters 50 \
        --out results/F1_stage_rrb_combo.json

Confirm at N=100 (both arms fresh, B1's N=30 reference no longer valid):

    python scripts/run_f1_stage_rrb_combo.py --n-images 100 --n-iters 50 \
        --variants E2_ALL_RRB,S2_RRB \
        --out results/F1_confirm_n100.json
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

VARIANT_TO_CONFIG_NAME = {
    "E2_ALL_RRB": "E2_osfd_rrb",
    "S2_RRB": "S2_RRB",
}
DEFAULT_VARIANTS = "S2_RRB"  # E2_ALL_RRB reused from B1 by default -- see module docstring
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
        print(f"[f1] WARNING: {path} not found -- {label} reference omitted", flush=True)
        return None
    data = json.loads(path.read_text())
    if data["n_images"] != 30 or data["n_iters"] != 50:
        print(
            f"[f1] WARNING: {path} scale (n_images={data['n_images']}, n_iters={data['n_iters']}) "
            f"doesn't match this pilot's default -- {label} reference may not be directly comparable",
            flush=True,
        )
    ref = data["results"][key]
    missing = [t for t in target_names if t not in ref]
    if missing:
        print(f"[f1] WARNING: {path} missing targets {missing} for {label}", flush=True)
    return ref


def main() -> None:
    args = parse_args()
    target_names = [t for t in args.targets.split(",") if t]
    requested = [v for v in args.variants.split(",") if v]
    unknown = [v for v in requested if v not in VARIANT_TO_CONFIG_NAME]
    if unknown:
        raise ValueError(f"unknown --variants entries: {unknown}, must be from {list(VARIANT_TO_CONFIG_NAME)}")
    run_all_rrb_fresh = "E2_ALL_RRB" in requested

    experiments = make_experiments(0.05, 1)  # drop_prob/num_masks unused: both F1 configs mask_enabled=False
    subset = load_subset(args.n_images, args.manifest)
    label_to_cat_id = subset.label_to_cat_id()
    print(
        f"[f1] {len(subset)} images from manifest={args.manifest}, "
        f"variants={requested}, targets={target_names}",
        flush=True,
    )

    results: dict = {}
    if not run_all_rrb_fresh:
        ref = _load_reference(B1_REFERENCE_FILE, B1_REFERENCE_KEY, target_names, "E2_ALL_RRB")
        if ref is not None:
            results["E2_ALL_RRB"] = ref

    for variant in requested:
        cfg_name = VARIANT_TO_CONFIG_NAME[variant]
        cfg = dataclasses.replace(experiments[cfg_name].attack, max_iterations=args.n_iters)
        print(f"[f1] running {variant} cfg={cfg}", flush=True)
        t0 = time.time()

        surrogate = load_model(SURROGATE_NAME, args.device)
        adv_images, gt_by_image = generate_adversarial(surrogate, subset, cfg, args.device)
        del surrogate
        print(f"[f1] {variant} adversarial generation done in {time.time() - t0:.1f}s", flush=True)

        variant_results: dict = {
            SURROGATE_NAME: dataclasses.asdict(
                evaluate_on_model(
                    SURROGATE_NAME, args.device, subset, adv_images, gt_by_image, label_to_cat_id
                )
            )
        }
        for name in target_names:
            print(f"[f1] {variant}: evaluating {name}", flush=True)
            variant_results[name] = dataclasses.asdict(
                evaluate_on_model(name, args.device, subset, adv_images, gt_by_image, label_to_cat_id)
            )
        results[variant] = variant_results

    order = (["E2_ALL_RRB"] if "E2_ALL_RRB" in results and not run_all_rrb_fresh else []) + requested
    print("\n[f1] summary (white-box + per-target + avg black-box mAP-drop):")
    header = f"{'variant':<14}{'white_box':>10}" + "".join(f"{t:>14}" for t in target_names) + f"{'avg_bb':>9}"
    print(header)
    for variant in order:
        if variant not in results:
            continue
        r = results[variant]
        wb = r[SURROGATE_NAME]["map_drop"]
        per_t = [r[t]["map_drop"] for t in target_names if t in r]
        avg = sum(per_t) / len(per_t) if per_t else float("nan")
        row = (
            f"{variant:<14}{wb:>10.3f}"
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
    print(f"\n[f1] wrote {out_path}")


if __name__ == "__main__":
    main()
