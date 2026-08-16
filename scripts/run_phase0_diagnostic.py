#!/usr/bin/env python3
"""Phase 0 diagnostic (idea.txt): does S_O > S_V hold on CLEAN images, across
backbone stages and architectures? No attack, no epsilon -- pure forward pass.

Answers, per (model, stage, r, metric): P(S_O > S_V) over the eval set, mean
S_O/S_V ratio, and the distribution of the bounded, scale-free contrast
    C = (S_O - S_V) / (S_O + S_V + eps)  in [-1, 1]
(mean/median/variance) -- P(O>V) alone can't distinguish "small but very
consistent margin" from "large but noisy margin"; C's spread does.

    python scripts/run_phase0_diagnostic.py --n-images 200 --out results/phase0_diagnostic.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch

from evasion_od.data import load_image_bgr
from evasion_od.models import backbone_features, load_model
from evasion_od.preprocessing import build_resized_input, preprocess_batch, scale_gt_boxes
from evasion_od.regions import (
    channel_stats,
    object_vicinity_masks,
    region_score_magnitude,
    region_score_normalized_energy,
)
from evasion_od.runner import load_subset

# Surrogate + all 6 cross-family targets from plan.md's Target Models table
# (config.py TARGET_NAMES), so Phase 0 covers exactly the same model set that
# Phase 1+ attack transfer will be evaluated on. fcos_r50/deformable_detr
# share the surrogate's ResNet-50 backbone (redundant for a backbone-only
# diagnostic) but are included anyway for 1:1 alignment with plan.md; cost is
# negligible (clean forward pass only, no attack).
DEFAULT_MODELS = [
    "faster_rcnn_r50_fpn",  # ResNet-50 (surrogate)
    "fcos_r50",  # ResNet-50 (Group A)
    "deformable_detr",  # ResNet-50 (Group A)
    "yolov3_d53",  # Darknet-53 (Group B)
    "yolox_l",  # CSPNet (Group B)
    "mask_rcnn_swin_t",  # Swin-T (Group C)
    "dino_swin_l",  # Swin-L (Group C)
]
R_VALUES = [0.5, 1.0, 2.0]
MIN_MARGIN_CELLS = 1
METRICS = {
    "magnitude": region_score_magnitude,
    "norm_energy": region_score_normalized_energy,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-images", type=int, default=200)
    p.add_argument("--manifest", default="dev_300")
    p.add_argument("--models", default=",".join(DEFAULT_MODELS))
    p.add_argument("--r-values", default=",".join(str(r) for r in R_VALUES))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", required=True)
    return p.parse_args()


def _new_bucket() -> dict:
    return {"n": 0, "n_wins": 0, "sum_ratio": 0.0, "contrasts": []}


def main() -> None:
    args = parse_args()
    model_names = [m for m in args.models.split(",") if m]
    r_values = [float(x) for x in args.r_values.split(",") if x]

    subset = load_subset(args.n_images, args.manifest)
    print(f"[phase0] {len(subset)} images from manifest={args.manifest}", flush=True)

    # stats[model][stage_idx][r][metric] -> {"n", "n_wins", "sum_ratio"}
    stats: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(_new_bucket))))
    n_images_skipped_no_gt: dict = defaultdict(int)
    n_regions_skipped_empty: dict = defaultdict(int)

    for model_name in model_names:
        t0 = time.time()
        model = load_model(model_name, args.device)
        n_stages = None

        for i, sample in enumerate(subset, start=1):
            img = load_image_bgr(sample.image_path)
            resized = build_resized_input(model, img, sample.image_id, args.device)
            gt = scale_gt_boxes(sample.gt_boxes, resized.data_sample, args.device)
            if gt.numel() == 0:
                n_images_skipped_no_gt[model_name] += 1
                continue

            with torch.no_grad():
                batch = preprocess_batch(model, resized.clean_chw, resized.data_sample)
                feats = backbone_features(model, batch["inputs"])
                img_h, img_w = batch["inputs"].shape[-2:]
                n_stages = len(feats)

                for stage_idx, feat in enumerate(feats):
                    f = feat[0]  # drop batch dim -> (C,H,W)
                    feat_h, feat_w = f.shape[-2:]
                    mu, sigma = channel_stats(f)

                    for r in r_values:
                        O_mask, V_mask = object_vicinity_masks(
                            gt, img_h, img_w, feat_h, feat_w, r, MIN_MARGIN_CELLS
                        )
                        if not O_mask.any() or not V_mask.any():
                            n_regions_skipped_empty[(model_name, stage_idx, r)] += 1
                            continue

                        for metric_name, score_fn in METRICS.items():
                            if metric_name == "magnitude":
                                s_o = score_fn(f, O_mask).item()
                                s_v = score_fn(f, V_mask).item()
                            else:
                                s_o = score_fn(f, O_mask, mu, sigma).item()
                                s_v = score_fn(f, V_mask, mu, sigma).item()
                            bucket = stats[model_name][stage_idx][r][metric_name]
                            bucket["n"] += 1
                            bucket["n_wins"] += int(s_o > s_v)
                            bucket["sum_ratio"] += s_o / (s_v + 1e-9)
                            bucket["contrasts"].append((s_o - s_v) / (s_o + s_v + 1e-9))

            if i % 50 == 0 or i == len(subset):
                print(f"  [{model_name}] image {i}/{len(subset)}", flush=True)

        del model
        torch.cuda.empty_cache()
        print(
            f"[phase0] {model_name} done in {time.time() - t0:.1f}s "
            f"({n_stages} backbone stages, {n_images_skipped_no_gt[model_name]} images skipped (no GT))",
            flush=True,
        )

    # Finalize: P(S_O>S_V) and mean ratio per (model, stage, r, metric)
    summary = []
    for model_name, by_stage in stats.items():
        for stage_idx, by_r in by_stage.items():
            for r, by_metric in by_r.items():
                for metric_name, b in by_metric.items():
                    if b["n"] == 0:
                        continue
                    c = b["contrasts"]
                    summary.append(
                        {
                            "model": model_name,
                            "stage": stage_idx,
                            "r": r,
                            "metric": metric_name,
                            "n": b["n"],
                            "p_o_gt_v": b["n_wins"] / b["n"],
                            "mean_ratio_o_over_v": b["sum_ratio"] / b["n"],
                            "mean_contrast": statistics.mean(c),
                            "median_contrast": statistics.median(c),
                            "var_contrast": statistics.pvariance(c) if len(c) > 1 else 0.0,
                            "n_regions_skipped_empty": n_regions_skipped_empty.get(
                                (model_name, stage_idx, r), 0
                            ),
                        }
                    )

    print("\n[phase0] summary (P(S_O>S_V), mean ratio, contrast C=(S_O-S_V)/(S_O+S_V)):")
    header = (
        f"{'model':<20}{'stage':>6}{'r':>6}{'metric':>14}{'n':>6}{'P(O>V)':>9}"
        f"{'ratio':>8}{'meanC':>8}{'medC':>8}{'varC':>8}"
    )
    print(header)
    for row in sorted(summary, key=lambda x: (x["model"], x["stage"], x["r"], x["metric"])):
        print(
            f"{row['model']:<20}{row['stage']:>6}{row['r']:>6}{row['metric']:>14}"
            f"{row['n']:>6}{row['p_o_gt_v']:>9.3f}{row['mean_ratio_o_over_v']:>8.3f}"
            f"{row['mean_contrast']:>8.3f}{row['median_contrast']:>8.3f}{row['var_contrast']:>8.4f}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_images": len(subset),
        "manifest": args.manifest,
        "models": model_names,
        "r_values": r_values,
        "min_margin_cells": MIN_MARGIN_CELLS,
        "images_skipped_no_gt": dict(n_images_skipped_no_gt),
        "summary": summary,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[phase0] wrote {out_path}")


if __name__ == "__main__":
    main()
