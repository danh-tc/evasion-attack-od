#!/usr/bin/env python3
"""Phase 1b diagnostic: object-to-context feature MISALIGNMENT hypothesis.

Phase 1 (results/phase0_diagnostic.json) validated a coarse, mean-pooled
S_O vs S_V relation; Phase 2 (results/P2_rel_*_go.json, P2_hybrid_*_go.json)
showed attacking that mean-pooled scalar is too coarse -- flipping it barely
denting even white-box detection. Before writing a per-pixel attack loss
around it, check the underlying PER-PIXEL clean-image property actually
holds (same cheap-diagnostic-before-expensive-attack discipline as Phase 1):

    mu_O, mu_V = mean clean feature vector over O, V regions (prototypes)
    for each object pixel p in O:
        margin(p) = cos(F(p), mu_O) - cos(F(p), mu_V)

Does margin(p) > 0 hold consistently (object pixels look more like their own
region's prototype than the vicinity's), across backbone stages and
architectures? No attack, no epsilon -- pure forward pass, same 7-model set
and r-sweep as Phase 1 for direct comparability.

    python scripts/run_p1b_diagnostic.py --n-images 200 --out results/P1b_prototype_diagnostic.json
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
from evasion_od.regions import object_vicinity_masks, pixelwise_cosine_margin, region_mean_vector
from evasion_od.runner import load_subset

# Same model set as Phase 1 (results/phase0_diagnostic.json) for direct comparison.
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
    return {"n_images": 0, "n_images_positive": 0, "n_pixels": 0, "margins": []}


def main() -> None:
    args = parse_args()
    model_names = [m for m in args.models.split(",") if m]
    r_values = [float(x) for x in args.r_values.split(",") if x]

    subset = load_subset(args.n_images, args.manifest)
    print(f"[p1b] {len(subset)} images from manifest={args.manifest}", flush=True)

    # stats[model][stage_idx][r] -> {"n_images", "n_images_positive", "n_pixels", "margins": [per-image mean margin]}
    stats: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(_new_bucket)))
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

                    for r in r_values:
                        O_mask, V_mask = object_vicinity_masks(
                            gt, img_h, img_w, feat_h, feat_w, r, MIN_MARGIN_CELLS
                        )
                        if not O_mask.any() or not V_mask.any():
                            n_regions_skipped_empty[(model_name, stage_idx, r)] += 1
                            continue

                        mu_o = region_mean_vector(f, O_mask)
                        mu_v = region_mean_vector(f, V_mask)
                        pixel_margins = pixelwise_cosine_margin(f, O_mask, mu_o, mu_v)
                        image_mean_margin = pixel_margins.mean().item()

                        bucket = stats[model_name][stage_idx][r]
                        bucket["n_images"] += 1
                        bucket["n_images_positive"] += int(image_mean_margin > 0)
                        bucket["n_pixels"] += pixel_margins.numel()
                        bucket["margins"].append(image_mean_margin)

            if i % 50 == 0 or i == len(subset):
                print(f"  [{model_name}] image {i}/{len(subset)}", flush=True)

        del model
        torch.cuda.empty_cache()
        print(
            f"[p1b] {model_name} done in {time.time() - t0:.1f}s "
            f"({n_stages} backbone stages, {n_images_skipped_no_gt[model_name]} images skipped (no GT))",
            flush=True,
        )

    # Finalize: P(mean_margin_image > 0), mean/median/var of the per-image mean margin
    summary = []
    for model_name, by_stage in stats.items():
        for stage_idx, by_r in by_stage.items():
            for r, b in by_r.items():
                if b["n_images"] == 0:
                    continue
                m = b["margins"]
                summary.append(
                    {
                        "model": model_name,
                        "stage": stage_idx,
                        "r": r,
                        "n_images": b["n_images"],
                        "n_pixels_total": b["n_pixels"],
                        "avg_pixels_per_image": b["n_pixels"] / b["n_images"],
                        "p_margin_gt_0": b["n_images_positive"] / b["n_images"],
                        "mean_margin": statistics.mean(m),
                        "median_margin": statistics.median(m),
                        "var_margin": statistics.pvariance(m) if len(m) > 1 else 0.0,
                        "n_regions_skipped_empty": n_regions_skipped_empty.get(
                            (model_name, stage_idx, r), 0
                        ),
                    }
                )

    print("\n[p1b] summary (P(margin>0), mean/median/var of per-image mean margin):")
    header = (
        f"{'model':<20}{'stage':>6}{'r':>6}{'n_img':>7}{'px/img':>8}"
        f"{'P(>0)':>8}{'meanM':>8}{'medM':>8}{'varM':>8}"
    )
    print(header)
    for row in sorted(summary, key=lambda x: (x["model"], x["stage"], x["r"])):
        print(
            f"{row['model']:<20}{row['stage']:>6}{row['r']:>6}{row['n_images']:>7}"
            f"{row['avg_pixels_per_image']:>8.1f}{row['p_margin_gt_0']:>8.3f}"
            f"{row['mean_margin']:>8.3f}{row['median_margin']:>8.3f}{row['var_margin']:>8.4f}"
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
    print(f"\n[p1b] wrote {out_path}")


if __name__ == "__main__":
    main()
