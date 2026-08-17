#!/usr/bin/env python3
"""Phase B2 -- Augmentation Transfer Signature (plan.md).

B1 (RRB Component Ablation) found transfer requirement is architecture-
difficulty dependent: resize alone nearly saturates transfer to Group A/B
(same/near-family CNN targets), but Group C (DINO-Swin-L, the hardest
cross-family target) specifically needs the full rotation+resize+noise
composition -- no partial subset without noise gets there. In a real
black-box attack the target architecture is unknown, so a method can't just
special-case "target looks hard -> add noise". This asks whether B1's
per-augmentation transfer behavior is predictable from surrogate-side-only
signal, cheaply, on clean images (no attack trajectory, no MI-FGSM loop --
one forward/backward per draw at delta=0):

  - gradient alignment: does augmentation a's gradient still point toward
    the raw (unaugmented) attack direction, or does it diverge?
  - feature distortion stability: how consistent is the feature-space
    distortion a induces, across K independent draws of the same kind?
  - loss sensitivity: how much (and how consistently) does a shift the OSFD
    loss relative to no augmentation?

If one of these ranks {rot, resize, noise} the way B1's measured transfer
does, it's a candidate signal Q(a) for adaptively weighting augmentation
composition (p(a|x) ~ f(Q(a))) instead of RRB's fixed pipeline.

Scope notes (see plan.md Phase B2): measured at delta=0 (intrinsic
per-augmentation behavior on the clean image, not trajectory-aware);
characterizes each kind standalone vs "none", not pairwise
interactions/combinations -- so this is better suited to explain B1's
single-component ranking (RESIZE>ROT>NOISE) than the combination-specific
finding (noise's marginal value only when added to rot+resize, mainly on
Group C). Only 3 ranked kinds -> Spearman/Pearson have n=3, negligible
statistical power; read as a qualitative ranking match.

    python scripts/run_b2_diagnostic.py --n-images 30 --k 5 \
        --out results/B2_augmentation_transfer_signature.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from evasion_od.config import SURROGATE_NAME
from evasion_od.data import load_image_bgr
from evasion_od.gradient_diagnostics import pairwise_cosine_stats
from evasion_od.losses import backbone_feature_loss
from evasion_od.models import backbone_features, load_model
from evasion_od.preprocessing import build_resized_input, preprocess_batch, scale_gt_boxes
from evasion_od.rrb import additive_gaussian_noise, apply_resize_only, random_axis_rotation
from evasion_od.runner import load_subset

AUG_KINDS = ["rot", "resize", "noise"]  # "none" is the deterministic reference, not ranked
METRIC_KEYS = [
    "gradient_alignment",
    "feature_stability_mean_cos",
    "feature_stability_median_cos",
    "loss_sensitivity_mean",
    "loss_sensitivity_std",
]


def _apply(kind: str, adv: torch.Tensor, gt_resized: torch.Tensor) -> torch.Tensor:
    if kind == "none":
        return adv
    if kind == "rot":
        return random_axis_rotation(adv, gt_resized)
    if kind == "resize":
        return apply_resize_only(adv, gt_resized)
    if kind == "noise":
        return additive_gaussian_noise(adv)
    raise ValueError(f"unknown kind: {kind!r}")


def _one_draw(model, clean_chw, gt_resized, data_sample, feats_clean, kind, loss_k):
    """One forward/backward at delta=0. Returns (loss, grad (C,H,W), deepest-stage feats_adv (C,H,W))."""
    delta = torch.zeros_like(clean_chw, requires_grad=True)
    adv = torch.clamp(clean_chw + delta, 0.0, 255.0)
    adv = _apply(kind, adv, gt_resized)
    batch = preprocess_batch(model, adv, data_sample)
    feats_adv = backbone_features(model, batch["inputs"])
    loss = backbone_feature_loss(feats_adv, feats_clean, loss_k)
    (grad,) = torch.autograd.grad(loss, delta)
    return float(loss.item()), grad.detach(), feats_adv[-1][0].detach()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-images", type=int, default=30)
    p.add_argument("--manifest", default="dev_300")
    p.add_argument("--k", type=int, default=5, help="independent draws per stochastic augmentation kind")
    p.add_argument("--loss-k", type=float, default=3.0, help="OSFD amplification factor, matches B1")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    subset = load_subset(args.n_images, args.manifest)
    print(f"[b2] {len(subset)} images from manifest={args.manifest}, K={args.k}", flush=True)

    surrogate = load_model(SURROGATE_NAME, args.device)

    per_image: list[dict] = []
    t0 = time.time()
    for i, sample in enumerate(subset, start=1):
        img = load_image_bgr(sample.image_path)
        resized = build_resized_input(surrogate, img, sample.image_id, args.device)
        clean_chw = resized.clean_chw
        gt_resized = scale_gt_boxes(sample.gt_boxes, resized.data_sample, args.device)

        with torch.no_grad():
            clean_batch = preprocess_batch(surrogate, clean_chw, resized.data_sample)
            feats_clean = tuple(f.detach() for f in backbone_features(surrogate, clean_batch["inputs"]))

        loss_none, grad_none, feat_none = _one_draw(
            surrogate, clean_chw, gt_resized, resized.data_sample, feats_clean, "none", args.loss_k
        )
        grad_none_flat = grad_none.flatten()
        grad_none_unit = grad_none_flat / (grad_none_flat.norm() + 1e-12)

        image_result: dict = {"image_id": sample.image_id, "loss_none": loss_none}
        for kind in AUG_KINDS:
            draws = [
                _one_draw(surrogate, clean_chw, gt_resized, resized.data_sample, feats_clean, kind, args.loss_k)
                for _ in range(args.k)
            ]
            losses = torch.tensor([d[0] for d in draws])
            grads = torch.stack([d[1] for d in draws], dim=0)  # (K,C,H,W)
            feats = torch.stack([d[2] for d in draws], dim=0)  # (K,C,H,W), deepest stage

            grads_flat = grads.reshape(args.k, -1)
            grads_unit = grads_flat / (grads_flat.norm(dim=1, keepdim=True) + 1e-12)
            alignment = float((grads_unit @ grad_none_unit).mean())

            distortion = feats - feat_none.unsqueeze(0)  # (K,C,H,W)
            mean_stab, median_stab = pairwise_cosine_stats(distortion)

            delta_loss = losses - loss_none
            image_result[kind] = {
                "gradient_alignment": alignment,
                "feature_stability_mean_cos": mean_stab,
                "feature_stability_median_cos": median_stab,
                "loss_sensitivity_mean": float(delta_loss.mean()),
                "loss_sensitivity_std": float(delta_loss.std(unbiased=False)),
            }
        per_image.append(image_result)

        if i % 10 == 0 or i == len(subset):
            print(f"  [b2] image {i}/{len(subset)} ({time.time() - t0:.1f}s elapsed)", flush=True)

    del surrogate
    torch.cuda.empty_cache()

    summary = {
        kind: {
            key: float(sum(r[kind][key] for r in per_image) / len(per_image)) for key in METRIC_KEYS
        }
        for kind in AUG_KINDS
    }

    print("\n[b2] surrogate-side signature (avg over images):")
    print(f"{'kind':<8}{'align':>10}{'feat_stab':>12}{'loss_sens_mean':>16}{'loss_sens_std':>15}")
    for kind in AUG_KINDS:
        s = summary[kind]
        print(
            f"{kind:<8}{s['gradient_alignment']:>10.3f}{s['feature_stability_mean_cos']:>12.3f}"
            f"{s['loss_sensitivity_mean']:>16.3f}{s['loss_sensitivity_std']:>15.3f}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_images": len(subset),
        "manifest": args.manifest,
        "k": args.k,
        "loss_k": args.loss_k,
        "summary": summary,
        "per_image": per_image,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[b2] wrote {out_path}")


if __name__ == "__main__":
    main()
