"""Glue: manifest -> attack on surrogate -> evaluate on N target models."""

from __future__ import annotations

import time

import torch

from evasion_od.attack import run_attack
from evasion_od.config import AttackConfig
from evasion_od.data import CocoSubset, load_image_bgr, load_manifest
from evasion_od.gradient_diagnostics import CheckpointStats
from evasion_od.inference import detect
from evasion_od.metrics import EvalResult, evaluate, per_image_asr
from evasion_od.models import load_model


def load_subset(n_images: int, manifest: str = "dev_300") -> CocoSubset:
    image_ids = load_manifest(manifest)[:n_images]
    return CocoSubset(image_ids)


def generate_adversarial(surrogate_model, subset: CocoSubset, cfg: AttackConfig, device: str):
    adv_images: dict[int, "np.ndarray"] = {}
    gt_by_image: dict[int, tuple] = {}
    n = len(subset)
    for i, sample in enumerate(subset, start=1):
        t0 = time.time()
        img = load_image_bgr(sample.image_path)
        adv_images[sample.image_id] = run_attack(
            surrogate_model, img, sample.image_id, sample.gt_boxes, cfg, device
        )
        gt_by_image[sample.image_id] = (sample.gt_boxes, sample.gt_labels)
        print(
            f"    [attack] image {i}/{n} id={sample.image_id} {time.time() - t0:.2f}s",
            flush=True,
        )
    return adv_images, gt_by_image


def generate_adversarial_with_diagnostics(
    surrogate_model,
    subset: CocoSubset,
    cfg: AttackConfig,
    device: str,
    diagnostic_iters: frozenset[int],
) -> tuple[dict, dict, dict[int, list[CheckpointStats]]]:
    """Same as `generate_adversarial`, plus per-image gradient-agreement
    checkpoints (gradient_diagnostics.py) captured on the exact trajectory
    each image's adversarial delta actually takes -- Phase G0 only.
    """
    adv_images: dict[int, "np.ndarray"] = {}
    gt_by_image: dict[int, tuple] = {}
    diagnostics: dict[int, list[CheckpointStats]] = {}
    n = len(subset)
    for i, sample in enumerate(subset, start=1):
        t0 = time.time()
        img = load_image_bgr(sample.image_path)
        diag_out: list[CheckpointStats] = []
        adv_images[sample.image_id] = run_attack(
            surrogate_model,
            img,
            sample.image_id,
            sample.gt_boxes,
            cfg,
            device,
            diagnostic_iters=diagnostic_iters,
            diagnostic_out=diag_out,
        )
        gt_by_image[sample.image_id] = (sample.gt_boxes, sample.gt_labels)
        diagnostics[sample.image_id] = diag_out
        print(
            f"    [attack] image {i}/{n} id={sample.image_id} {time.time() - t0:.2f}s",
            flush=True,
        )
    return adv_images, gt_by_image, diagnostics


def _detect_on_model(
    model_name: str,
    device: str,
    subset: CocoSubset,
    adv_images: dict,
    label_to_cat_id: list[int],
) -> tuple[dict, dict]:
    model = load_model(model_name, device)
    clean_dets, adv_dets = {}, {}
    n = len(subset)
    for i, sample in enumerate(subset, start=1):
        t0 = time.time()
        clean_img = load_image_bgr(sample.image_path)
        clean_dets[sample.image_id] = detect(model, clean_img, sample.image_id, label_to_cat_id)
        adv_dets[sample.image_id] = detect(
            model, adv_images[sample.image_id], sample.image_id, label_to_cat_id
        )
        print(
            f"    [eval:{model_name}] image {i}/{n} id={sample.image_id} {time.time() - t0:.2f}s",
            flush=True,
        )
    del model
    torch.cuda.empty_cache()
    return clean_dets, adv_dets


def evaluate_on_model(
    model_name: str,
    device: str,
    subset: CocoSubset,
    adv_images: dict,
    gt_by_image: dict,
    label_to_cat_id: list[int],
) -> EvalResult:
    clean_dets, adv_dets = _detect_on_model(model_name, device, subset, adv_images, label_to_cat_id)
    image_ids = list(adv_images.keys())
    return evaluate(subset.coco, image_ids, gt_by_image, clean_dets, adv_dets)


def evaluate_on_model_per_image(
    model_name: str,
    device: str,
    subset: CocoSubset,
    adv_images: dict,
    gt_by_image: dict,
    label_to_cat_id: list[int],
) -> tuple[EvalResult, dict[int, float | None]]:
    """Same as `evaluate_on_model`, plus per-image ASR -- Phase G0 (plan.md)
    needs image-level granularity to correlate transfer with per-image
    gradient agreement, which the aggregate EvalResult alone can't give.
    """
    clean_dets, adv_dets = _detect_on_model(model_name, device, subset, adv_images, label_to_cat_id)
    image_ids = list(adv_images.keys())
    agg = evaluate(subset.coco, image_ids, gt_by_image, clean_dets, adv_dets)
    per_image = per_image_asr(image_ids, gt_by_image, clean_dets, adv_dets)
    return agg, per_image
