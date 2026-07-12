"""Glue: manifest -> attack on surrogate -> evaluate on N target models."""

from __future__ import annotations

import torch

from evasion_od.attack import run_attack
from evasion_od.config import AttackConfig
from evasion_od.data import CocoSubset, load_image_bgr, load_manifest
from evasion_od.inference import detect
from evasion_od.metrics import EvalResult, evaluate
from evasion_od.models import load_model


def load_subset(n_images: int, manifest: str = "dev_300") -> CocoSubset:
    image_ids = load_manifest(manifest)[:n_images]
    return CocoSubset(image_ids)


def generate_adversarial(surrogate_model, subset: CocoSubset, cfg: AttackConfig, device: str):
    adv_images: dict[int, "np.ndarray"] = {}
    gt_by_image: dict[int, tuple] = {}
    for sample in subset:
        img = load_image_bgr(sample.image_path)
        adv_images[sample.image_id] = run_attack(
            surrogate_model, img, sample.image_id, sample.gt_boxes, cfg, device
        )
        gt_by_image[sample.image_id] = (sample.gt_boxes, sample.gt_labels)
    return adv_images, gt_by_image


def evaluate_on_model(
    model_name: str,
    device: str,
    subset: CocoSubset,
    adv_images: dict,
    gt_by_image: dict,
    label_to_cat_id: list[int],
) -> EvalResult:
    model = load_model(model_name, device)
    clean_dets, adv_dets = {}, {}
    for sample in subset:
        clean_img = load_image_bgr(sample.image_path)
        clean_dets[sample.image_id] = detect(model, clean_img, sample.image_id, label_to_cat_id)
        adv_dets[sample.image_id] = detect(
            model, adv_images[sample.image_id], sample.image_id, label_to_cat_id
        )
    del model
    torch.cuda.empty_cache()

    image_ids = list(adv_images.keys())
    return evaluate(subset.coco, image_ids, gt_by_image, clean_dets, adv_dets)
