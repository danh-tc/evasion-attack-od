"""Thin wrapper around mmdet's own inference pipeline for evaluation.

Each target model runs its own native preprocessing (resize scale,
normalization, pad_size_divisor) via `inference_detector`, so a single
full-resolution adversarial image transfers correctly regardless of which
target model consumes it next -- exactly OSFD's eval-time convention.
"""

from __future__ import annotations

import numpy as np

from evasion_od.metrics import ImageDetections, pred_instances_to_arrays


def detect(model, img_bgr_uint8: np.ndarray, image_id: int, label_to_cat_id: list[int]) -> ImageDetections:
    from mmdet.apis import inference_detector

    result = inference_detector(model, img_bgr_uint8)
    bboxes, cat_ids, scores = pred_instances_to_arrays(result.pred_instances, label_to_cat_id)
    return ImageDetections(image_id=image_id, bboxes=bboxes, cat_ids=cat_ids, scores=scores)
