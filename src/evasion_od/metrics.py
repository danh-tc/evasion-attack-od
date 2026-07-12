"""ASR / Object Disappearance Rate + mAP-drop evaluation.

ASR (per plan.md): for each image, how many of the objects the *clean*
detector got right does the *adversarial* detector fail to get right?
Aggregated as a ratio-of-sums across the eval set (more stable than
averaging per-image ratios when some images have zero clean-matched GT).
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from dataclasses import dataclass

import numpy as np

IOU_MATCH_THR = 0.5
SCORE_THR = 0.3


@dataclass
class EvalResult:
    n_images: int
    n_gt: int
    clean_matched: int
    adv_matched: int
    asr: float | None
    map_clean: float
    map_adv: float
    map_drop: float


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - inter
    return inter / np.clip(union, 1e-9, None)


def count_matched_gt(
    gt_boxes: np.ndarray,
    gt_cat_ids: np.ndarray,
    pred_boxes: np.ndarray,
    pred_cat_ids: np.ndarray,
    pred_scores: np.ndarray,
    iou_thr: float = IOU_MATCH_THR,
    score_thr: float = SCORE_THR,
) -> int:
    """Greedy GT->pred matching (same category, IoU>=thr, score>=thr). Returns #GT matched."""
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return 0
    keep = pred_scores >= score_thr
    pred_boxes, pred_cat_ids, pred_scores = pred_boxes[keep], pred_cat_ids[keep], pred_scores[keep]
    if len(pred_boxes) == 0:
        return 0

    order = np.argsort(-pred_scores)
    pred_boxes, pred_cat_ids = pred_boxes[order], pred_cat_ids[order]

    used = np.zeros(len(pred_boxes), dtype=bool)
    matched = 0
    for box, cat in zip(gt_boxes, gt_cat_ids):
        same_cat = (pred_cat_ids == cat) & (~used)
        if not same_cat.any():
            continue
        ious = _iou(box, pred_boxes)
        ious = np.where(same_cat, ious, 0.0)
        best = int(np.argmax(ious))
        if ious[best] >= iou_thr:
            used[best] = True
            matched += 1
    return matched


def pred_instances_to_arrays(pred_instances, label_to_cat_id: list[int]):
    bboxes = pred_instances.bboxes.detach().cpu().numpy()
    scores = pred_instances.scores.detach().cpu().numpy()
    labels = pred_instances.labels.detach().cpu().numpy()
    cat_ids = np.array([label_to_cat_id[int(l)] for l in labels])
    return bboxes, cat_ids, scores


def to_coco_result_dicts(image_id: int, bboxes, cat_ids, scores) -> list[dict]:
    out = []
    for (x1, y1, x2, y2), cat_id, score in zip(bboxes, cat_ids, scores):
        out.append(
            {
                "image_id": image_id,
                "category_id": int(cat_id),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(score),
            }
        )
    return out


def coco_map(coco, image_ids: list[int], result_dicts: list[dict]) -> float:
    from pycocotools.cocoeval import COCOeval

    if not result_dicts:
        return 0.0
    with redirect_stdout(io.StringIO()):
        coco_dt = coco.loadRes(result_dicts)
        ev = COCOeval(coco, coco_dt, iouType="bbox")
        ev.params.imgIds = image_ids
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return float(ev.stats[0])  # mAP @ IoU=.50:.95


@dataclass
class ImageDetections:
    image_id: int
    bboxes: np.ndarray
    cat_ids: np.ndarray
    scores: np.ndarray


def evaluate(
    coco,
    image_ids: list[int],
    gt_by_image: dict[int, tuple[np.ndarray, np.ndarray]],  # image_id -> (boxes_xyxy, cat_ids)
    clean_dets: dict[int, ImageDetections],
    adv_dets: dict[int, ImageDetections],
) -> EvalResult:
    clean_matched_total = 0
    adv_matched_total = 0
    n_gt_total = 0
    clean_results: list[dict] = []
    adv_results: list[dict] = []

    for image_id in image_ids:
        gt_boxes, gt_cat_ids = gt_by_image[image_id]
        n_gt_total += len(gt_boxes)

        cd, ad = clean_dets[image_id], adv_dets[image_id]
        clean_matched_total += count_matched_gt(gt_boxes, gt_cat_ids, cd.bboxes, cd.cat_ids, cd.scores)
        adv_matched_total += count_matched_gt(gt_boxes, gt_cat_ids, ad.bboxes, ad.cat_ids, ad.scores)

        clean_results += to_coco_result_dicts(image_id, cd.bboxes, cd.cat_ids, cd.scores)
        adv_results += to_coco_result_dicts(image_id, ad.bboxes, ad.cat_ids, ad.scores)

    asr = None
    if clean_matched_total > 0:
        asr = (clean_matched_total - adv_matched_total) / clean_matched_total

    map_clean = coco_map(coco, image_ids, clean_results)
    map_adv = coco_map(coco, image_ids, adv_results)

    return EvalResult(
        n_images=len(image_ids),
        n_gt=n_gt_total,
        clean_matched=clean_matched_total,
        adv_matched=adv_matched_total,
        asr=asr,
        map_clean=map_clean,
        map_adv=map_adv,
        map_drop=map_clean - map_adv,
    )
