"""COCO val2017 manifest-based image + ground-truth loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from evasion_od.config import COCO_ANNOTATIONS, COCO_IMAGES_DIR, MANIFEST_DIR


def load_manifest(name: str) -> list[int]:
    """name is e.g. 'dev_300' or 'val_100' (see setup_env.sh step 15)."""
    path = MANIFEST_DIR / f"{name}.json"
    with open(path) as f:
        manifest = json.load(f)
    return manifest["image_ids"]


@dataclass
class Sample:
    image_id: int
    image_path: Path
    gt_boxes: np.ndarray  # (N, 4) xyxy, pixel coords in the *original* image
    gt_labels: np.ndarray  # (N,) COCO category ids (not contiguous 0..79)


class CocoSubset:
    """Thin wrapper around pycocotools.COCO restricted to a manifest's image ids."""

    def __init__(self, image_ids: list[int]):
        from pycocotools.coco import COCO

        self.coco = COCO(str(COCO_ANNOTATIONS))
        self.image_ids = image_ids

    def __len__(self) -> int:
        return len(self.image_ids)

    def __iter__(self):
        for image_id in self.image_ids:
            yield self.get(image_id)

    def get(self, image_id: int) -> Sample:
        img_info = self.coco.loadImgs([image_id])[0]
        ann_ids = self.coco.getAnnIds(imgIds=[image_id], iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)

        boxes = np.zeros((len(anns), 4), dtype=np.float32)
        labels = np.zeros((len(anns),), dtype=np.int64)
        for i, ann in enumerate(anns):
            x, y, w, h = ann["bbox"]
            boxes[i] = [x, y, x + w, y + h]
            labels[i] = ann["category_id"]

        return Sample(
            image_id=image_id,
            image_path=COCO_IMAGES_DIR / img_info["file_name"],
            gt_boxes=boxes,
            gt_labels=labels,
        )

    def label_to_cat_id(self) -> list[int]:
        """contiguous mmdet label index (0..79) -> COCO category id.

        mmdet's CocoDataset.METAINFO['classes'] is the canonical name list in
        the exact order used for contiguous label indices.
        """
        from mmdet.datasets import CocoDataset

        name_to_id = {c["name"]: c["id"] for c in self.coco.loadCats(self.coco.getCatIds())}
        return [name_to_id[name] for name in CocoDataset.METAINFO["classes"]]


def load_image_bgr(path: Path) -> np.ndarray:
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img  # HWC, BGR, uint8
