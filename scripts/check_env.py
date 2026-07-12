#!/usr/bin/env python3
"""Verify the environment setup_env.sh is supposed to produce: package
versions, CUDA, checkpoint/config files, and data manifests."""

from __future__ import annotations

import sys


def main() -> int:
    ok = True

    print("===== Package versions =====")
    try:
        import torch

        print(f"  torch     : {torch.__version__}")
        print(f"  CUDA avail: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU       : {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"  [FAIL] torch: {e}")
        ok = False

    for pkg in ("mmcv", "mmengine", "mmdet"):
        try:
            mod = __import__(pkg)
            print(f"  {pkg:<10}: {mod.__version__}")
        except Exception as e:
            print(f"  [FAIL] {pkg}: {e}")
            ok = False

    print()
    print("===== mmdet registry + CUDA ops =====")
    try:
        from mmdet.utils import register_all_modules

        register_all_modules(init_default_scope=False)
        print("  register_all_modules: OK")
        if torch.cuda.is_available():
            from mmcv.ops import nms

            boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11]], device="cuda", dtype=torch.float32)
            scores = torch.tensor([0.9, 0.8], device="cuda")
            _, keep = nms(boxes, scores, 0.5)
            assert keep.numel() == 1
            print("  MMCV CUDA ops: OK")
    except Exception as e:
        print(f"  [FAIL] {e}")
        ok = False

    print()
    print("===== Model zoo (configs + checkpoints) =====")
    from evasion_od.config import MODEL_ZOO

    for name, spec in MODEL_ZOO.items():
        cfg_ok = spec.config_path.exists()
        ckpt_ok = spec.checkpoint_path.exists()
        status = "OK" if (cfg_ok and ckpt_ok) else "MISSING"
        if status != "OK":
            ok = False
        print(f"  [{status}] {name:<20} group={spec.group:<10} cfg={cfg_ok} ckpt={ckpt_ok}")

    print()
    print("===== Data =====")
    from evasion_od.config import COCO_ANNOTATIONS, COCO_IMAGES_DIR, MANIFEST_DIR

    print(f"  val2017 images dir exists : {COCO_IMAGES_DIR.exists()}")
    print(f"  annotations file exists   : {COCO_ANNOTATIONS.exists()}")
    if not (COCO_IMAGES_DIR.exists() and COCO_ANNOTATIONS.exists()):
        ok = False

    for manifest in ("dev_300", "val_100"):
        path = MANIFEST_DIR / f"{manifest}.json"
        if path.exists():
            import json

            with open(path) as f:
                n = len(json.load(f)["image_ids"])
            print(f"  [OK] {manifest}.json ({n} images)")
        else:
            print(f"  [MISSING] {manifest}.json")
            ok = False

    print()
    print("===== RESULT =====")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
