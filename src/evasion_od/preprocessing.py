"""mmdet-3.x-native preprocessing glue.

Rather than re-implementing keep-ratio resize / pad / normalize by hand, we
reuse each model's own test pipeline (for resize + metadata) and its own
`data_preprocessor` (for normalize + pad). This keeps per-model quirks
(input scale, pad_size_divisor, mean/std, bgr_to_rgb) automatically correct
without duplicating mmdet internals, and keeps the whole path differentiable
since neither step is wrapped in `torch.no_grad()` here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ResizedInput:
    """Everything needed to run `model` on a perturbable image."""

    clean_chw: torch.Tensor  # (C,H,W) float32, raw pixel scale [0,255], resized, NO grad
    data_sample: object  # mmdet DetDataSample with img_shape/ori_shape/scale_factor set
    ori_shape: tuple[int, int]  # (H, W) of the *original* (pre-resize) image


def build_resized_input(model, img_bgr_uint8: np.ndarray, image_id: int, device) -> ResizedInput:
    from mmcv.transforms import Compose
    from mmdet.utils import get_test_pipeline_cfg

    pipeline_cfg = get_test_pipeline_cfg(model.cfg.copy())
    pipeline_cfg[0]["type"] = "LoadImageFromNDArray"
    test_pipeline = Compose(pipeline_cfg)

    data = test_pipeline(dict(img=img_bgr_uint8, img_id=image_id))
    clean_chw = data["inputs"].to(device=device, dtype=torch.float32)
    return ResizedInput(
        clean_chw=clean_chw,
        data_sample=data["data_samples"],
        ori_shape=img_bgr_uint8.shape[:2],
    )


def preprocess_batch(model, chw_uint8_range: torch.Tensor, data_sample) -> dict:
    """Normalize + pad a single (C,H,W) image via the model's own data_preprocessor.

    `chw_uint8_range` may require grad -- this function does not detach and
    is not wrapped in no_grad, so gradients flow back to it.
    """
    raw = dict(inputs=[chw_uint8_range], data_samples=[data_sample])
    return model.data_preprocessor(raw, False)


def scale_gt_boxes(gt_boxes_xyxy: np.ndarray, data_sample, device) -> torch.Tensor:
    """GT boxes (original-image pixel coords) -> resized-image pixel coords."""
    if len(gt_boxes_xyxy) == 0:
        return torch.zeros((0, 4), device=device)
    sx, sy = data_sample.metainfo["scale_factor"]
    scale = torch.tensor([sx, sy, sx, sy], device=device, dtype=torch.float32)
    return torch.as_tensor(gt_boxes_xyxy, device=device, dtype=torch.float32) * scale


def upscale_delta_to_original(delta_chw: torch.Tensor, ori_shape: tuple[int, int]) -> torch.Tensor:
    """Resize an attack-resolution perturbation back up to the original image size."""
    import torch.nn.functional as F

    h, w = ori_shape
    return F.interpolate(
        delta_chw.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
    ).squeeze(0)


def build_adversarial_image(
    clean_bgr_uint8_full: np.ndarray, delta_chw: torch.Tensor, ori_shape: tuple[int, int]
) -> np.ndarray:
    """Reconstruct a full-resolution adversarial image (numpy, HWC BGR uint8).

    Mirrors OSFD's eval-time noise handling (resize noise -> add to full-res
    clean image -> clamp -> round), so the same delta can be transferred to
    any target model regardless of that model's own input resolution.
    """
    delta_full = upscale_delta_to_original(delta_chw, ori_shape)  # (C,H,W)
    delta_full_np = delta_full.detach().cpu().numpy().transpose(1, 2, 0)  # HWC
    adv = clean_bgr_uint8_full.astype(np.float32) + delta_full_np
    adv = np.clip(np.round(adv), 0, 255).astype(np.uint8)
    return adv
