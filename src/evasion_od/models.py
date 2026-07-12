"""mmdet model loading."""

from __future__ import annotations

import torch

from evasion_od.config import MODEL_ZOO, ModelSpec

_modules_registered = False


def _ensure_registered() -> None:
    global _modules_registered
    if not _modules_registered:
        from mmdet.utils import register_all_modules

        # init_default_scope=True so mmcv.transforms.Compose (used in
        # preprocessing.py) can resolve mmdet-registered transform names
        # (e.g. "LoadImageFromNDArray") without an explicit scope argument.
        register_all_modules(init_default_scope=True)
        _modules_registered = True


def load_model(name: str, device: str = "cuda:0"):
    """Load a detector from the model zoo in eval mode."""
    _ensure_registered()
    from mmdet.apis import init_detector

    spec: ModelSpec = MODEL_ZOO[name]
    if not spec.config_path.exists():
        raise FileNotFoundError(
            f"Config not found for '{name}': {spec.config_path}. "
            "Run setup_env.sh (checkpoint download step) first."
        )
    if not spec.checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found for '{name}': {spec.checkpoint_path}."
        )
    model = init_detector(str(spec.config_path), str(spec.checkpoint_path), device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def backbone_features(model, batch_inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Raw backbone (pre-neck) multi-stage feature maps for a normalized+padded batch."""
    feats = model.backbone(batch_inputs)
    if not isinstance(feats, (tuple, list)):
        feats = (feats,)
    return tuple(feats)
