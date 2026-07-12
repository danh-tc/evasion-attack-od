"""RaPA-style DropConnect: random parameter pruning applied via forward hooks.

Adapted from reference-repo/RaPA/core/attacker/DropConnect.py. Unlike the
original (which targets Linear + Normalization layers of a classifier), our
scope is restricted to the *backbone* submodule and, for E4/E5, to
BatchNorm2d affine parameters only -- ResNet-50's backbone has no Linear
layers (see plan.md "Attack Scope"). `layer_types` is kept configurable so a
later step (I2) can extend masking to Conv2d without touching this module.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn

_LAYER_TYPE_MAP = {
    "BatchNorm2d": nn.BatchNorm2d,
    "GroupNorm": nn.GroupNorm,
    "LayerNorm": nn.LayerNorm,
    "Linear": nn.Linear,
    "Conv2d": nn.Conv2d,
}


def _resolve_layer_types(layer_types: tuple[str, ...]) -> tuple[type, ...]:
    return tuple(_LAYER_TYPE_MAP[name] for name in layer_types)


def _pre_hook(module: nn.Module, _input):
    p = module._rapa_drop_prob
    if not hasattr(module, "_rapa_orig_weight"):
        module._rapa_orig_weight = module.weight.data.clone()
    mask_w = torch.bernoulli((1.0 - p) * torch.ones_like(module.weight))
    module.weight.data = module._rapa_orig_weight * mask_w

    if module.bias is not None:
        if not hasattr(module, "_rapa_orig_bias"):
            module._rapa_orig_bias = module.bias.data.clone()
        mask_b = torch.bernoulli((1.0 - p) * torch.ones_like(module.bias))
        module.bias.data = module._rapa_orig_bias * mask_b


def _post_hook(module: nn.Module, _input, _output):
    module.weight.data = module._rapa_orig_weight
    if module.bias is not None:
        module.bias.data = module._rapa_orig_bias


def add_dropconnect_hooks(
    module: nn.Module, drop_prob: float, layer_types: tuple[str, ...] = ("BatchNorm2d",)
) -> list[torch.utils.hooks.RemovableHandle]:
    """Register RaPA DropConnect hooks on every matching submodule of `module`.

    Mask is resampled on every forward call (each call to `module(...)`),
    so running S independent forward passes per attack iteration yields S
    independently-masked variants, matching RaPA's Algorithm 1.
    """
    types = _resolve_layer_types(layer_types)
    handles: list[torch.utils.hooks.RemovableHandle] = []
    n = 0
    for _name, sub in module.named_modules():
        if isinstance(sub, types):
            sub._rapa_drop_prob = drop_prob
            handles.append(sub.register_forward_pre_hook(_pre_hook))
            handles.append(sub.register_forward_hook(_post_hook))
            n += 1
    if n == 0:
        raise ValueError(f"No layers of type {layer_types} found to mask in {module}")
    return handles


def remove_dropconnect_hooks(handles: list[torch.utils.hooks.RemovableHandle]) -> None:
    for h in handles:
        h.remove()


@contextmanager
def dropconnect_scope(module: nn.Module, drop_prob: float, layer_types: tuple[str, ...]):
    """Add hooks for the duration of the `with` block, then remove them."""
    handles = add_dropconnect_hooks(module, drop_prob, layer_types)
    try:
        yield
    finally:
        remove_dropconnect_hooks(handles)
