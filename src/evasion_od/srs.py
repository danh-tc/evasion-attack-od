"""SRS augmentation (Spectral + Resize + Spatial) -- proposed in new-plan.txt
Sec 5.2.E, this project's spectral-augmentation candidate.

Each call: FFT -> random radial-band magnitude attenuation -> inverse FFT ->
bbox-aware adaptive resize -> mild rotation. The resize/rotation steps reuse
`rrb.py`'s existing bbox-centered helpers (same behavior as RRB, just a
smaller rotation range per new-plan.txt's SRS spec) instead of duplicating
that logic.
"""

from __future__ import annotations

import random

import torch

from evasion_od.rrb import adaptive_random_resizing, random_axis_rotation


def spectral_band_attenuation(
    img_chw: torch.Tensor,
    r_lo: float | None = None,
    r_hi: float | None = None,
    factor: float | None = None,
) -> torch.Tensor:
    """Attenuate FFT magnitude within a random normalized radial band [r_lo, r_hi].

    Radius 0 = DC (image center after fftshift), radius 1 = the corner
    farthest from center. Multiplying the complex spectrum by a positive
    real `factor` scales magnitude while preserving phase.
    """
    if r_lo is None or r_hi is None:
        a, b = random.random(), random.random()
        r_lo, r_hi = min(a, b), max(a, b)
    if factor is None:
        factor = random.uniform(0.3, 0.7)

    device = img_chw.device
    _, h, w = img_chw.shape

    spectrum = torch.fft.fftshift(torch.fft.fft2(img_chw), dim=(-2, -1))

    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    radius = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_radius = (cy**2 + cx**2) ** 0.5
    norm_radius = radius / max_radius

    atten = torch.where(
        (norm_radius >= r_lo) & (norm_radius <= r_hi),
        torch.full_like(norm_radius, factor),
        torch.ones_like(norm_radius),
    )
    spectrum = spectrum * atten  # broadcasts (H,W) across the C channel dim

    spectrum = torch.fft.ifftshift(spectrum, dim=(-2, -1))
    return torch.fft.ifft2(spectrum).real


def apply_srs(
    img_chw: torch.Tensor,
    gt_boxes_xyxy: torch.Tensor,
    theta: float = 5.0,
    l_s: int = 10,
    rho: float = 0.8,
    s_max: float = 1.1,
) -> torch.Tensor:
    out = spectral_band_attenuation(img_chw)
    if gt_boxes_xyxy.numel() == 0:
        return out
    out = adaptive_random_resizing(out, gt_boxes_xyxy, rho=rho, s_max=s_max)
    out = random_axis_rotation(out, gt_boxes_xyxy, theta=theta, l_s=l_s)
    return out
