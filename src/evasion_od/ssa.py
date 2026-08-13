"""SSA augmentation (Spectrum Simulation Attack, Long et al. ECCV 2022).

Classification-transfer SOTA adapted for detection, per new-plan.txt Sec
5.2.D's simplified FFT description: FFT the image, multiply by a random
per-pixel spectral scale in [1-rho, 1+rho], inverse FFT. Averaging the
gradient over N such spectral copies (new-plan.txt's "N=20") is handled by
the caller reusing `AttackConfig.num_masks` -- this module only produces one
spectral copy per call, same contract as rrb.py's `apply_rrb`.
"""

from __future__ import annotations

import torch


def apply_ssa(img_chw: torch.Tensor, rho: float = 0.5) -> torch.Tensor:
    spectrum = torch.fft.fft2(img_chw)
    scale = 1.0 + rho * (2.0 * torch.rand_like(img_chw) - 1.0)
    return torch.fft.ifft2(spectrum * scale).real
