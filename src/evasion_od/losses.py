"""Backbone-feature distortion loss (OSFD), with NRDM as the k=1 special case.

    L = sum_i (1/N_i) * sum_j ( F_adv[i][j] - k * F_clean[i][j] )^2

Maximizing this loss suppresses the significant (object) features and, for
k>1, amplifies the vicinal (boundary/background) features -- see OSFD paper
Eq. 2 and Fig. 3. k=1 reduces to NRDM (plain feature-distance maximization,
no suppress/amplify asymmetry).

Attack loops should compute gradients as `grad(loss, delta)` and step in the
`+sign(grad)` direction (gradient *ascent* on this loss).
"""

from __future__ import annotations

import torch


def backbone_feature_loss(
    feats_adv: tuple[torch.Tensor, ...], feats_clean: tuple[torch.Tensor, ...], k: float
) -> torch.Tensor:
    if len(feats_adv) != len(feats_clean):
        raise ValueError(
            f"stage count mismatch: adv={len(feats_adv)} clean={len(feats_clean)}"
        )
    loss = feats_adv[0].new_zeros(())
    for f_adv, f_clean in zip(feats_adv, feats_clean):
        loss = loss + torch.mean((f_adv - k * f_clean) ** 2)
    return loss
