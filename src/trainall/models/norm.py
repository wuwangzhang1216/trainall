"""Root-mean-square layer normalisation.

RMSNorm (Zhang & Sennrich 2019, "Root Mean Square Layer Normalization") drops
the mean-centring of LayerNorm and rescales activations by their RMS, then
applies a learned per-channel gain.  It is the de-facto pre-norm of modern
decoder-only models.
"""
from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """RMSNorm (Zhang & Sennrich 2019).

    ``y = x / sqrt(mean(x^2) + eps) * weight``.  Normalisation is computed in
    fp32 for numerical stability and cast back to the input dtype.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._norm(x.float()).type_as(x)
        return out * self.weight


__all__ = ["RMSNorm"]
