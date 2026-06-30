"""Gated feed-forward networks: SwiGLU and GeGLU.

Gated linear units (Shazeer 2020, "GLU Variants Improve Transformer") replace
the ReLU MLP with ``proj(act(gate(x)) * up(x))``.  SwiGLU uses SiLU as the gate
activation (Llama/PaLM); GeGLU uses GELU.  Both expand to ``ffn_dim`` and
project back to ``dim``.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class _GatedMLP(nn.Module):
    """Shared gated-MLP body parameterised by its activation."""

    def __init__(self, dim: int, ffn_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _act(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._act(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(h))


class SwiGLU(_GatedMLP):
    """SwiGLU gated MLP (Shazeer 2020): gate activation is SiLU/swish."""

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x)


class GeGLU(_GatedMLP):
    """GeGLU gated MLP (Shazeer 2020): gate activation is GELU."""

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x)


__all__ = ["SwiGLU", "GeGLU"]
