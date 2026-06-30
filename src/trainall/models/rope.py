"""Rotary position embeddings with linear / NTK / YaRN scaling.

RoPE (Su et al. 2021, "RoFormer: Enhanced Transformer with Rotary Position
Embedding") encodes absolute position by rotating pairs of query/key channels;
the dot product then depends only on relative offset.  Long-context scaling
strategies adjust the inverse frequencies:

* ``linear`` (Chen et al. 2023) — divide positions by ``factor`` (position
  interpolation).
* ``ntk``    (bloc97 2023)      — scale the base ``theta`` so high frequencies
  stretch less than low ones.
* ``yarn``   (Peng et al. 2023) — per-frequency interpolation ramp.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Map ``[x1, x2]`` (split on last dim) to ``[-x2, x1]``."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply the rotary transform to query/key tensors.

    ``q``/``k``: ``(B, n_heads, T, D)`` where ``D`` is the rotary dimension.
    ``cos``/``sin``: ``(T, D)`` (broadcast over batch & heads).
    Returns the rotated ``(q, k)``.
    """
    # (T, D) -> (1, 1, T, D) for broadcasting over batch and heads.
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot.type_as(q), k_rot.type_as(k)


class RotaryEmbedding(nn.Module):
    """Precompute and serve ``cos``/``sin`` rotary tables.

    Supports the three common long-context scaling schemes selected by a
    ``rope_scaling`` dict ``{"type": ..., "factor": ...}``.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
        scaling: Optional[dict] = None,
    ) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"rotary dim must be even, got {dim}")
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.scaling = scaling or None

        inv_freq, attn_factor = self._compute_inv_freq()
        self.attn_factor = attn_factor
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        cos, sin = self._build_tables(max_seq_len)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def _compute_inv_freq(self) -> Tuple[torch.Tensor, float]:
        half = self.dim // 2
        idx = torch.arange(0, half, dtype=torch.float32)
        scaling = self.scaling
        theta = self.theta
        attn_factor = 1.0
        if scaling and scaling.get("type") == "ntk":
            # NTK-aware: rescale the base so the rotation budget is spread.
            factor = float(scaling.get("factor", 1.0))
            theta = theta * (factor ** (self.dim / (self.dim - 2)))
        inv_freq = 1.0 / (theta ** (2.0 * idx / self.dim))
        if scaling and scaling.get("type") == "linear":
            inv_freq = inv_freq / float(scaling.get("factor", 1.0))
        elif scaling and scaling.get("type") == "yarn":
            factor = float(scaling.get("factor", 1.0))
            # Per-frequency interpolation ramp between low/high cutoffs.
            low = scaling.get("beta_fast", 32)
            high = scaling.get("beta_slow", 1)
            ramp = self._yarn_ramp(low, high, half)
            inv_freq = inv_freq / factor * (1 - ramp) + inv_freq * ramp
            # mscale temperature attenuates logits for extrapolation.
            attn_factor = 0.1 * math.log(factor) + 1.0
        return inv_freq, attn_factor

    @staticmethod
    def _yarn_ramp(low: float, high: float, n: int) -> torch.Tensor:
        idx = torch.arange(n, dtype=torch.float32)
        if low == high:
            high += 1e-3
        ramp = (idx - low) / (high - low)
        return ramp.clamp(0.0, 1.0)

    def _build_tables(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)  # (T, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (T, dim)
        cos = emb.cos() * self.attn_factor
        sin = emb.sin() * self.attn_factor
        return cos, sin

    def forward(
        self, seq_len: int, device=None, dtype=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(cos, sin)`` tables of shape ``(seq_len, dim)``."""
        if seq_len > self.cos_cached.shape[0]:
            cos, sin = self._build_tables(seq_len)
            self.cos_cached = cos.to(self.cos_cached.device)
            self.sin_cached = sin.to(self.sin_cached.device)
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        if device is not None:
            cos, sin = cos.to(device), sin.to(device)
        if dtype is not None:
            cos, sin = cos.to(dtype), sin.to(dtype)
        return cos, sin


__all__ = ["RotaryEmbedding", "apply_rotary"]
