"""Self-attention variants: grouped-query attention and DeepSeek-style MLA.

* :class:`Attention` — standard scaled-dot-product attention with grouped /
  multi-query KV sharing (Ainslie et al. 2023, "GQA: Training Generalized
  Multi-Query Transformer Models") and integrated RoPE. Uses
  ``F.scaled_dot_product_attention`` for the causal kernel.
* :class:`MultiHeadLatentAttention` — DeepSeek-V2 (2024) Multi-head Latent
  Attention: keys/values (and optionally queries) are compressed through a
  low-rank latent then up-projected, with a *decoupled* RoPE sub-dimension
  carried separately so position information survives the compression.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .config import ArchConfig
from .rope import RotaryEmbedding, apply_rotary


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand ``(B, n_kv, T, D)`` to ``(B, n_kv*n_rep, T, D)`` for GQA."""
    if n_rep == 1:
        return x
    b, n_kv, t, d = x.shape
    x = x[:, :, None, :, :].expand(b, n_kv, n_rep, t, d)
    return x.reshape(b, n_kv * n_rep, t, d)


class Attention(nn.Module):
    """Grouped-query causal self-attention with RoPE.

    ``n_kv_heads < n_heads`` gives GQA; ``n_kv_heads == 1`` gives MQA.  KV heads
    are repeated to match the query heads before the attention kernel.  An
    optional ``kv_cache`` of ``(k, v)`` is concatenated for incremental decode.
    """

    def __init__(self, config: ArchConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = config.head_dim
        self.dropout = config.dropout

        self.q_proj = nn.Linear(config.dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.dim, bias=False)

        self.rope = RotaryEmbedding(
            self.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
            scaling=config.rope_scaling,
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

        past_len = kv_cache[0].shape[2] if kv_cache is not None else 0
        cos, sin = self.rope(past_len + t, device=x.device, dtype=x.dtype)
        cos, sin = cos[past_len:], sin[past_len:]
        q, k = apply_rotary(q, k, cos, sin)

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_cache = (k, v) if use_cache else None

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # Causal only when there is no cache (full forward); incremental decode
        # of a single step attends to the whole cached prefix.
        is_causal = attn_mask is None and past_len == 0 and t > 1
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        out = out.transpose(1, 2).contiguous().view(b, t, self.n_heads * self.head_dim)
        out = self.o_proj(out)
        if use_cache:
            return out, new_cache
        return out


class MultiHeadLatentAttention(nn.Module):
    """DeepSeek-V2 Multi-head Latent Attention (2024).

    Keys and values are jointly compressed into a ``kv_lora_rank`` latent
    (``kv_a_proj``) then up-projected per-head (``kv_b_proj``).  The query is
    optionally compressed through ``q_lora_rank``.  Each head additionally
    carries a small *decoupled* RoPE sub-dimension (``qk_rope_head_dim``) whose
    keys are shared across heads, so positional rotation is applied outside the
    compressed-content path.  The per-head attention dimension is therefore
    ``head_dim (nope) + qk_rope_head_dim (rope)``.
    """

    def __init__(self, config: ArchConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim  # the "nope" (content) part per head
        self.rope_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank or max(2, config.dim // 2)
        self.q_lora_rank = config.q_lora_rank  # may be None -> dense query
        self.dropout = config.dropout
        self.scale = (self.head_dim + self.rope_dim) ** -0.5

        # --- query path (optionally low-rank) ---
        q_out = self.n_heads * (self.head_dim + self.rope_dim)
        if self.q_lora_rank is not None:
            self.q_a_proj = nn.Linear(config.dim, self.q_lora_rank, bias=False)
            self.q_a_norm = nn.LayerNorm(self.q_lora_rank)
            self.q_b_proj = nn.Linear(self.q_lora_rank, q_out, bias=False)
        else:
            self.q_proj = nn.Linear(config.dim, q_out, bias=False)

        # --- joint KV compression + decoupled rope key ---
        self.kv_a_proj = nn.Linear(
            config.dim, self.kv_lora_rank + self.rope_dim, bias=False
        )
        self.kv_a_norm = nn.LayerNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank, self.n_heads * (self.head_dim + self.head_dim), bias=False
        )
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.dim, bias=False)

        self.rope = RotaryEmbedding(
            self.rope_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
            scaling=config.rope_scaling,
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        kv_cache=None,
        use_cache: bool = False,
    ):
        b, t, _ = x.shape
        h, nope, rope = self.n_heads, self.head_dim, self.rope_dim

        # ---- query ----
        if self.q_lora_rank is not None:
            q = self.q_b_proj(self.q_a_norm(self.q_a_proj(x)))
        else:
            q = self.q_proj(x)
        q = q.view(b, t, h, nope + rope).transpose(1, 2)  # (B,H,T,nope+rope)
        q_nope, q_rope = q.split([nope, rope], dim=-1)

        # ---- compressed KV ----
        compressed = self.kv_a_proj(x)  # (B,T,kv_rank+rope)
        kv_latent, k_rope = compressed.split([self.kv_lora_rank, rope], dim=-1)
        kv = self.kv_b_proj(self.kv_a_norm(kv_latent))
        kv = kv.view(b, t, h, nope + nope).transpose(1, 2)
        k_nope, v = kv.split([nope, nope], dim=-1)
        # decoupled rope key is shared across heads -> (B,1,T,rope)
        k_rope = k_rope.view(b, t, 1, rope).transpose(1, 2)

        # ---- rope on the decoupled sub-dim ----
        cos, sin = self.rope(t, device=x.device, dtype=x.dtype)
        q_rope, k_rope = apply_rotary(q_rope, k_rope, cos, sin)
        k_rope = k_rope.expand(b, h, t, rope)

        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=attn_mask is None and t > 1,
            scale=self.scale,
        )
        out = out.transpose(1, 2).contiguous().view(b, t, h * nope)
        out = self.o_proj(out)
        if use_cache:
            return out, None
        return out


__all__ = ["Attention", "MultiHeadLatentAttention", "repeat_kv"]
