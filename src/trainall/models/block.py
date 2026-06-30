"""A single pre-norm decoder block.

Pre-norm transformer block (used by GPT-NeoX / Llama): RMSNorm precedes each
sub-layer and a residual wraps it::

    h = x + Attn(RMSNorm(x))
    y = h + FFN(RMSNorm(h))

``FFN`` is a :class:`MoEFeedForward` when ``config.use_moe`` else a
:class:`SwiGLU`.  Attention is :class:`MultiHeadLatentAttention` when an MLA
rank is configured else grouped-query :class:`Attention`.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from .attention import Attention, MultiHeadLatentAttention
from .config import ArchConfig
from .mlp import SwiGLU
from .moe import MoEFeedForward
from .norm import RMSNorm


class DecoderBlock(nn.Module):
    """Pre-RMSNorm attention + feed-forward block with residuals.

    When the feed-forward is a MoE layer its load-balancing auxiliary loss is
    surfaced as the second element of the forward tuple; otherwise that element
    is ``None``.
    """

    def __init__(self, config: ArchConfig) -> None:
        super().__init__()
        self.use_moe = config.use_moe
        use_mla = config.kv_lora_rank is not None or config.q_lora_rank is not None

        self.attn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.attn = (
            MultiHeadLatentAttention(config) if use_mla else Attention(config)
        )
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn = (
            MoEFeedForward(config)
            if config.use_moe
            else SwiGLU(config.dim, config.ffn_dim, dropout=config.dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        kv_cache=None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], object]:
        attn_out = self.attn(
            self.attn_norm(x), attn_mask=attn_mask, kv_cache=kv_cache, use_cache=use_cache
        )
        new_cache = None
        if use_cache:
            attn_out, new_cache = attn_out
        x = x + attn_out

        ffn_in = self.ffn_norm(x)
        aux_loss = None
        if self.use_moe:
            ffn_out, aux_loss = self.ffn(ffn_in)
        else:
            ffn_out = self.ffn(ffn_in)
        x = x + ffn_out
        return x, aux_loss, new_cache


__all__ = ["DecoderBlock"]
