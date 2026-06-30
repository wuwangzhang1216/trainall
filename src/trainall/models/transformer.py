"""The assembled decoder-only language model.

:class:`DecoderLM` stacks an embedding, ``n_layers`` :class:`DecoderBlock`s, a
final RMSNorm and an LM head (optionally weight-tied to the embedding, Press &
Wolf 2017).  :meth:`forward` returns an :class:`LMOutput` carrying ``logits``
and the summed MoE ``aux_loss`` (a zero scalar when MoE is off), matching the
HF ``.logits`` convention used by the objectives layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from .block import DecoderBlock
from .config import ArchConfig
from .norm import RMSNorm


@dataclass(frozen=True)
class LMOutput:
    """Forward output: next-token ``logits`` and MoE ``aux_loss`` scalar."""

    logits: torch.Tensor
    aux_loss: torch.Tensor


class DecoderLM(nn.Module):
    """Decoder-only transformer LM built from an :class:`ArchConfig`."""

    def __init__(self, config: ArchConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.dim)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(DecoderBlock(config) for _ in range(config.n_layers))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @classmethod
    def from_config(cls, config: ArchConfig) -> "DecoderLM":
        """Build a model from its architecture config."""
        return cls(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> LMOutput:
        x = self.dropout(self.embed_tokens(input_ids))

        attn_mask = None
        if attention_mask is not None:
            # (B, T) padding mask -> additive (B, 1, 1, T) bias; combine with the
            # causal structure so SDPA gets an explicit mask.
            b, t = attention_mask.shape
            pad = attention_mask[:, None, None, :].to(torch.bool)
            causal = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
            attn_mask = pad & causal[None, None, :, :]

        aux_total = x.new_zeros(())
        for layer in self.layers:
            x, aux_loss, _ = layer(x, attn_mask=attn_mask)
            if aux_loss is not None:
                aux_total = aux_total + aux_loss

        x = self.norm(x)
        logits = self.lm_head(x)
        return LMOutput(logits=logits, aux_loss=aux_total)


__all__ = ["DecoderLM", "LMOutput"]
