"""Sparse mixture-of-experts feed-forward with top-k token routing.

Top-k gating (Shazeer 2017, "Outrageously Large Neural Networks") routes each
token to its ``k`` highest-scoring experts (each a SwiGLU MLP), combines their
outputs by the renormalised softmax gate weights, and adds a load-balancing
auxiliary loss (Mixtral / Switch-Transformer style) that pushes the router
toward uniform expert utilisation.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .config import ArchConfig
from .mlp import SwiGLU


class MoEFeedForward(nn.Module):
    """Top-k routed mixture of SwiGLU experts with a load-balancing aux loss.

    Returns ``(output, aux_loss)``.  ``aux_loss`` is the product of the mean
    fraction of tokens dispatched to each expert and the mean router
    probability mass on each expert, scaled by ``n_experts`` and the config
    coefficient (Switch/Mixtral importance-load loss).  The most recent router
    softmax probabilities are stashed on :attr:`router_probs` for tests.
    """

    def __init__(self, config: ArchConfig) -> None:
        super().__init__()
        self.n_experts = config.n_experts
        self.top_k = config.n_experts_per_tok
        self.aux_coef = config.moe_aux_loss_coef
        self.gate = nn.Linear(config.dim, self.n_experts, bias=False)
        self.experts = nn.ModuleList(
            SwiGLU(config.dim, config.ffn_dim, dropout=config.dropout)
            for _ in range(self.n_experts)
        )
        #: Exposed for tests: last forward's router probabilities ``(N, E)``.
        self.router_probs: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, t, d = x.shape
        x_flat = x.reshape(-1, d)  # (N, d)
        n = x_flat.shape[0]

        logits = self.gate(x_flat)  # (N, E)
        probs = F.softmax(logits, dim=-1)
        self.router_probs = probs

        topk_w, topk_idx = probs.topk(self.top_k, dim=-1)  # (N, k)
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True).clamp(min=1e-9)

        out = torch.zeros_like(x_flat)
        for e in range(self.n_experts):
            # token, slot positions routed to expert e
            sel = topk_idx == e
            if not sel.any():
                continue
            token_idx, slot_idx = sel.nonzero(as_tuple=True)
            weights = topk_w[token_idx, slot_idx].unsqueeze(-1)
            expert_out = self.experts[e](x_flat[token_idx])
            out.index_add_(0, token_idx, expert_out * weights)

        # --- load-balancing aux loss (Switch/Mixtral importance x load) ---
        # fraction of tokens dispatched to each expert
        one_hot = F.one_hot(topk_idx, self.n_experts).float().sum(dim=1)  # (N,E)
        load = one_hot.mean(dim=0)  # P(token -> expert e)
        importance = probs.mean(dim=0)  # mean router mass on expert e
        aux_loss = self.aux_coef * self.n_experts * (load * importance).sum()

        return out.view(b, t, d), aux_loss


__all__ = ["MoEFeedForward"]
