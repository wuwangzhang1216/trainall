"""Architecture hyperparameters for the from-scratch decoder stack.

:class:`ArchConfig` is the *shape* spec for :class:`trainall.models.DecoderLM`
(distinct from the run-level ``ModelConfig`` in :mod:`trainall.config`).  The
defaults are deliberately tiny so unit tests instantiate and run a forward +
backward pass on CPU in milliseconds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class ArchConfig:
    """Decoder-only transformer hyperparameters.

    Mirrors the knobs of a modern Llama/DeepSeek-style stack: RMSNorm,
    RoPE (optionally scaled), GQA/MQA via ``n_kv_heads``, SwiGLU or MoE
    feed-forward, and optional MLA low-rank attention via ``q_lora_rank`` /
    ``kv_lora_rank``.
    """

    vocab_size: int = 64
    dim: int = 32
    n_layers: int = 2
    n_heads: int = 4
    n_kv_heads: int = 2
    #: Per-head dimension; defaults to ``dim // n_heads`` when left ``None``.
    head_dim: Optional[int] = None
    ffn_dim: int = 64
    rope_theta: float = 10000.0
    #: e.g. ``{"type": "linear"|"ntk"|"yarn", "factor": 2.0}`` or ``None``.
    rope_scaling: Optional[Dict] = None
    norm_eps: float = 1e-5
    max_seq_len: int = 64
    dropout: float = 0.0
    tie_embeddings: bool = True
    # --- Mixture-of-Experts ---
    use_moe: bool = False
    n_experts: int = 8
    n_experts_per_tok: int = 2
    moe_aux_loss_coef: float = 0.01
    # --- attention backend / MLA ---
    attn_impl: str = "sdpa"
    #: Multi-head Latent Attention rank for the query path (DeepSeek-V2). When
    #: ``None`` the query is not low-rank compressed.
    q_lora_rank: Optional[int] = None
    #: Latent rank for the joint key/value compression (DeepSeek-V2). Required
    #: by :class:`MultiHeadLatentAttention`.
    kv_lora_rank: Optional[int] = None
    #: Decoupled RoPE sub-dimension per head for MLA. Defaults to ``head_dim//2``.
    qk_rope_head_dim: Optional[int] = None

    def __post_init__(self) -> None:
        if self.head_dim is None:
            if self.dim % self.n_heads != 0:
                raise ValueError(
                    f"dim ({self.dim}) must be divisible by n_heads "
                    f"({self.n_heads}) when head_dim is unset"
                )
            self.head_dim = self.dim // self.n_heads
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be a multiple of "
                f"n_kv_heads ({self.n_kv_heads}) for GQA"
            )
        if self.qk_rope_head_dim is None:
            self.qk_rope_head_dim = max(2, (self.head_dim // 2 // 2) * 2)


__all__ = ["ArchConfig"]
