"""Modern decoder-only architecture building blocks.

This subpackage is **torch-required by design**: its classes subclass
``torch.nn.Module`` at definition time.  Importing ``trainall`` does *not* pull
this in; ``import trainall.models`` (or ``trainall.models.*``) does.

Pieces map to the frontier decoder-only stack:

* :class:`RMSNorm`                      — pre-norm normalisation
* :class:`RotaryEmbedding`              — RoPE with NTK / YaRN scaling
* :class:`Attention`                    — GQA / MQA (set ``num_kv_heads``)
* :class:`MultiHeadLatentAttention`     — DeepSeek-style MLA (KV compression)
* :class:`SwiGLU` / :class:`GeGLU`      — gated MLPs
* :class:`MoEFeedForward`               — top-k routed experts + aux loss
* :class:`DecoderBlock` / :class:`DecoderLM` — assembled model
"""
from __future__ import annotations

from .config import ArchConfig
from .norm import RMSNorm
from .rope import RotaryEmbedding, apply_rotary
from .attention import Attention, MultiHeadLatentAttention
from .mlp import GeGLU, SwiGLU
from .moe import MoEFeedForward
from .block import DecoderBlock
from .transformer import DecoderLM

__all__ = [
    "ArchConfig",
    "RMSNorm",
    "RotaryEmbedding",
    "apply_rotary",
    "Attention",
    "MultiHeadLatentAttention",
    "SwiGLU",
    "GeGLU",
    "MoEFeedForward",
    "DecoderBlock",
    "DecoderLM",
]
