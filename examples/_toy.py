"""Shared toy helpers for the trainall examples.

Everything here is *tiny* and CPU-only: a byte-level tokenizer (vocab 256) and a
minimal :class:`~trainall.models.DecoderLM` factory.  The examples import from
this module so each script stays focused on the API it is demonstrating rather
than on boilerplate.

Run ``import trainall`` works with no ML stack, but the examples themselves do
need torch installed (they actually train); that is expected.
"""
from __future__ import annotations

from typing import List, Sequence

from trainall.models import ArchConfig, DecoderLM

VOCAB_SIZE = 256  # one id per byte — no training / files required.
MAX_SEQ_LEN = 64


def encode(text: str, max_len: int = MAX_SEQ_LEN) -> List[int]:
    """Byte-level encode ``text`` into a list of ints in ``[0, 256)``."""
    return list(text.encode("utf-8"))[:max_len]


def decode(ids: Sequence[int]) -> str:
    """Inverse of :func:`encode` (lossy on truncation)."""
    return bytes(int(i) % 256 for i in ids).decode("utf-8", errors="replace")


def tiny_model(n_layers: int = 2, **overrides) -> DecoderLM:
    """Build a tiny CPU-friendly :class:`DecoderLM` (a few thousand params)."""
    cfg = ArchConfig(
        vocab_size=VOCAB_SIZE,
        dim=32,
        n_layers=n_layers,
        n_heads=4,
        n_kv_heads=2,
        ffn_dim=64,
        max_seq_len=MAX_SEQ_LEN,
        tie_embeddings=True,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return DecoderLM.from_config(cfg)


def sft_record(prompt: str, response: str, max_len: int = MAX_SEQ_LEN) -> dict:
    """Tokenise a prompt/response pair into a collate-ready SFT record.

    Returns a dict with ``input_ids`` (prompt+response) and ``labels`` whose
    prompt positions are masked to ``-100`` so the SFT loss only covers the
    response — exactly what ``trainall``'s ``default_collate`` expects.
    """
    p_ids = encode(prompt, max_len)
    r_ids = encode(response, max_len - len(p_ids))
    input_ids = p_ids + r_ids
    labels = [-100] * len(p_ids) + list(r_ids)
    return {"input_ids": input_ids, "labels": labels}


def lm_record(text: str, max_len: int = MAX_SEQ_LEN) -> dict:
    """Tokenise raw text into a pretrain/CPT record (every token is a target)."""
    ids = encode(text, max_len)
    return {"input_ids": ids, "labels": list(ids)}
