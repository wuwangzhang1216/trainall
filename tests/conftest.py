"""Shared fixtures and helpers for the trainall test suite.

Everything here builds *tiny* objects (a ~few-thousand-param ``DecoderLM``,
random integer tensors) so each test runs on CPU in well under a second.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def _seed():
    """Deterministic tensors across the suite."""
    torch.manual_seed(0)
    yield


@pytest.fixture
def arch():
    """A minimal dense ``ArchConfig`` instance."""
    from trainall.models import ArchConfig

    return ArchConfig(
        vocab_size=37,
        dim=16,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ffn_dim=32,
        max_seq_len=32,
    )


@pytest.fixture
def tiny_model(arch):
    """A tiny ``DecoderLM`` ready for forward/backward on CPU."""
    from trainall.models import DecoderLM

    return DecoderLM.from_config(arch)


def make_ids(batch=2, seqlen=6, vocab=37):
    """Random token ids ``(batch, seqlen)``."""
    return torch.randint(0, vocab, (batch, seqlen))


def causal_batch(batch=2, seqlen=6, vocab=37, mask_prompt=0):
    """A causal-LM/SFT :class:`Batch` with optional prompt masking.

    ``mask_prompt`` positions in ``labels`` are set to ``-100``.
    """
    from trainall.types import Batch

    ids = make_ids(batch, seqlen, vocab)
    labels = ids.clone()
    if mask_prompt:
        labels[:, :mask_prompt] = -100
    return Batch.of(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        labels=labels,
    )


def preference_batch(batch=2, seqlen=6, vocab=37, with_ref=True):
    """A preference :class:`Batch` (chosen/rejected) for DPO-family objectives.

    Labels are unmasked (whole sequence supervised) which is fine for the
    per-sequence log-prob math the preference losses use.
    """
    from trainall.types import Batch

    cids = make_ids(batch, seqlen, vocab)
    rids = make_ids(batch, seqlen, vocab)
    tensors = dict(
        chosen_input_ids=cids,
        chosen_attention_mask=torch.ones_like(cids),
        chosen_labels=cids.clone(),
        rejected_input_ids=rids,
        rejected_attention_mask=torch.ones_like(rids),
        rejected_labels=rids.clone(),
    )
    return Batch(tensors=tensors)
