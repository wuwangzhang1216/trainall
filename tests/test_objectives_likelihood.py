"""Likelihood objectives: CausalLM / CPT / SFT loss finite + backward, masking."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import trainall  # noqa: E402
from conftest import causal_batch  # noqa: E402


def _assert_scalar_loss_backward(model, loss):
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_causal_lm_loss(tiny_model):
    obj = trainall.build("clm", category="objective")
    batch = causal_batch()
    loss, metrics = obj.compute_loss(tiny_model, batch)
    _assert_scalar_loss_backward(tiny_model, loss)
    assert "ppl" in metrics and metrics["ppl"] > 0


def test_cpt_loss_fast_path(tiny_model):
    obj = trainall.build("cpt", category="objective")
    batch = causal_batch()
    loss, metrics = obj.compute_loss(tiny_model, batch)
    _assert_scalar_loss_backward(tiny_model, loss)


def test_cpt_loss_with_weights(tiny_model):
    obj = trainall.build("cpt", category="objective", replay_weight=0.5)
    batch = causal_batch(batch=2)
    batch.extra["weights"] = [1.0, 0.5]
    loss, metrics = obj.compute_loss(tiny_model, batch)
    _assert_scalar_loss_backward(tiny_model, loss)


def test_sft_loss(tiny_model):
    obj = trainall.build("sft", category="objective")
    batch = causal_batch()
    loss, metrics = obj.compute_loss(tiny_model, batch)
    _assert_scalar_loss_backward(tiny_model, loss)
    assert "loss" in metrics


def test_sft_prompt_masking_changes_loss(tiny_model):
    """Masking prompt tokens to -100 must change which tokens contribute."""
    obj = trainall.build("sft", category="objective")
    torch.manual_seed(1)
    full = causal_batch(batch=2, seqlen=8, mask_prompt=0)
    # Same ids/labels but with the first 4 positions masked out.
    masked = type(full)(tensors=dict(full.tensors))
    labels = full["labels"].clone()
    labels[:, :4] = -100
    masked.tensors["labels"] = labels

    loss_full, _ = obj.compute_loss(tiny_model, full)
    loss_masked, _ = obj.compute_loss(tiny_model, masked)
    # Different token subsets -> losses should differ.
    assert not torch.allclose(loss_full, loss_masked)


def test_sft_all_masked_gives_zero_token_loss(tiny_model):
    """When every label is -100 the loss is well-defined (no NaN)."""
    obj = trainall.build("sft", category="objective")
    batch = causal_batch(batch=2, seqlen=6)
    batch.tensors["labels"] = torch.full_like(batch["labels"], -100)
    loss, _ = obj.compute_loss(tiny_model, batch)
    assert torch.isfinite(loss)


def test_sft_label_smoothing(tiny_model):
    obj = trainall.build("sft", category="objective", label_smoothing=0.1)
    batch = causal_batch()
    loss, _ = obj.compute_loss(tiny_model, batch)
    assert torch.isfinite(loss)
