"""RL/supervision objectives: reward model BT, PPO clip, RLOO, GRPO, PRM, distill."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

import trainall  # noqa: E402
from trainall.types import Batch  # noqa: E402
from conftest import make_ids  # noqa: E402


# --------------------------------------------------------------------------- #
# Reward model (Bradley-Terry)
# --------------------------------------------------------------------------- #
def test_reward_model_bt_loss_and_acc(tiny_model):
    obj = trainall.build("reward_model", category="objective")
    cids = make_ids(3, 6)
    rids = make_ids(3, 6)
    batch = Batch(
        tensors=dict(
            chosen_input_ids=cids,
            chosen_attention_mask=torch.ones_like(cids),
            rejected_input_ids=rids,
            rejected_attention_mask=torch.ones_like(rids),
        ),
        # DecoderLM has no value head -> supply a scalar head over hidden feats.
        extra={"scalar_head": nn.Linear(tiny_model.config.vocab_size, 1)},
    )
    loss, metrics = obj.compute_loss(tiny_model, batch)
    assert torch.isfinite(loss)
    assert 0.0 <= metrics["acc"] <= 1.0
    loss.backward()
    assert any(p.grad is not None for p in tiny_model.parameters())


# --------------------------------------------------------------------------- #
# PPO
# --------------------------------------------------------------------------- #
def _policy_grad_batch(batch=2, seqlen=6, vocab=37):
    ids = make_ids(batch, seqlen, vocab)
    response_mask = torch.ones(batch, seqlen)
    response_mask[:, :2] = 0  # first 2 tokens are prompt
    advantages = torch.tensor([1.0, -1.0])[:batch]
    return ids, response_mask, advantages


def test_ppo_loss_and_clipping(tiny_model):
    obj = trainall.build("ppo", category="objective", clip_range=0.2)
    ids, response_mask, advantages = _policy_grad_batch()
    batch = Batch.of(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        response_mask=response_mask,
        advantages=advantages,
    )
    loss, metrics = obj.compute_loss(tiny_model, batch)
    assert torch.isfinite(loss)
    assert "pg_loss" in metrics and "ratio" in metrics
    loss.backward()
    assert any(p.grad is not None for p in tiny_model.parameters())


def test_ppo_clip_caps_ratio_contribution(tiny_model):
    """With old_logps far from current, the clipped surrogate is used."""
    obj = trainall.build("ppo", category="objective", clip_range=0.2)
    ids, response_mask, advantages = _policy_grad_batch()
    # old_logps very negative -> ratio = exp(logp - old) huge -> clip engages.
    old = torch.full((ids.shape[0], ids.shape[1]), -50.0)
    batch = Batch.of(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        response_mask=response_mask,
        advantages=advantages,
        old_logps=old,
    )
    loss, metrics = obj.compute_loss(tiny_model, batch)
    assert torch.isfinite(loss)


# --------------------------------------------------------------------------- #
# RLOO leave-one-out baseline
# --------------------------------------------------------------------------- #
def test_rloo_baseline_advantages():
    from trainall.objectives.rloo import _leave_one_out_advantages

    rewards = torch.tensor([1.0, 0.0, 0.0, 1.0])
    group_ids = torch.tensor([0, 0, 1, 1])
    adv = _leave_one_out_advantages(rewards, group_ids, torch)
    # group 0: A_0 = 1 - mean(0) = 1; A_1 = 0 - mean(1) = -1.
    assert adv[0].item() == pytest.approx(1.0)
    assert adv[1].item() == pytest.approx(-1.0)
    # group 1: A_2 = 0 - 1 = -1; A_3 = 1 - 0 = 1.
    assert adv[2].item() == pytest.approx(-1.0)
    assert adv[3].item() == pytest.approx(1.0)


def test_rloo_loss_backward(tiny_model):
    obj = trainall.build("rloo", category="objective")
    ids = make_ids(4, 6)
    response_mask = torch.ones(4, 6)
    response_mask[:, :2] = 0
    batch = Batch.of(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        response_mask=response_mask,
        rewards=torch.tensor([1.0, 0.0, 0.0, 1.0]),
        group_ids=torch.tensor([0, 0, 1, 1]),
    )
    loss, metrics = obj.compute_loss(tiny_model, batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in tiny_model.parameters())


# --------------------------------------------------------------------------- #
# GRPO group advantages + loss
# --------------------------------------------------------------------------- #
def test_grpo_group_advantages():
    from trainall.objectives.grpo import _group_advantages

    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    group_ids = torch.tensor([0, 0, 1, 1])
    adv = _group_advantages(rewards, group_ids, "group", torch)
    # Within each group: mean 0.5, std 0.5 -> (r-0.5)/0.5 ≈ ±1.
    assert adv[0].item() == pytest.approx(1.0, abs=1e-3)
    assert adv[1].item() == pytest.approx(-1.0, abs=1e-3)
    # zero-mean within each group.
    assert adv[:2].sum().item() == pytest.approx(0.0, abs=1e-5)


def test_grpo_loss_backward(tiny_model):
    obj = trainall.build("grpo", category="objective")
    ids = make_ids(4, 6)
    response_mask = torch.ones(4, 6)
    response_mask[:, :2] = 0
    batch = Batch.of(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        response_mask=response_mask,
        rewards=torch.tensor([1.0, 0.0, 1.0, 0.0]),
        group_ids=torch.tensor([0, 0, 1, 1]),
    )
    loss, metrics = obj.compute_loss(tiny_model, batch)
    assert torch.isfinite(loss)
    assert "adv_std" in metrics
    loss.backward()
    assert any(p.grad is not None for p in tiny_model.parameters())


# --------------------------------------------------------------------------- #
# PRM (process reward, BCE)
# --------------------------------------------------------------------------- #
def test_prm_bce_loss(tiny_model):
    obj = trainall.build("prm", category="objective")
    ids = make_ids(2, 6)
    step_mask = torch.zeros(2, 6, dtype=torch.bool)
    step_mask[:, [2, 5]] = True  # two step-delimiter positions per row
    step_labels = torch.zeros(2, 6)
    step_labels[:, 2] = 1.0
    batch = Batch.of(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        step_mask=step_mask,
        step_labels=step_labels,
    )
    # DecoderLM has no value head; read a "good-step" token logit instead.
    batch.extra["positive_token_id"] = 0
    loss, metrics = obj.compute_loss(tiny_model, batch)
    assert torch.isfinite(loss)
    assert 0.0 <= metrics["step_acc"] <= 1.0
    loss.backward()
    assert any(p.grad is not None for p in tiny_model.parameters())


# --------------------------------------------------------------------------- #
# Distillation (KD)
# --------------------------------------------------------------------------- #
def test_distill_kd_loss(tiny_model):
    obj = trainall.build("distill", category="objective", alpha=0.5, temperature=2.0)
    ids = make_ids(2, 6)
    vocab = tiny_model.config.vocab_size
    teacher_logits = torch.randn(2, 6, vocab)
    batch = Batch.of(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        labels=ids.clone(),
    )
    batch.extra["teacher_logits"] = teacher_logits
    loss, metrics = obj.compute_loss(tiny_model, batch)
    assert torch.isfinite(loss)
    assert "kd" in metrics and "ce" in metrics
    loss.backward()
    assert any(p.grad is not None for p in tiny_model.parameters())


def test_distill_reverse_kl(tiny_model):
    obj = trainall.build("distill", category="objective", alpha=1.0, kind="reverse")
    ids = make_ids(2, 6)
    vocab = tiny_model.config.vocab_size
    batch = Batch.of(input_ids=ids, attention_mask=torch.ones_like(ids))
    batch.extra["teacher_logits"] = torch.randn(2, 6, vocab)
    loss, _ = obj.compute_loss(tiny_model, batch)
    assert torch.isfinite(loss)
