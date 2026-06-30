"""Preference objectives: dpo/ipo/kto/orpo/simpo/cpo scalar loss + backward.

Reference-requiring objectives are fed a *copy* of the model as the frozen
reference via ``batch.extra['ref_model']``.
"""
from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

import trainall  # noqa: E402
from conftest import preference_batch  # noqa: E402


def _ref_copy(model):
    ref = copy.deepcopy(model)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


@pytest.mark.parametrize("key", ["dpo", "ipo", "kto", "orpo", "simpo", "cpo"])
def test_preference_loss_finite_and_backward(tiny_model, key):
    obj = trainall.build(key, category="objective")
    batch = preference_batch(batch=3, seqlen=6)
    if key == "kto":
        batch.extra["labels"] = torch.tensor([True, False, True])
    if getattr(obj, "requires_reference_model", False):
        batch.extra["ref_model"] = _ref_copy(tiny_model)

    loss, metrics = obj.compute_loss(tiny_model, batch)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    # Gradient must flow into the (policy) model.
    assert any(p.grad is not None for p in tiny_model.parameters())
    assert "loss" in metrics


def test_reference_required_flags():
    assert trainall.build("dpo", category="objective").requires_reference_model is True
    assert trainall.build("ipo", category="objective").requires_reference_model is True
    assert trainall.build("kto", category="objective").requires_reference_model is True
    assert trainall.build("orpo", category="objective").requires_reference_model is False
    assert trainall.build("simpo", category="objective").requires_reference_model is False
    assert trainall.build("cpo", category="objective").requires_reference_model is False


def test_dpo_loss_lower_when_chosen_clearly_preferred(tiny_model):
    """A large positive implicit-reward margin must yield a smaller DPO loss.

    We bypass the reference model by supplying precomputed reference logps and
    drive the policy/ref gap directly so the sanity check is deterministic.
    """
    obj = trainall.build("dpo", category="objective", beta=0.1)
    batch = preference_batch(batch=4, seqlen=6)

    # Policy logps come from the model; set reference logps so that the
    # delta = (pi_c - ref_c) - (pi_r - ref_r) is large & positive in the
    # "preferred" case and large & negative in the "dispreferred" case.
    with torch.no_grad():
        from trainall.objectives.preference.dpo import _policy_logps

        pi_c, pi_r = _policy_logps(tiny_model, batch, average=False)

    # Preferred: ref makes chosen look much worse than policy, rejected better.
    pref = type(batch)(tensors=dict(batch.tensors))
    pref.tensors["ref_chosen_logps"] = pi_c - 5.0
    pref.tensors["ref_rejected_logps"] = pi_r + 5.0
    loss_pref, _ = obj.compute_loss(tiny_model, pref)

    # Dispreferred: opposite sign.
    disp = type(batch)(tensors=dict(batch.tensors))
    disp.tensors["ref_chosen_logps"] = pi_c + 5.0
    disp.tensors["ref_rejected_logps"] = pi_r - 5.0
    loss_disp, _ = obj.compute_loss(tiny_model, disp)

    assert float(loss_pref.detach()) < float(loss_disp.detach())


def test_dpo_precomputed_ref_logps_path(tiny_model):
    obj = trainall.build("dpo", category="objective")
    batch = preference_batch(batch=2, seqlen=5)
    batch.tensors["ref_chosen_logps"] = torch.zeros(2)
    batch.tensors["ref_rejected_logps"] = torch.zeros(2)
    loss, metrics = obj.compute_loss(tiny_model, batch)
    assert torch.isfinite(loss)
    assert 0.0 <= metrics["reward_acc"] <= 1.0


def test_dpo_missing_reference_raises(tiny_model):
    obj = trainall.build("dpo", category="objective")
    batch = preference_batch(batch=2, seqlen=5)
    with pytest.raises(ValueError):
        obj.compute_loss(tiny_model, batch)
