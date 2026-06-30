"""Direct Preference Optimization (DPO, Rafailov et al. 2023).

DPO reparameterises the RLHF reward as the log-ratio between the policy and a
frozen reference model and fits it to pairwise preferences with a simple
classification loss::

    delta = (logp_pi(chosen)  - logp_ref(chosen))
          - (logp_pi(rejected)- logp_ref(rejected))
    loss  = -log sigmoid(beta * delta)

with optional conservative ``label_smoothing`` (cDPO) and an alternative
``hinge`` link (SLiC-style).
"""
from __future__ import annotations

from typing import Any, Tuple

from ...base import Objective
from ...registry import register
from ...utils.tensorops import sequence_logps


def _logits(model: Any, input_ids: Any, attention_mask: Any) -> Any:
    """Run ``model`` and return logits, tolerating a raw-tensor return."""
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    return out.logits if hasattr(out, "logits") else out


def _policy_logps(model: Any, batch: Any, average: bool) -> Tuple[Any, Any]:
    """Per-sequence log-probs of the policy for the chosen/rejected sides."""
    c = _logits(model, batch["chosen_input_ids"], batch.get("chosen_attention_mask"))
    r = _logits(model, batch["rejected_input_ids"], batch.get("rejected_attention_mask"))
    pi_c = sequence_logps(c, batch["chosen_labels"], average=average)
    pi_r = sequence_logps(r, batch["rejected_labels"], average=average)
    return pi_c, pi_r


@register("dpo", category="objective")
class DPOObjective(Objective):
    """DPO preference loss (Rafailov et al. 2023).

    Args:
        beta: Temperature on the implicit-reward margin.
        label_smoothing: Conservative-DPO smoothing ``eps`` in ``[0, 0.5)``.
        loss_type: ``"sigmoid"`` (logistic) or ``"hinge"`` (SLiC margin).
    """

    requires_reference_model = True

    def __init__(
        self,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        loss_type: str = "sigmoid",
    ) -> None:
        self.beta = float(beta)
        self.label_smoothing = float(label_smoothing)
        self.loss_type = loss_type

    def compute_loss(self, model: Any, batch: Any) -> Tuple[Any, dict]:
        from ..._optional import require

        torch = require("torch", feature="DPO loss")
        F = torch.nn.functional

        pi_c, pi_r = _policy_logps(model, batch, average=False)
        ref_c, ref_r = self._ref(torch, batch)

        delta = (pi_c - ref_c) - (pi_r - ref_r)

        if self.loss_type == "hinge":
            loss = F.relu(1.0 - self.beta * delta).mean()
        elif self.loss_type == "sigmoid":
            eps = self.label_smoothing
            loss = (
                -(1.0 - eps) * F.logsigmoid(self.beta * delta)
                - eps * F.logsigmoid(-self.beta * delta)
            ).mean()
        else:  # pragma: no cover - guarded by config
            raise ValueError(f"unknown DPO loss_type {self.loss_type!r}")

        metrics = {
            "loss": float(loss.detach()),
            "reward_acc": float((delta > 0).float().mean().detach()),
            "reward_margin": float(delta.mean().detach()),
        }
        return loss, metrics

    def _ref(self, torch: Any, batch: Any) -> Tuple[Any, Any]:
        if "ref_chosen_logps" in batch and "ref_rejected_logps" in batch:
            return batch["ref_chosen_logps"], batch["ref_rejected_logps"]
        ref_model = batch.extra.get("ref_model")
        if ref_model is not None:
            with torch.no_grad():
                return _policy_logps(ref_model, batch, average=False)
        raise ValueError(
            "DPO requires a reference: provide ref_chosen_logps/ref_rejected_logps "
            "or batch.extra['ref_model']."
        )


__all__ = ["DPOObjective"]
