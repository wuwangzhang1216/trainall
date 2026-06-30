"""Kahneman-Tversky Optimization (KTO, Ethayarajh et al. 2024).

KTO drops the *paired* assumption of DPO: each example is a single response
tagged desirable or undesirable, and the loss is a prospect-theory utility on
the (policy - reference) log-ratio relative to a KL baseline ``z``::

    r = logp_pi - logp_ref
    desirable:   w_d * (1 - sigmoid(beta * (r - z)))
    undesirable: w_u * (1 - sigmoid(beta * (z - r)))

Simplification (documented): the reference KL term ``z`` is estimated here as
the detached, non-negative batch mean of ``r`` — ``z = mean(r).detach().clamp(min=0)``
— rather than the mismatched-pair KL of the original implementation.  This keeps
the objective single-pass and dependency-free while preserving the
"reward relative to a shared baseline" shape that drives KTO.

Labels are read from ``batch.extra['labels']`` (a bool tensor / sequence, True ==
desirable).  Each row carries its log-probs in the ``chosen_*`` fields; a row's
``rejected_*`` fields are ignored for the unpaired loss.
"""
from __future__ import annotations

from typing import Any, Tuple

from ...base import Objective
from ...registry import register
from ...utils.tensorops import sequence_logps
from .dpo import _logits


@register("kto", category="objective")
class KTOObjective(Objective):
    """KTO unpaired prospect-theory loss (Ethayarajh et al. 2024).

    Args:
        beta: Temperature on the centred log-ratio.
        desirable_weight: Loss weight for desirable samples.
        undesirable_weight: Loss weight for undesirable samples.
    """

    requires_reference_model = True

    def __init__(
        self,
        beta: float = 0.1,
        desirable_weight: float = 1.0,
        undesirable_weight: float = 1.0,
    ) -> None:
        self.beta = float(beta)
        self.desirable_weight = float(desirable_weight)
        self.undesirable_weight = float(undesirable_weight)

    def compute_loss(self, model: Any, batch: Any) -> Tuple[Any, dict]:
        from ..._optional import require

        torch = require("torch", feature="KTO loss")

        logits = _logits(model, batch["chosen_input_ids"], batch.get("chosen_attention_mask"))
        pi = sequence_logps(logits, batch["chosen_labels"], average=False)
        ref = self._ref(torch, batch)

        r = pi - ref
        # Simplified KL baseline: shared, detached, non-negative batch mean.
        z = r.mean().detach().clamp(min=0)

        labels = self._labels(torch, batch, r)
        desirable = labels.to(r.dtype)
        undesirable = 1.0 - desirable

        loss_d = self.desirable_weight * (1.0 - torch.sigmoid(self.beta * (r - z)))
        loss_u = self.undesirable_weight * (1.0 - torch.sigmoid(self.beta * (z - r)))
        per_sample = desirable * loss_d + undesirable * loss_u
        loss = per_sample.mean()

        metrics = {
            "loss": float(loss.detach()),
            "kl_baseline": float(z.detach()),
            "logratio_mean": float(r.mean().detach()),
            "frac_desirable": float(desirable.mean().detach()),
        }
        return loss, metrics

    def _ref(self, torch: Any, batch: Any) -> Any:
        if "ref_chosen_logps" in batch:
            return batch["ref_chosen_logps"]
        ref_model = batch.extra.get("ref_model")
        if ref_model is not None:
            with torch.no_grad():
                logits = _logits(
                    ref_model, batch["chosen_input_ids"], batch.get("chosen_attention_mask")
                )
                return sequence_logps(logits, batch["chosen_labels"], average=False)
        raise ValueError(
            "KTO requires a reference: provide ref_chosen_logps or batch.extra['ref_model']."
        )

    def _labels(self, torch: Any, batch: Any, like: Any) -> Any:
        raw = batch.extra.get("labels")
        if raw is None:
            raise ValueError(
                "KTO needs per-sample desirability in batch.extra['labels'] "
                "(True == desirable)."
            )
        if not torch.is_tensor(raw):
            raw = torch.as_tensor(list(raw))
        return raw.to(device=like.device, dtype=torch.bool)


__all__ = ["KTOObjective"]
