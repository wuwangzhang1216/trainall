"""Identity Preference Optimization (IPO, Azar et al. 2023).

IPO replaces DPO's logistic link with a squared loss on the implicit-reward
margin, regressing it toward ``1 / (2*beta)``.  This avoids DPO's tendency to
drive the margin to extremes (over-fitting deterministic preferences).  Margins
use *length-normalised* log-probs::

    h    = (avg_logp_pi(chosen)  - avg_logp_ref(chosen))
         - (avg_logp_pi(rejected)- avg_logp_ref(rejected))
    loss = (h - 1/(2*beta))**2
"""
from __future__ import annotations

from typing import Any, Tuple

from ...base import Objective
from ...registry import register
from .dpo import _policy_logps


@register("ipo", category="objective")
class IPOObjective(Objective):
    """IPO squared-margin preference loss (Azar et al. 2023).

    Args:
        beta: Sets the regression target ``1/(2*beta)`` for the margin.
    """

    requires_reference_model = True

    def __init__(self, beta: float = 0.1) -> None:
        self.beta = float(beta)

    def compute_loss(self, model: Any, batch: Any) -> Tuple[Any, dict]:
        from ..._optional import require

        torch = require("torch", feature="IPO loss")

        pi_c, pi_r = _policy_logps(model, batch, average=True)
        ref_c, ref_r = self._ref(torch, batch)

        h = (pi_c - ref_c) - (pi_r - ref_r)
        target = 1.0 / (2.0 * self.beta)
        loss = ((h - target) ** 2).mean()

        metrics = {
            "loss": float(loss.detach()),
            "reward_acc": float((h > 0).float().mean().detach()),
            "reward_margin": float(h.mean().detach()),
        }
        return loss, metrics

    def _ref(self, torch: Any, batch: Any) -> Tuple[Any, Any]:
        if "ref_chosen_logps" in batch and "ref_rejected_logps" in batch:
            return batch["ref_chosen_logps"], batch["ref_rejected_logps"]
        ref_model = batch.extra.get("ref_model")
        if ref_model is not None:
            with torch.no_grad():
                return _policy_logps(ref_model, batch, average=True)
        raise ValueError(
            "IPO requires a reference: provide length-normalised "
            "ref_chosen_logps/ref_rejected_logps or batch.extra['ref_model']."
        )


__all__ = ["IPOObjective"]
