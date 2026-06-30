"""Simple Preference Optimization (SimPO, Meng et al. 2024).

SimPO is *reference-free*: the implicit reward is just the length-normalised
average log-prob (no reference model), and a target margin ``gamma`` separates
chosen from rejected::

    r_c  = beta * avg_logp(chosen)
    r_r  = beta * avg_logp(rejected)
    loss = -log sigmoid(r_c - r_r - gamma)

Length normalisation removes DPO's length bias; ``gamma`` enforces a minimum
reward gap.
"""
from __future__ import annotations

from typing import Any, Tuple

from ...base import Objective
from ...registry import register
from .dpo import _policy_logps


@register("simpo", category="objective")
class SimPOObjective(Objective):
    """SimPO reference-free length-normalised preference loss (Meng et al. 2024).

    Args:
        beta: Reward scaling on the average log-prob.
        gamma: Target reward margin subtracted before the logistic link.
    """

    requires_reference_model = False

    def __init__(self, beta: float = 2.0, gamma: float = 0.5) -> None:
        self.beta = float(beta)
        self.gamma = float(gamma)

    def compute_loss(self, model: Any, batch: Any) -> Tuple[Any, dict]:
        from ..._optional import require

        torch = require("torch", feature="SimPO loss")
        F = torch.nn.functional

        avg_c, avg_r = _policy_logps(model, batch, average=True)
        r_c = self.beta * avg_c
        r_r = self.beta * avg_r
        logits = r_c - r_r - self.gamma
        loss = -F.logsigmoid(logits).mean()

        metrics = {
            "loss": float(loss.detach()),
            "reward_acc": float((logits > 0).float().mean().detach()),
            "reward_margin": float((r_c - r_r).mean().detach()),
        }
        return loss, metrics


__all__ = ["SimPOObjective"]
