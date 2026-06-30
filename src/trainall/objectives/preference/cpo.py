"""Contrastive Preference Optimization (CPO, Xu et al. 2024).

CPO is *reference-free*: it pairs a DPO-style contrastive term (over raw, summed
policy log-probs) with a negative-log-likelihood anchor on the chosen response,
approximating DPO's reference KL with a behaviour-cloning regulariser::

    contrastive = -log sigmoid(beta * (logp(chosen) - logp(rejected)))
    nll         = -mean(logp(chosen))
    loss        = contrastive + lambda_ * nll
"""
from __future__ import annotations

from typing import Any, Tuple

from ...base import Objective
from ...registry import register
from .dpo import _policy_logps


@register("cpo", category="objective")
class CPOObjective(Objective):
    """CPO reference-free contrastive + SFT-anchor loss (Xu et al. 2024).

    Args:
        beta: Temperature on the chosen-minus-rejected log-prob margin.
        lambda_: Weight on the NLL (behaviour-cloning) anchor.
    """

    requires_reference_model = False

    def __init__(self, beta: float = 0.1, lambda_: float = 1.0) -> None:
        self.beta = float(beta)
        self.lambda_ = float(lambda_)

    def compute_loss(self, model: Any, batch: Any) -> Tuple[Any, dict]:
        from ..._optional import require

        torch = require("torch", feature="CPO loss")
        F = torch.nn.functional

        pi_c, pi_r = _policy_logps(model, batch, average=False)
        margin = pi_c - pi_r
        contrastive = -F.logsigmoid(self.beta * margin).mean()
        nll = -pi_c.mean()
        loss = contrastive + self.lambda_ * nll

        metrics = {
            "loss": float(loss.detach()),
            "contrastive_loss": float(contrastive.detach()),
            "nll_loss": float(nll.detach()),
            "reward_acc": float((margin > 0).float().mean().detach()),
            "reward_margin": float(margin.mean().detach()),
        }
        return loss, metrics


__all__ = ["CPOObjective"]
