"""Odds Ratio Preference Optimization (ORPO, Hong et al. 2024).

ORPO is *reference-free*: it folds preference alignment directly into the SFT
loss via an odds-ratio penalty, so no frozen reference model is needed::

    sft      = -mean(avg_logp(chosen))                       # negative log-likelihood
    log_odds = (lp_c - log1mexp(lp_c)) - (lp_r - log1mexp(lp_r))
    or_loss  = -log sigmoid(log_odds)
    loss     = sft + lambda_ * or_loss

where ``lp_*`` are length-normalised log-probs and ``log1mexp(x) = log(1-exp(x))``
is computed in a numerically stable way for ``x < 0``.
"""
from __future__ import annotations

import math
from typing import Any, Tuple

from ...base import Objective
from ...registry import register
from ...utils.tensorops import sequence_logps
from .dpo import _logits


def _log1mexp(x: Any, torch: Any) -> Any:
    """Stable ``log(1 - exp(x))`` for ``x < 0`` (Mächler 2012).

    Uses ``log(-expm1(x))`` near 0 and ``log1p(-exp(x))`` otherwise to keep full
    precision across the range.  ``x`` is clamped just below 0 so log-probs that
    round up to 0 do not produce ``-inf``.
    """
    x = x.clamp(max=-1e-6)
    return torch.where(
        x > math.log(0.5),
        torch.log(-torch.expm1(x)),
        torch.log1p(-torch.exp(x)),
    )


@register("orpo", category="objective")
class ORPOObjective(Objective):
    """ORPO reference-free odds-ratio + SFT loss (Hong et al. 2024).

    Args:
        lambda_: Weight on the odds-ratio preference penalty.
    """

    requires_reference_model = False

    def __init__(self, lambda_: float = 0.1) -> None:
        self.lambda_ = float(lambda_)

    def compute_loss(self, model: Any, batch: Any) -> Tuple[Any, dict]:
        from ..._optional import require

        torch = require("torch", feature="ORPO loss")
        F = torch.nn.functional

        c = _logits(model, batch["chosen_input_ids"], batch.get("chosen_attention_mask"))
        r = _logits(model, batch["rejected_input_ids"], batch.get("rejected_attention_mask"))
        lp_c = sequence_logps(c, batch["chosen_labels"], average=True)
        lp_r = sequence_logps(r, batch["rejected_labels"], average=True)

        sft = -lp_c.mean()
        log_odds = (lp_c - _log1mexp(lp_c, torch)) - (lp_r - _log1mexp(lp_r, torch))
        or_loss = -F.logsigmoid(log_odds).mean()
        loss = sft + self.lambda_ * or_loss

        metrics = {
            "loss": float(loss.detach()),
            "sft_loss": float(sft.detach()),
            "or_loss": float(or_loss.detach()),
            "reward_acc": float((log_odds > 0).float().mean().detach()),
            "log_odds": float(log_odds.mean().detach()),
        }
        return loss, metrics


__all__ = ["ORPOObjective"]
