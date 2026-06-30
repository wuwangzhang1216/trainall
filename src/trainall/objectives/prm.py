"""Process Reward Model — step-level supervision (Lightman et al., 2023).

Rather than scoring only the final answer, a PRM labels each intermediate
reasoning *step* as correct or not.  At the step-delimiter positions
(``step_mask == 1``) the model emits a scalar logit, trained with
binary-cross-entropy against per-step labels ``step_labels``.  This is the
"let's verify step by step" signal used to train verifiers for best-of-N search.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

from ..base import Objective
from ..registry import register


def _step_logits(model: Any, input_ids: Any, attention_mask: Any, extra: Dict[str, Any]) -> Any:
    """Return a per-token scalar logit ``(B, T)``.

    Uses the model's value head when present (output ``.value``/``.values`` of
    shape ``(B, T)`` or ``(B, T, 1)``); otherwise reads the logit of a special
    token id given in ``extra["positive_token_id"]`` from the LM logits.
    """
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    value = getattr(out, "value", None)
    if value is None:
        value = getattr(out, "values", None)
    if value is not None:
        if value.dim() == 3 and value.shape[-1] == 1:
            value = value.squeeze(-1)
        return value  # (B, T)

    logits = getattr(out, "logits", out)  # (B, T, V)
    token_id = extra.get("positive_token_id")
    if token_id is None:
        raise ValueError(
            "model has no value head; provide batch.extra['positive_token_id'] "
            "so PRM can read the logit of a special 'good-step' token"
        )
    return logits[..., int(token_id)]  # (B, T)


@register("prm", category="objective")
class ProcessRewardObjective(Objective):
    """Per-step BCE-with-logits at step-delimiter positions.

    ``loss`` is the mean BCE over positions where ``step_mask == 1`` against
    ``step_labels`` (0/1); ``step_acc`` is the thresholded accuracy there.
    """

    def __init__(self) -> None:
        pass

    def compute_loss(self, model: Any, batch: Any) -> Tuple[Any, Dict[str, float]]:
        from .._optional import require

        torch = require("torch", feature="PRM loss")

        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        logits = _step_logits(model, input_ids, attention_mask, batch.extra)  # (B, T)
        step_mask = batch["step_mask"].to(torch.bool)
        step_labels = batch["step_labels"].to(logits.dtype)

        sel_logits = logits[step_mask]
        sel_labels = step_labels[step_mask]
        if sel_logits.numel() == 0:
            loss = (logits.sum() * 0.0)  # keep graph connected, zero signal
            return loss, {"loss": 0.0, "step_acc": 0.0}

        loss = torch.nn.functional.binary_cross_entropy_with_logits(sel_logits, sel_labels)
        preds = (sel_logits > 0).to(sel_labels.dtype)
        step_acc = (preds == sel_labels).float().mean()
        return loss, {
            "loss": float(loss.detach()),
            "step_acc": float(step_acc.detach()),
        }


__all__ = ["ProcessRewardObjective"]
