"""Bradley-Terry reward-model training — pairwise preference scoring.

A reward model learns a scalar ``r(x, y)`` such that preferred responses score
higher than dispreferred ones.  Under the Bradley-Terry model
(Bradley & Terry, 1952; as used for RLHF reward models in
Ouyang et al., 2022 / Christiano et al., 2017) the probability that ``chosen``
beats ``rejected`` is ``sigmoid(r_chosen - r_rejected)``, so the loss is
``-log sigmoid(r_chosen - r_rejected)``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

from ..base import Objective
from ..registry import register


def _reward_for(model: Any, input_ids: Any, attention_mask: Any, extra: Dict[str, Any]) -> Any:
    """Return a scalar reward per sequence ``(B,)``.

    Strategy, in order of preference:

    1. If the model exposes a value/score head producing ``(B, T)`` or
       ``(B, T, 1)`` scores (HF ``...ForSequenceClassification`` returns
       ``logits`` of shape ``(B, num_labels)``), take the value at the last
       non-pad token (or the single logit).
    2. Otherwise run the model for ``hidden_states`` and apply a scalar head
       supplied in ``extra["scalar_head"]`` to the mean of the last hidden
       state over non-pad tokens.
    """
    from .._optional import require

    torch = require("torch", feature="reward-model loss")

    # Index of the last non-pad token per sequence.
    lengths = attention_mask.long().sum(dim=-1) - 1  # (B,)
    lengths = lengths.clamp(min=0)
    bsz = input_ids.shape[0]
    arange = torch.arange(bsz, device=input_ids.device)

    out = model(input_ids=input_ids, attention_mask=attention_mask)

    # Case 1a: an explicit scalar/value head on the output object.
    value = getattr(out, "value", None)
    if value is None:
        value = getattr(out, "values", None)
    if value is not None:
        v = value
        if v.dim() == 3 and v.shape[-1] == 1:
            v = v.squeeze(-1)
        if v.dim() == 2:  # (B, T)
            return v[arange, lengths]
        if v.dim() == 1:  # (B,)
            return v

    # Case 1b: sequence-classification style logits (B, num_labels).
    logits = getattr(out, "logits", out)
    if logits.dim() == 2:
        if logits.shape[-1] == 1:
            return logits.squeeze(-1)
        return logits[:, 0]
    if logits.dim() == 3 and logits.shape[-1] == 1:  # (B, T, 1) per-token score
        return logits.squeeze(-1)[arange, lengths]

    # Case 2: no value head -> mean-pool last hidden state through scalar head.
    hidden = getattr(out, "hidden_states", None)
    if hidden is not None and isinstance(hidden, (list, tuple)):
        hidden = hidden[-1]
    if hidden is None:
        hidden = getattr(out, "last_hidden_state", None)
    if hidden is None:
        # As a last resort treat (B, T, V) logits as a feature and mean over V.
        hidden = logits
    mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (B, H)
    head = extra.get("scalar_head")
    if head is None:
        raise ValueError(
            "model has no value head; provide batch.extra['scalar_head'] "
            "(a callable mapping (B, H) hidden states to (B,) or (B, 1) rewards)"
        )
    r = head(pooled)
    if r.dim() == 2 and r.shape[-1] == 1:
        r = r.squeeze(-1)
    return r


@register("reward_model", category="objective", aliases=["rm", "bt"])
class BradleyTerryObjective(Objective):
    """Pairwise Bradley-Terry reward-model loss.

    ``loss = -log sigmoid(r_chosen - r_rejected)`` where each reward is the
    scalar score of the last non-pad token.  ``acc`` is the fraction of pairs
    where the chosen reward exceeds the rejected one.
    """

    def __init__(self) -> None:
        pass

    def compute_loss(self, model: Any, batch: Any) -> Tuple[Any, Dict[str, float]]:
        from .._optional import require

        torch = require("torch", feature="reward-model loss")
        nn = torch.nn

        r_chosen = _reward_for(
            model,
            batch["chosen_input_ids"],
            batch["chosen_attention_mask"],
            batch.extra,
        )
        r_rejected = _reward_for(
            model,
            batch["rejected_input_ids"],
            batch["rejected_attention_mask"],
            batch.extra,
        )

        margin = r_chosen - r_rejected
        loss = -nn.functional.logsigmoid(margin).mean()
        acc = (margin > 0).float().mean()
        return loss, {
            "loss": float(loss.detach()),
            "acc": float(acc.detach()),
            "reward_margin": float(margin.detach().mean()),
        }


__all__ = ["BradleyTerryObjective"]
