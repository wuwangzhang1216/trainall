"""Causal language-modelling objective — the bedrock pretraining loss.

The standard next-token maximum-likelihood objective: minimise the negative
log-likelihood of each token given its prefix (Bengio et al. 2003; Radford et
al. 2018, "Improving Language Understanding by Generative Pre-Training").  Every
other likelihood-based loss in the library (CPT, SFT) is a thin variation of
this one.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

from ..base import Objective
from ..registry import register
from ..types import Batch
from ..utils.tensorops import gather_token_logps


def _logits_of(out: Any) -> Any:
    """Return the logits tensor from a model output (HF object or bare tensor)."""
    return out.logits if hasattr(out, "logits") else out


@register("pretrain", category="objective", aliases=["clm"])
class CausalLMObjective(Objective):
    """Next-token cross-entropy over all (unmasked) label positions.

    Loss is ``-(sum of per-token log p) / (number of tokens)`` after the causal
    shift handled by :func:`gather_token_logps`.  Positions with label
    ``-100`` are ignored.  A model-reported ``aux_loss`` (e.g. MoE
    load-balancing) is added to the returned loss but excluded from perplexity.
    """

    def compute_loss(self, model: Any, batch: Batch) -> Tuple[Any, Dict[str, float]]:
        from .._optional import require

        torch = require("torch", feature="causal LM loss")

        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        # Labels default to input_ids (pure LM) when not supplied.
        labels = batch.get("labels", input_ids)

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = _logits_of(out)

        token_logp, mask = gather_token_logps(logits, labels)
        n_tokens = mask.sum().clamp(min=1)
        nll = -token_logp.sum() / n_tokens

        loss = nll
        aux = getattr(out, "aux_loss", None)
        if aux is not None:
            loss = loss + aux

        with torch.no_grad():
            ppl = torch.exp(nll.detach())
        metrics = {
            "loss": float(loss.detach()),
            "ppl": float(ppl),
            "n_tokens": float(mask.sum().detach()),
        }
        return loss, metrics


__all__ = ["CausalLMObjective"]
