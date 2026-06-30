"""Continued / domain-adaptive pretraining objective.

Continued pretraining (a.k.a. domain-adaptive pretraining, DAPT; Gururangan et
al. 2020, "Don't Stop Pretraining") keeps the causal-LM loss but trains on a new
corpus.  *Mechanically* it is identical to plain pretraining — CPT is mostly a
**data** concern (curating the domain corpus and a replay mixture of the
original distribution to avoid catastrophic forgetting).

The one knob this objective adds is optional **per-sample loss reweighting**, so
a collator can up- or down-weight replay vs. domain examples within a batch
(``batch.extra["weights"]``) without changing the data pipeline.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

from ..registry import register
from ..types import Batch
from ..utils.tensorops import gather_token_logps
from .pretrain import CausalLMObjective, _logits_of


@register("cpt", category="objective", aliases=["dapt"])
class ContinuedPretrainObjective(CausalLMObjective):
    """Causal-LM loss with an optional replay/domain reweighting knob.

    Parameters
    ----------
    replay_weight:
        Default weight applied to samples flagged as *replay* (from the
        original pretraining distribution) when no explicit per-sample weights
        are provided.  ``0.0`` (the default) means "no reweighting": behave
        exactly like :class:`CausalLMObjective`.
    domain_field:
        Name of the field in ``batch.extra`` that marks which samples are
        domain (truthy) vs. replay (falsy), used to derive weights from
        ``replay_weight`` when explicit ``weights`` are absent.

    Reweighting sources, in priority order:

    1. ``batch.extra["weights"]`` — an explicit per-sample weight tensor/list.
    2. ``batch.extra[domain_field]`` + ``replay_weight`` — domain samples get
       weight ``1.0``, replay samples get ``replay_weight``.
    3. Neither present -> uniform weights (plain CPT == plain pretraining).
    """

    def __init__(self, replay_weight: float = 0.0, domain_field: str = "domain") -> None:
        self.replay_weight = float(replay_weight)
        self.domain_field = domain_field

    def _per_sample_weights(self, torch: Any, batch: Batch, batch_size: int, device: Any) -> Any:
        """Resolve a ``(B,)`` weight tensor from ``batch.extra`` (or uniform)."""
        weights = batch.extra.get("weights")
        if weights is not None:
            w = torch.as_tensor(weights, dtype=torch.float32, device=device)
            return w.reshape(batch_size)

        flags = batch.extra.get(self.domain_field)
        if flags is not None and self.replay_weight != 0.0:
            is_domain = torch.as_tensor(flags, device=device).reshape(batch_size).bool()
            w = torch.where(
                is_domain,
                torch.ones(batch_size, dtype=torch.float32, device=device),
                torch.full((batch_size,), self.replay_weight, dtype=torch.float32, device=device),
            )
            return w

        return torch.ones(batch_size, dtype=torch.float32, device=device)

    def compute_loss(self, model: Any, batch: Batch) -> Tuple[Any, Dict[str, float]]:
        from .._optional import require

        torch = require("torch", feature="continued-pretrain loss")

        # Fast path: no reweighting requested -> defer to the parent CLM loss.
        if "weights" not in batch.extra and (
            self.replay_weight == 0.0 or self.domain_field not in batch.extra
        ):
            return super().compute_loss(model, batch)

        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        labels = batch.get("labels", input_ids)

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = _logits_of(out)

        token_logp, mask = gather_token_logps(logits, labels)  # (B, T-1)
        device = token_logp.device
        batch_size = token_logp.shape[0]

        # Per-sample weighted token-mean NLL, then weighted average across the
        # batch so the global scale matches the unweighted case.
        per_sample_tokens = mask.sum(dim=-1).clamp(min=1)  # (B,)
        per_sample_nll = -token_logp.sum(dim=-1) / per_sample_tokens  # (B,)

        w = self._per_sample_weights(torch, batch, batch_size, device)
        w_sum = w.sum().clamp(min=1e-8)
        nll = (per_sample_nll * w).sum() / w_sum

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


__all__ = ["ContinuedPretrainObjective"]
