"""Supervised fine-tuning objective — instruction following via NLL on responses.

SFT is the causal-LM loss restricted to the *response* tokens of a
prompt/response pair (Ouyang et al. 2022, "Training language models to follow
instructions").  The collator is expected to set prompt positions to ``-100`` in
``labels`` so the loss is computed only over the tokens the model should learn
to generate.  Optional **label smoothing** (Szegedy et al. 2016) regularises the
target distribution toward uniform.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

from ..base import Objective
from ..registry import register
from ..types import Batch
from .pretrain import _logits_of


@register("sft", category="objective")
class SFTObjective(Objective):
    """Cross-entropy over response tokens (prompt masked to ``-100``).

    Parameters
    ----------
    label_smoothing:
        Smoothing coefficient ``eps`` in ``[0, 1)``.  The per-token loss becomes
        ``(1 - eps) * NLL + eps * mean_over_vocab(-log p)``, i.e. it also rewards
        keeping mass on every other token.  ``0.0`` recovers plain NLL.
    train_on_prompt:
        When ``True`` the loss is taken over *all* tokens (prompt + response),
        ignoring the prompt mask in ``labels`` — useful when no masking was
        applied and the whole sequence should be learned.

    Notes
    -----
    A model-reported ``aux_loss`` (MoE balancing) is added to the returned loss.
    """

    def __init__(self, label_smoothing: float = 0.0, train_on_prompt: bool = False) -> None:
        self.label_smoothing = float(label_smoothing)
        self.train_on_prompt = bool(train_on_prompt)

    def compute_loss(self, model: Any, batch: Batch) -> Tuple[Any, Dict[str, float]]:
        from .._optional import require

        torch = require("torch", feature="SFT loss")

        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        labels = batch.get("labels", input_ids)

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = _logits_of(out)

        # Standard causal shift: token t's logits predict token t+1.
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]

        # ``-100`` always marks "no target here" (padding / prompt). We never
        # gather at those positions. ``train_on_prompt`` means the caller is
        # expected to have left the prompt *unmasked* (labels == input_ids), so
        # there is nothing extra to drop; we simply honour every valid label.
        valid = shift_labels != -100
        mask = valid
        safe_labels = shift_labels.masked_fill(~valid, 0)
        logp = torch.log_softmax(shift_logits.float(), dim=-1)  # (B, T-1, V)

        # Negative log-likelihood of the gold token.
        nll_token = -torch.gather(logp, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)

        if self.label_smoothing > 0.0:
            # Uniform component: mean of -log p over the vocabulary.
            smooth_token = -logp.mean(dim=-1)
            eps = self.label_smoothing
            per_token = (1.0 - eps) * nll_token + eps * smooth_token
        else:
            per_token = nll_token

        mask_f = mask.to(per_token.dtype)
        n_tokens = mask_f.sum().clamp(min=1)
        loss = (per_token * mask_f).sum() / n_tokens

        aux = getattr(out, "aux_loss", None)
        if aux is not None:
            loss = loss + aux

        with torch.no_grad():
            # Perplexity reflects the (unsmoothed) NLL of the gold tokens.
            mean_nll = (nll_token * mask_f).sum() / n_tokens
            ppl = torch.exp(mean_nll.detach())
        metrics = {
            "loss": float(loss.detach()),
            "ppl": float(ppl),
        }
        return loss, metrics


__all__ = ["SFTObjective"]
