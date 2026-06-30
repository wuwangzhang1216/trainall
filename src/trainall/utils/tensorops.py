"""Token-level log-prob plumbing shared by every likelihood-based objective.

These are the small, easy-to-get-wrong tensor ops at the core of SFT, DPO,
PPO and GRPO.  Implementing them once, correctly, is half the value of the
library.  ``torch`` is imported lazily so this module is importable without it.
"""
from __future__ import annotations

from typing import Any


def _torch() -> Any:
    from .._optional import require

    return require("torch", feature="tensor ops")


def gather_token_logps(logits: Any, labels: Any, ignore_index: int = -100) -> Any:
    """Per-token log p(label) under ``logits``, shifted for causal LMs.

    ``logits``: ``(B, T, V)`` — model outputs aligned to ``input_ids``.
    ``labels``: ``(B, T)``    — same alignment; ``ignore_index`` positions are
    zeroed in the output and masked by the returned ``mask``.

    Returns ``(per_token_logps, mask)`` both shaped ``(B, T-1)`` after the
    standard "predict next token" shift.
    """
    torch = _torch()
    # shift: token t's logits predict token t+1
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]
    mask = labels != ignore_index
    safe_labels = labels.masked_fill(~mask, 0)
    logp = torch.log_softmax(logits.float(), dim=-1)
    token_logp = torch.gather(logp, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logp = token_logp * mask
    return token_logp, mask


def sequence_logps(logits: Any, labels: Any, ignore_index: int = -100, average: bool = False) -> Any:
    """Sum (or mean) of per-token log-probs over each sequence -> ``(B,)``.

    ``average=True`` gives the length-normalised log-prob used by IPO / SimPO.
    """
    token_logp, mask = gather_token_logps(logits, labels, ignore_index)
    seq = token_logp.sum(dim=-1)
    if average:
        denom = mask.sum(dim=-1).clamp(min=1)
        seq = seq / denom
    return seq


def masked_mean(values: Any, mask: Any, dim: Any = None) -> Any:
    """Mean of ``values`` over positions where ``mask`` is truthy."""
    torch = _torch()
    mask = mask.to(values.dtype)
    if dim is None:
        return (values * mask).sum() / mask.sum().clamp(min=1)
    return (values * mask).sum(dim=dim) / mask.sum(dim=dim).clamp(min=1)


def masked_sum(values: Any, mask: Any, dim: Any = None) -> Any:
    mask = mask.to(values.dtype)
    if dim is None:
        return (values * mask).sum()
    return (values * mask).sum(dim=dim)


def entropy_from_logits(logits: Any) -> Any:
    """Token-level entropy ``-sum p log p`` -> ``(B, T)``; used for KL/explore."""
    torch = _torch()
    logp = torch.log_softmax(logits.float(), dim=-1)
    p = logp.exp()
    return -(p * logp).sum(dim=-1)


__all__ = [
    "gather_token_logps",
    "sequence_logps",
    "masked_mean",
    "masked_sum",
    "entropy_from_logits",
]
