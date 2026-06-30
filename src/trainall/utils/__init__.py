"""Cross-cutting helpers (tensor ops, seeding, logging)."""
from __future__ import annotations

from .logging import get_logger
from .seeding import seed_everything
from .tensorops import (
    entropy_from_logits,
    gather_token_logps,
    masked_mean,
    masked_sum,
    sequence_logps,
)

__all__ = [
    "get_logger",
    "seed_everything",
    "gather_token_logps",
    "sequence_logps",
    "masked_mean",
    "masked_sum",
    "entropy_from_logits",
]
