"""Full fine-tuning — train every parameter.

The simplest :class:`~trainall.base.Algorithm`: it adapts nothing.  Every weight
in the model stays trainable and the optimiser moves all of them.  This is the
baseline against which parameter-efficient methods (LoRA, QLoRA) trade quality
for memory.

``torch`` is kept lazy here so importing the algorithm registry stays cheap; the
class only needs torch when persisting a checkpoint (handled by the base class).
"""
from __future__ import annotations

from typing import Any, Iterable

from ..base import Algorithm
from ..registry import register


@register("full", category="algorithm")
class FullFinetune(Algorithm):
    """Dense full-parameter fine-tuning.

    ``prepare_model`` is a pass-through: the model is returned unchanged with all
    parameters trainable.  This is the canonical full-SFT / full-RLHF setup.
    """

    def prepare_model(self, model: Any) -> Any:
        """Return ``model`` unchanged; ensure every parameter is trainable."""
        params = getattr(model, "parameters", None)
        if callable(params):
            for p in model.parameters():
                if hasattr(p, "requires_grad_"):
                    p.requires_grad_(True)
        return model

    def trainable_parameters(self, model: Any) -> Iterable[Any]:
        """Yield *all* parameters — full fine-tuning moves the whole model."""
        return model.parameters()


__all__ = ["FullFinetune"]
