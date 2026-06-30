"""Training callbacks — hooks the :class:`~trainall.training.Trainer` fires.

A callback is a tiny observer over the training loop, in the spirit of the
Keras / HuggingFace ``Trainer`` callback protocol.  The trainer calls
:meth:`Callback.on_train_begin` once before the loop, :meth:`on_step_end` after
each optimiser step (with the running metrics), and :meth:`on_train_end` when
the loop finishes.  Callbacks own no parameters and never touch the model's
weights — they observe and report.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from ..utils import get_logger

__all__ = ["Callback", "LoggingCallback"]


class Callback:
    """Base no-op callback.

    Subclass and override the hooks you care about.  Every hook receives the
    owning trainer (so a callback can read ``trainer.config`` / ``trainer.model``)
    plus hook-specific arguments.  Defaults do nothing so subclasses only
    implement what they need.
    """

    def on_train_begin(self, trainer: Any = None, **kwargs: Any) -> None:
        """Called once before the first optimiser step."""

    def on_step_end(self, step: int, metrics: Dict[str, float], trainer: Any = None, **kwargs: Any) -> None:
        """Called after every optimiser step with the running ``metrics``."""

    def on_train_end(self, trainer: Any = None, **kwargs: Any) -> None:
        """Called once after the loop terminates (or hits ``max_steps``)."""


class LoggingCallback(Callback):
    """Print running metrics every ``log_every`` steps via ``utils.get_logger``.

    The trainer already throttles its own logging, but this callback gives users
    an independent, configurable cadence and a clean one-line metric dump that is
    easy to scrape.  It is the default callback installed by the trainer.
    """

    def __init__(self, log_every: int = 10, logger_name: str = "trainer") -> None:
        self.log_every = max(1, int(log_every))
        self._log = get_logger(logger_name)

    def on_train_begin(self, trainer: Any = None, **kwargs: Any) -> None:
        self._log.info("training started")

    def on_step_end(self, step: int, metrics: Dict[str, float], trainer: Any = None, **kwargs: Any) -> None:
        if step % self.log_every != 0:
            return
        self._log.info("step %d | %s", step, self._fmt(metrics))

    def on_train_end(self, trainer: Any = None, **kwargs: Any) -> None:
        self._log.info("training finished")

    @staticmethod
    def _fmt(metrics: Optional[Dict[str, float]]) -> str:
        if not metrics:
            return "(no metrics)"
        parts = []
        for k, v in metrics.items():
            try:
                parts.append(f"{k}={float(v):.4f}")
            except (TypeError, ValueError):
                parts.append(f"{k}={v}")
        return " ".join(parts)
