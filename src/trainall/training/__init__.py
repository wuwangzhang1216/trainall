"""The training loop that ties objective + algorithm + data together.

* :class:`Trainer` — a minimal, dependency-light torch loop that drives any
  :class:`~trainall.base.Objective` under any :class:`~trainall.base.Algorithm`.
  It also exposes a HuggingFace ``Trainer`` adapter for users who want the full
  ecosystem.  Torch is lazily required when you actually call ``.train()``.
* :class:`Callback` — hooks for logging / eval / checkpointing.
"""
from __future__ import annotations

from .trainer import Trainer, TrainerConfig
from .callbacks import Callback, LoggingCallback

__all__ = ["Trainer", "TrainerConfig", "Callback", "LoggingCallback"]
