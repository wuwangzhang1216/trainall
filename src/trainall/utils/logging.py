"""A single, dependency-free logger factory used across the library."""
from __future__ import annotations

import logging
import os

_CONFIGURED = False


def _configure() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = os.environ.get("TRAINALL_LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(name)s] %(levelname)s %(message)s", "%H:%M:%S")
    )
    root = logging.getLogger("trainall")
    if not root.handlers:
        root.addHandler(handler)
    root.setLevel(getattr(logging, level, logging.INFO))
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str = "trainall") -> logging.Logger:
    _configure()
    if not name.startswith("trainall"):
        name = f"trainall.{name}"
    return logging.getLogger(name)


__all__ = ["get_logger"]
