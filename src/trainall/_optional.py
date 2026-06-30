"""Helpers for lazily importing optional heavy dependencies.

The core of ``trainall`` (config, verifiers, rewards, pipeline DSL, self-play
and agentic scaffolding) is pure-python and imports with no third-party ML
stack installed.  Anything that actually trains a model pulls its dependency in
on demand through :func:`require`, so users get a precise, actionable error
instead of a bare ``ModuleNotFoundError`` deep in a stack trace.
"""
from __future__ import annotations

import importlib
import importlib.util
from typing import Any

# Map an importable module name to the extras group that ships it.
_EXTRA_FOR = {
    "torch": "train",
    "transformers": "train",
    "trl": "train",
    "peft": "train",
    "datasets": "train",
    "accelerate": "train",
    "bitsandbytes": "quant",
    "sympy": "verify",
    "jsonschema": "verify",
}


class MissingDependencyError(ImportError):
    """Raised when an optional dependency is needed but not installed."""


def require(module: str, *, feature: str | None = None) -> Any:
    """Import and return ``module`` or raise a helpful install hint.

    Parameters
    ----------
    module:
        Importable module name, e.g. ``"trl"``.
    feature:
        Human-readable name of the feature needing it, used in the message.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - exercised via integration
        extra = _EXTRA_FOR.get(module.split(".")[0], "all")
        what = feature or module
        raise MissingDependencyError(
            f"'{module}' is required for {what} but is not installed.\n"
            f"Install it with:  pip install 'trainall[{extra}]'"
        ) from exc


def has(module: str) -> bool:
    """Return True if ``module`` can be imported, without importing it."""
    return importlib.util.find_spec(module) is not None  # type: ignore[attr-defined]
