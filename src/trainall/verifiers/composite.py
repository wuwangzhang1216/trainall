"""Composite verifier.

Combines several sub-verifiers into one signal.  ``"weighted"`` returns the
weighted mean of sub-rewards (with ``passed`` = all sub-verifiers passed),
``"all"`` requires every component to pass, and ``"any"`` requires at least one.
This is how multi-faceted RLVR rewards are assembled — e.g. format reward AND
answer-correctness reward (DeepSeek-R1, Guo et al. 2025).
"""
from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence, Tuple

from ..base import Verifier
from ..registry import register
from ..types import VerifierResult

__all__ = ["CompositeVerifier"]

_MODES = ("weighted", "all", "any")


@register("composite", category="verifier")
class CompositeVerifier(Verifier):
    """Aggregate several verifiers under one ``verify`` call.

    Parameters
    ----------
    components:
        Either a list of ``(verifier, weight)`` pairs (bare verifiers default to
        weight 1.0) or a mapping ``{name: verifier}`` / ``{name: (verifier, w)}``.
    mode:
        ``"weighted"`` (default), ``"all"`` or ``"any"`` — see module docstring.
    """

    name = "composite"

    def __init__(self, components: Any, mode: str = "weighted") -> None:
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        self.mode = mode
        self._components: List[Tuple[str, Verifier, float]] = []
        self._ingest(components)

    def _ingest(self, components: Any) -> None:
        items: Sequence[Tuple[str, Any]]
        if isinstance(components, Mapping):
            items = list(components.items())
        else:
            items = [(f"v{i}", c) for i, c in enumerate(components)]
        for name, spec in items:
            if isinstance(spec, (list, tuple)) and len(spec) == 2:
                verifier, weight = spec[0], float(spec[1])
            else:
                verifier, weight = spec, 1.0
            self._components.append((str(name), verifier, weight))
        if not self._components:
            raise ValueError("CompositeVerifier needs at least one component")

    def verify(
        self,
        response: str,
        reference: Any = None,
        *,
        prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> VerifierResult:
        sub_results: List[Tuple[str, float, VerifierResult]] = []
        for nm, verifier, weight in self._components:
            res = verifier.verify(response, reference, prompt=prompt, **kwargs)
            sub_results.append((nm, weight, res))

        passes = [bool(r.passed) for _, _, r in sub_results]
        if self.mode == "all":
            passed = all(passes)
        elif self.mode == "any":
            passed = any(passes)
        else:  # weighted
            passed = all(passes)

        total_w = sum(w for _, w, _ in sub_results) or 1.0
        reward = sum(w * float(r.reward) for _, w, r in sub_results) / total_w

        parts = [
            f"{nm}(w={w:g}): reward={r.reward:.3f} passed={r.passed}"
            for nm, w, r in sub_results
        ]
        detail = f"mode={self.mode}; " + " | ".join(parts)
        return VerifierResult(
            reward=reward,
            passed=passed,
            detail=detail,
            meta={"components": {nm: float(r.reward) for nm, _, r in sub_results}},
        )
