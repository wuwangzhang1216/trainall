"""Verifier-backed reward — the RLVR signal.

Wraps a deterministic :class:`~trainall.base.Verifier` so sampled
:class:`~trainall.types.Trajectory` objects can be scored with a verifiable,
side-effect-free check (math equality, unit tests, JSON validity, ...).  This is
the reward used by RLVR / GRPO with verifiable rewards (DeepSeek-R1, Lambert et
al. 2024 "Tulu 3", and the broader RLVR line of work).
"""
from __future__ import annotations

from typing import Any, List, Sequence

from ..base import Reward, Verifier
from ..registry import register
from ..types import Trajectory


@register("verifier", category="reward")
class VerifierReward(Reward):
    """Score trajectories by running a verifier on each response.

    The ``verifier`` may be passed either as a concrete :class:`Verifier`
    instance or as a registry key string (e.g. ``"math"``), in which case it is
    built lazily through :func:`trainall.registry.build`.  For every trajectory
    the verifier is called with ``response=traj.response``,
    ``reference=traj.meta.get("reference")`` and ``prompt=traj.prompt``; the
    resulting :class:`~trainall.types.VerifierResult` reward (in ``[0, 1]``) is
    returned as the trajectory's scalar reward.
    """

    def __init__(self, verifier: Any, **verifier_kwargs: Any) -> None:
        self.verifier = self._resolve(verifier, verifier_kwargs)

    @staticmethod
    def _resolve(verifier: Any, kwargs: dict) -> Verifier:
        if isinstance(verifier, Verifier):
            return verifier
        if isinstance(verifier, str):
            from ..registry import build

            built = build(verifier, category="verifier", **kwargs)
            if not isinstance(built, Verifier):  # pragma: no cover - defensive
                raise TypeError(
                    f"registry key {verifier!r} did not build a Verifier, got "
                    f"{type(built).__name__}"
                )
            return built
        if callable(verifier):
            # Accept a bare callable for tests: wrap it so .verify works.
            return _CallableVerifier(verifier)
        raise TypeError(
            "verifier must be a Verifier instance, a registry key string, or a "
            f"callable; got {type(verifier).__name__}"
        )

    def score(self, trajectories: Sequence[Trajectory]) -> List[float]:
        rewards: List[float] = []
        for traj in trajectories:
            result = self.verifier.verify(
                traj.response,
                traj.meta.get("reference"),
                prompt=traj.prompt,
            )
            rewards.append(float(result.reward))
        return rewards


class _CallableVerifier(Verifier):
    """Adapt a plain ``fn(response, reference, prompt=...) -> result`` callable."""

    def __init__(self, fn: Any) -> None:
        self._fn = fn

    def verify(self, response, reference=None, *, prompt=None, **kwargs):
        from ..types import VerifierResult

        try:
            out = self._fn(response, reference, prompt=prompt, **kwargs)
        except TypeError:
            # Tolerate simpler signatures, e.g. fn(response, reference).
            out = self._fn(response, reference)
        if isinstance(out, VerifierResult):
            return out
        r = float(out)
        return VerifierResult(reward=r, passed=r > 0.0)


__all__ = ["VerifierReward"]
