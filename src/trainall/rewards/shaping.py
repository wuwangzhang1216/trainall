"""Reward shaping — blend outcome, process and penalty signals.

Real RL training rarely uses a single reward.  Process supervision (Lightman et
al. 2023 "Let's Verify Step by Step") and reward shaping (Ng et al. 1999) both
combine multiple signals: an outcome reward (did the final answer verify), one
or more process rewards (intermediate quality), and penalties (per-step / length
costs that discourage rambling).  The helpers here express that blend.

* :func:`combine_rewards` — elementwise weighted sum of several reward lists.
* :class:`StepPenalty`     — a :class:`Reward` that penalises step count.
* :class:`ShapedReward`    — compose components + penalties with optional clamp.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from ..base import Reward
from ..registry import register
from ..types import Trajectory


def combine_rewards(
    reward_lists: Sequence[Sequence[float]],
    weights: Optional[Sequence[float]] = None,
) -> List[float]:
    """Elementwise weighted sum of equal-length reward lists.

    ``reward_lists`` is a list of per-trajectory reward vectors (one vector per
    component).  ``weights`` defaults to all-ones.  Returns one combined reward
    per trajectory.
    """
    lists = [list(r) for r in reward_lists]
    if not lists:
        return []
    n = len(lists[0])
    for i, r in enumerate(lists):
        if len(r) != n:
            raise ValueError(
                f"reward_lists[{i}] has length {len(r)} but expected {n}; all "
                "component reward lists must align with the same trajectories."
            )
    if weights is None:
        weights = [1.0] * len(lists)
    if len(weights) != len(lists):
        raise ValueError(
            f"got {len(weights)} weights for {len(lists)} reward lists."
        )
    return [
        sum(w * lists[c][i] for c, w in enumerate(weights))
        for i in range(n)
    ]


@register("shaped", category="reward")
class ShapedReward(Reward):
    """Blend outcome + process rewards and subtract penalties.

    Parameters
    ----------
    components:
        List of ``(Reward, weight)`` pairs.  Each reward scores every trajectory;
        their weighted sum is the base shaped reward.
    penalties:
        List of :class:`Reward` (typically :class:`StepPenalty`) whose scores are
        *added* (they should already be negative) after the weighted components.
        May also be ``(Reward, weight)`` pairs.
    clip_min / clip_max:
        Optional bounds; when set, each final reward is clamped into the range.
    """

    def __init__(
        self,
        components: Sequence[Tuple[Reward, float]],
        penalties: Optional[Sequence[object]] = None,
        *,
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None,
    ) -> None:
        if not components:
            raise ValueError("ShapedReward needs at least one (reward, weight) component.")
        self.components: List[Tuple[Reward, float]] = [
            (rw, float(w)) for rw, w in components
        ]
        self.penalties: List[Tuple[Reward, float]] = [
            self._as_pair(p) for p in (penalties or [])
        ]
        self.clip_min = clip_min
        self.clip_max = clip_max

    @staticmethod
    def _as_pair(p: object) -> Tuple[Reward, float]:
        if isinstance(p, tuple):
            reward, weight = p
            return reward, float(weight)
        return p, 1.0  # type: ignore[return-value]

    def score(self, trajectories: Sequence[Trajectory]) -> List[float]:
        n = len(trajectories)
        reward_lists: List[List[float]] = []
        weights: List[float] = []
        for reward, weight in self.components:
            reward_lists.append([float(x) for x in reward.score(trajectories)])
            weights.append(weight)
        for reward, weight in self.penalties:
            reward_lists.append([float(x) for x in reward.score(trajectories)])
            weights.append(weight)
        combined = combine_rewards(reward_lists, weights) if reward_lists else [0.0] * n
        if self.clip_min is not None or self.clip_max is not None:
            lo = self.clip_min if self.clip_min is not None else float("-inf")
            hi = self.clip_max if self.clip_max is not None else float("inf")
            combined = [min(max(x, lo), hi) for x in combined]
        return combined


class StepPenalty(Reward):
    """Penalise the number of steps taken, read from ``traj.meta[field]``.

    A simple cost-of-effort shaping term: each step incurs ``per_step`` reward
    (negative by default), encouraging shorter trajectories.  Missing metadata
    counts as zero steps.
    """

    def __init__(self, per_step: float = -0.01, field: str = "n_steps") -> None:
        self.per_step = float(per_step)
        self.field = field

    def score(self, trajectories: Sequence[Trajectory]) -> List[float]:
        out: List[float] = []
        for traj in trajectories:
            steps = traj.meta.get(self.field, 0) or 0
            out.append(self.per_step * float(steps))
        return out


__all__ = ["ShapedReward", "StepPenalty", "combine_rewards"]
