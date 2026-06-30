"""Rewards — adapt verifiers / reward models into RL training signals.

A :class:`trainall.base.Reward` scores sampled :class:`~trainall.types.Trajectory`
objects.  Three building blocks cover the landscape:

* :class:`VerifierReward`    — wrap a deterministic verifier (RLVR).
* :class:`RewardModelReward` — query a learned reward model (RLHF / RLAIF).
* :class:`ShapedReward`      — blend outcome + process + step penalties.
"""
from __future__ import annotations

from .verifier_reward import VerifierReward
from .model_reward import RewardModelReward
from .shaping import ShapedReward, StepPenalty, combine_rewards

__all__ = [
    "VerifierReward",
    "RewardModelReward",
    "ShapedReward",
    "StepPenalty",
    "combine_rewards",
]
