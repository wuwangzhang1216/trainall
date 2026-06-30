"""Rewards: VerifierReward, ShapedReward + StepPenalty, combine_rewards."""
from __future__ import annotations

import pytest

import trainall
from trainall.rewards import ShapedReward, StepPenalty, VerifierReward, combine_rewards
from trainall.types import Trajectory


def test_verifier_reward_from_key_math():
    rw = VerifierReward("math")
    trajs = [
        Trajectory(prompt="q", response=r"\boxed{42}", meta={"reference": "42"}),
        Trajectory(prompt="q", response=r"\boxed{0}", meta={"reference": "42"}),
    ]
    scores = rw.score(trajs)
    assert scores == [1.0, 0.0]


def test_verifier_reward_from_instance():
    v = trainall.build("regex", category="verifier", pattern=r"yes")
    rw = VerifierReward(v)
    trajs = [Trajectory(prompt="", response="yes"), Trajectory(prompt="", response="no")]
    scores = rw.score(trajs)
    assert scores[0] == 1.0 and scores[1] == 0.0


def test_verifier_reward_from_callable():
    rw = VerifierReward(lambda response, reference, **kw: 1.0 if "ok" in response else 0.0)
    trajs = [Trajectory(prompt="", response="ok"), Trajectory(prompt="", response="bad")]
    assert rw.score(trajs) == [1.0, 0.0]


def test_combine_rewards_weighted():
    out = combine_rewards([[1.0, 2.0], [10.0, 20.0]], weights=[1.0, 0.5])
    assert out == [pytest.approx(6.0), pytest.approx(12.0)]


def test_combine_rewards_length_mismatch():
    with pytest.raises(ValueError):
        combine_rewards([[1.0, 2.0], [3.0]])


def test_step_penalty():
    pen = StepPenalty(per_step=-0.1, field="num_steps")
    trajs = [Trajectory(prompt="", response="", meta={"num_steps": 3})]
    assert pen.score(trajs) == [pytest.approx(-0.3)]


def test_shaped_reward_blend_and_clip():
    base = VerifierReward(lambda r, ref, **kw: 1.0)
    pen = StepPenalty(per_step=-0.1, field="num_steps")
    shaped = ShapedReward(
        components=[(base, 1.0)],
        penalties=[pen],
        clip_min=0.0,
        clip_max=1.0,
    )
    trajs = [
        Trajectory(prompt="", response="x", meta={"num_steps": 2}),
        Trajectory(prompt="", response="x", meta={"num_steps": 100}),
    ]
    scores = shaped.score(trajs)
    assert scores[0] == pytest.approx(0.8)
    assert scores[1] == 0.0  # clipped from 1 - 10 = -9 to 0


def test_shaped_reward_registered():
    # ``shaped`` is registered; building requires components so just check key.
    assert "shaped" in trainall.available("reward")["reward"]
