"""RL: group advantages, Rollout group_sample, MultiStepEnv rollout, tools."""
from __future__ import annotations

import pytest

from trainall.rl import (
    CalculatorTool,
    PythonTool,
    Rollout,
    RolloutConfig,
    ToolRegistry,
    compute_group_advantages,
)
from trainall.rl.environment import ExpressionEnv
from trainall.types import Sample, Trajectory


# --------------------------------------------------------------------------- #
# compute_group_advantages (exact math)
# --------------------------------------------------------------------------- #
def test_compute_group_advantages_exact():
    trajs = [
        Trajectory(prompt="a", response="", reward=1.0, group_id=0),
        Trajectory(prompt="a", response="", reward=0.0, group_id=0),
        Trajectory(prompt="b", response="", reward=2.0, group_id=1),
        Trajectory(prompt="b", response="", reward=2.0, group_id=1),
    ]
    out = compute_group_advantages(trajs)
    # group 0: mean 0.5, std 0.5 -> +1 / -1.
    assert out[0].advantage == pytest.approx(1.0, abs=1e-3)
    assert out[1].advantage == pytest.approx(-1.0, abs=1e-3)
    # group 1: zero variance -> advantage 0 (division by std+eps).
    assert out[2].advantage == pytest.approx(0.0, abs=1e-3)
    assert out[3].advantage == pytest.approx(0.0, abs=1e-3)


# --------------------------------------------------------------------------- #
# Rollout with a callable policy
# --------------------------------------------------------------------------- #
def test_rollout_callable_generate():
    rollout = Rollout(policy=lambda p: p.upper())
    trajs = rollout.generate(["hello", "world"])
    assert [t.response for t in trajs] == ["HELLO", "WORLD"]


def test_rollout_group_sample_group_ids():
    rollout = Rollout(policy=lambda p: f"resp-{p}", config=RolloutConfig(group_size=3))
    trajs = rollout.group_sample(["a", "b"])
    assert len(trajs) == 6
    # First 3 share group_id 0, next 3 group_id 1.
    assert [t.group_id for t in trajs] == [0, 0, 0, 1, 1, 1]
    assert all(t.response == f"resp-{t.prompt}" for t in trajs)


def test_rollout_group_sample_into_advantages():
    # Policy whose reward we set so group advantages are computable.
    rollout = Rollout(policy=lambda p: p, config=RolloutConfig(group_size=2))
    trajs = rollout.group_sample(["a"])
    trajs[0].reward = 1.0
    trajs[1].reward = 0.0
    compute_group_advantages(trajs)
    assert trajs[0].advantage == pytest.approx(1.0, abs=1e-3)


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
def test_calculator_tool_safe_eval():
    calc = CalculatorTool()
    assert calc.run("2 * (3 + 4)") == "14"
    # Names / calls are refused.
    assert calc.run("__import__('os')").startswith("error")


def test_python_tool_subprocess():
    tool = PythonTool(timeout=15.0)
    out = tool.run("print(6 * 7)")
    assert out == "42"


def test_tool_registry_dispatch():
    reg = ToolRegistry([CalculatorTool()])
    assert "calculator" in reg
    assert reg.dispatch("calculator: 1 + 1") == "2"
    assert reg.dispatch("unknown: x").startswith("error")


def test_toolregistry_parse():
    assert ToolRegistry.parse("calculator: 1+1") == ("calculator", "1+1")
    assert ToolRegistry.parse("python(print(9))") == ("python", "print(9)")


# --------------------------------------------------------------------------- #
# MultiStepEnv reaching success
# --------------------------------------------------------------------------- #
def test_expression_env_success():
    env = ExpressionEnv()
    sample = Sample(prompt="reach 7", reference=7.0)

    # Scripted policy: first turn use the calculator, then submit the answer.
    state = {"step": 0}

    def policy(_obs):
        state["step"] += 1
        if state["step"] == 1:
            return "calculator: 3 + 4"
        return "answer: 7"

    ep = env.rollout(policy, sample=sample, max_steps=5)
    assert ep.success is True
    assert ep.total_reward >= 1.0


def test_expression_env_failure():
    env = ExpressionEnv()
    sample = Sample(prompt="reach 7", reference=7.0)
    ep = env.rollout(lambda _obs: "answer: 99", sample=sample, max_steps=3)
    assert ep.success is False
