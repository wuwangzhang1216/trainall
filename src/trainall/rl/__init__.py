"""Rollout + environment machinery for RLVR and agentic RL.

* :class:`Rollout`         — sample generations from a policy; group sampling
                             (N per prompt) for GRPO, with old-logprob capture.
* :class:`MultiStepEnv`    — base class for tool-using / browsing / coding envs.
* :class:`Tool` + friends  — the action space for agentic environments.
* :class:`AgenticRunner`   — drive a policy through an env into scored
                             trajectories, with outcome + process rewards.

Rollout / runner orchestration is pure-python (model calls are injected), so
this imports without torch; actual generation needs a model passed in.
"""
from __future__ import annotations

from .rollout import Rollout, RolloutConfig, compute_group_advantages
from .environment import MultiStepEnv
from .tools import CalculatorTool, PythonTool, Tool, ToolRegistry
from .agentic import AgenticRunner

__all__ = [
    "Rollout",
    "RolloutConfig",
    "compute_group_advantages",
    "MultiStepEnv",
    "Tool",
    "PythonTool",
    "CalculatorTool",
    "ToolRegistry",
    "AgenticRunner",
]
