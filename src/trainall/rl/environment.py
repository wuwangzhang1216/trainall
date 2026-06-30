"""Multi-step, tool-using environments for agentic RL.

:class:`MultiStepEnv` is the base for environments where the policy reaches a
goal over several turns by calling tools.  It wires together:

* a :class:`~trainall.rl.tools.ToolRegistry` (the action space), and
* a *success check* — a callable (or a :class:`~trainall.base.Verifier`) that
  decides, from the running transcript, whether the task is solved.

An action is a string the agent emits.  If it parses as a tool call it is
dispatched and its output is fed back as the next observation; otherwise it is
treated as a final answer and checked for success.  The episode terminates on
success, on a final answer, or when the step budget runs out.

:class:`ExpressionEnv` is a tiny concrete env used in tests: given a target
number, the agent must reach it using the :class:`CalculatorTool` and submit
the value.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Tuple, Union

from ..base import Environment, Verifier
from ..registry import register
from ..types import Sample
from .tools import CalculatorTool, ToolRegistry

SuccessCheck = Union[Callable[..., Any], Verifier]


class MultiStepEnv(Environment):
    """Base class for tool-using, multi-step environments.

    Subclasses provide the task setup in :meth:`reset` (storing the goal /
    reference on ``self``) and may override :meth:`render_observation`.  The
    :meth:`step` loop here handles tool dispatch versus final-answer
    submission and the verifier-based success check; subclasses rarely need to
    touch it.
    """

    name = "multi_step_env"

    def __init__(
        self,
        tools: Optional[ToolRegistry] = None,
        success_check: Optional[SuccessCheck] = None,
        *,
        step_penalty: float = 0.0,
    ) -> None:
        self.tools = tools or ToolRegistry()
        self.success_check = success_check
        self.step_penalty = step_penalty
        self.reference: Any = None
        self.steps: int = 0
        self.done: bool = False

    # ------------------------------------------------------------------ #
    # Environment contract
    # ------------------------------------------------------------------ #
    def reset(self, sample: Optional[Sample] = None) -> Any:
        """Start a new episode; subclasses should set ``self.reference``."""
        self.reference = sample.reference if sample is not None else None
        self.steps = 0
        self.done = False
        return self.render_observation("")

    def step(self, action: Any) -> Tuple[Any, float, bool, dict]:
        """Dispatch a tool call or check a final answer.

        Returns ``(observation, reward, done, info)`` with ``info['success']``
        set.  A tool call yields the tool's output as the observation and a
        (negative) step penalty; a final answer ends the episode with reward 1
        on success else 0.
        """
        self.steps += 1
        text = str(action).strip()
        name, _ = ToolRegistry.parse(text)
        if name in self.tools:
            obs = self.tools.dispatch(text)
            info = {"success": False, "tool": name}
            return self.render_observation(obs), -self.step_penalty, False, info

        # Treat as a final answer.
        answer = self._extract_answer(text)
        success = self._check_success(answer)
        self.done = True
        reward = 1.0 if success else 0.0
        info = {"success": success, "answer": answer}
        return self.render_observation(text), reward, True, info

    # ------------------------------------------------------------------ #
    # Hooks for subclasses
    # ------------------------------------------------------------------ #
    def render_observation(self, payload: str) -> str:
        """Format what the agent sees next; default echoes the payload."""
        return payload

    def _extract_answer(self, text: str) -> str:
        """Pull the submitted answer out of a final-answer action."""
        lowered = text.lower()
        for marker in ("answer:", "final:", "submit:"):
            if marker in lowered:
                idx = lowered.index(marker) + len(marker)
                return text[idx:].strip()
        return text

    def _check_success(self, answer: str) -> bool:
        """Run the configured success check against ``answer``."""
        if self.success_check is None:
            return False
        if isinstance(self.success_check, Verifier):
            return bool(self.success_check.verify(answer, self.reference))
        return bool(self.success_check(answer, self.reference))


@register("expression_env", category="environment")
class ExpressionEnv(MultiStepEnv):
    """A tiny env: reach a target number using the calculator tool.

    ``reset`` takes a :class:`Sample` whose ``reference`` is the target value
    (and whose ``prompt`` may describe the task).  The agent computes with the
    ``calculator`` tool, then submits ``"answer: <value>"``.  Success is exact
    numeric match (within a small tolerance), giving a verifiable reward.
    """

    name = "expression_env"

    def __init__(self, *, tolerance: float = 1e-6, step_penalty: float = 0.0) -> None:
        super().__init__(
            tools=ToolRegistry([CalculatorTool()]),
            success_check=self._matches_target,
            step_penalty=step_penalty,
        )
        self.tolerance = tolerance
        self.target: Optional[float] = None
        self.prompt: str = ""

    def reset(self, sample: Optional[Sample] = None) -> Any:
        obs = super().reset(sample)
        self.target = float(self.reference) if self.reference is not None else None
        self.prompt = (sample.prompt if sample and sample.prompt else "") or ""
        header = self.prompt or f"Reach the value {self.target} using the calculator."
        return f"{header}\nTools: calculator. Submit with 'answer: <value>'."

    def _matches_target(self, answer: str, _reference: Any) -> bool:
        if self.target is None:
            return False
        try:
            value = float(answer)
        except (TypeError, ValueError):
            return False
        return abs(value - self.target) <= self.tolerance


__all__ = ["MultiStepEnv", "ExpressionEnv"]
