"""The pipeline DSL — chain named training phases into one frontier run.

A :class:`Pipeline` is an ordered list of :class:`Stage` objects.  Each stage is
a *named phase* (CPT, SFT, DPO, RLVR, ...) whose ``build_fn`` consumes the
previous stage's :class:`StageResult` and returns its own.  :meth:`Pipeline.run`
threads the trained model from one stage into the next, accumulating per-stage
metrics, so the canonical frontier flow

    CPT → SFT → expand → DPO → RLVR(GRPO) → Agentic RL → Distill

is just a list of stages executed in order.

Everything here is pure-python and torch-free: the heavy lifting (building a
``Trainer``, running it) happens inside the stage ``build_fn`` closures, which
import :mod:`trainall.training` lazily.  That keeps ``import trainall.pipelines``
cheap and dependency-free.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

__all__ = ["StageResult", "Stage", "Pipeline"]


# --------------------------------------------------------------------------- #
# Stage result — what one phase hands to the next
# --------------------------------------------------------------------------- #
@dataclass
class StageResult:
    """The output of a single :class:`Stage`.

    Attributes:
        name: The stage that produced this result.
        model: The trained model, threaded into the next stage.
        metrics: Scalar logging metrics for this stage.
        extra: Free-form carry-over (expanded datasets, configs, trajectories).
    """

    name: str
    model: Any = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Stage — one named training phase
# --------------------------------------------------------------------------- #
# A build_fn takes the previous StageResult (or None for the first stage) and a
# shared mutable context dict, and returns a StageResult.
BuildFn = Callable[[Optional[StageResult], Dict[str, Any]], StageResult]


@dataclass
class Stage:
    """A single named phase of a :class:`Pipeline`.

    The ``build_fn`` receives ``(prev_result, ctx)`` where ``prev_result`` is the
    previous stage's :class:`StageResult` (``None`` for the first stage) and
    ``ctx`` is a shared dict the pipeline threads through every stage.  It is
    expected to construct and run whatever it needs (typically a
    :class:`~trainall.training.Trainer`) and return a :class:`StageResult`.
    """

    name: str
    build_fn: BuildFn

    def run(self, prev: Optional[StageResult], ctx: Dict[str, Any]) -> StageResult:
        """Execute this stage, coercing loose return values into a StageResult."""
        out = self.build_fn(prev, ctx)
        if isinstance(out, StageResult):
            if not out.name:
                out.name = self.name
            return out
        # Tolerate a build_fn that returns just a model.
        return StageResult(name=self.name, model=out)


# --------------------------------------------------------------------------- #
# Pipeline — an ordered list of stages
# --------------------------------------------------------------------------- #
class Pipeline:
    """An ordered sequence of :class:`Stage` phases run end-to-end.

    :meth:`run` threads each stage's output model into the next stage's input,
    collects per-stage metrics under :attr:`StageResult.extra` of the final
    result, and logs progress via :func:`trainall.utils.get_logger`.
    """

    def __init__(self, stages: List[Stage], name: str = "pipeline") -> None:
        self.stages = list(stages)
        self.name = name

    def __len__(self) -> int:
        return len(self.stages)

    def __iter__(self):
        return iter(self.stages)

    def run(
        self,
        initial: Optional[Any] = None,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> StageResult:
        """Run every stage in order, threading the model through.

        Args:
            initial: Optional seed model (or seed :class:`StageResult`) fed to
                the first stage as its ``prev_result``.
            ctx: Optional shared context dict made available to every stage and
                carried (mutated) across the whole run.

        Returns:
            The final :class:`StageResult`.  Its ``extra["history"]`` holds the
            ``{stage_name: metrics}`` map for every phase that ran.
        """
        from ..utils import get_logger

        log = get_logger(f"pipeline.{self.name}")
        ctx = ctx if ctx is not None else {}

        if isinstance(initial, StageResult):
            prev: Optional[StageResult] = initial
        elif initial is not None:
            prev = StageResult(name="__initial__", model=initial)
        else:
            prev = None

        history: Dict[str, Dict[str, Any]] = {}
        if prev is None:
            log.info("starting pipeline %r with %d stage(s)", self.name, len(self.stages))
        else:
            log.info(
                "starting pipeline %r with %d stage(s) from seed model",
                self.name,
                len(self.stages),
            )

        result = prev
        for i, stage in enumerate(self.stages):
            log.info("[%d/%d] stage %r", i + 1, len(self.stages), stage.name)
            result = stage.run(prev, ctx)
            history[stage.name] = dict(result.metrics)
            log.info("[%d/%d] stage %r done | metrics=%s", i + 1, len(self.stages), stage.name, result.metrics)
            prev = result

        if result is None:
            result = StageResult(name=self.name)
        result.extra.setdefault("history", history)
        log.info("pipeline %r finished", self.name)
        return result
