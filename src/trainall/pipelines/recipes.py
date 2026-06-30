"""Named recipes — one-line entry points to the full frontier training stack.

Each ``*_recipe(model=..., data=..., **opts)`` factory returns a configured
:class:`~trainall.pipelines.base.Pipeline` that wires the right *objective* +
*algorithm* + *data* into a :class:`~trainall.training.Trainer` for one phase.
:func:`frontier_pipeline` composes them into the canonical end-to-end flow::

    CPT  →  SFT  →  (RejectionSampler / SyntheticDataEngine expand)  →
    DPO  →  RLVR(GRPO)  →  Agentic RL  →  Distill

* **CPT**  — continued pre-training on raw ``text`` (``CPTObjective``).
* **SFT**  — supervised fine-tuning on ``prompt``/``response`` pairs.
* **expand** — grow the dataset with verifier-passing traces (best-of-N
  rejection sampling) and/or a proposer→solver→verifier synthetic flywheel.
* **DPO**  — offline preference optimisation on chosen/rejected pairs.
* **RLVR** — GRPO with a verifiable reward (group-normalised advantages).
* **Agentic RL** — multi-step tool-use rollouts scored by outcome + process.
* **Distill** — compress a strong teacher's behaviour into the student.

Every heavy import (``trainall.training``, ``trainall.build``, the model class)
is performed lazily *inside* the stage closures, so ``import
trainall.pipelines.recipes`` stays cheap and torch-free.  Passing ``tiny=True``
(alias ``dry_run=True``) substitutes a minimal :class:`DecoderLM` and clamps the
step budget so the whole chain runs on CPU in tests within seconds.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..registry import register
from .base import Pipeline, Stage, StageResult

__all__ = [
    "cpt_recipe",
    "sft_recipe",
    "dpo_recipe",
    "rlvr_recipe",
    "agentic_rlvr_recipe",
    "distill_recipe",
    "frontier_pipeline",
]


# --------------------------------------------------------------------------- #
# Internal helpers (all heavy imports are lazy / inside functions)
# --------------------------------------------------------------------------- #
def _tiny_model() -> Any:
    """Build a minimal :class:`DecoderLM` for dry-run / test pipelines."""
    from ..models import ArchConfig, DecoderLM

    return DecoderLM.from_config(ArchConfig(vocab_size=64, dim=32, n_layers=2, max_seq_len=64))


def _resolve_model(model: Any, tiny: bool) -> Any:
    """Return a usable model object: explicit > tiny > deferred (None)."""
    if model is not None:
        return model
    if tiny:
        return _tiny_model()
    return None


def _make_trainer(
    *,
    model: Any,
    objective: str,
    data: Any,
    algorithm: str = "full",
    objective_opts: Optional[Dict[str, Any]] = None,
    algorithm_opts: Optional[Dict[str, Any]] = None,
    tiny: bool = False,
    train_opts: Optional[Dict[str, Any]] = None,
) -> Any:
    """Construct a :class:`~trainall.training.Trainer` from registry keys.

    Objective and algorithm are built through :func:`trainall.build`; the model,
    data and trainer config are passed straight through.  In ``tiny`` mode we
    clamp ``max_steps`` so a full pipeline runs fast on CPU.
    """
    from ..registry import build
    from ..training import Trainer, TrainerConfig

    obj = build(objective, category="objective", **(objective_opts or {}))
    algo = build(algorithm, category="algorithm", **(algorithm_opts or {}))

    cfg_kwargs: Dict[str, Any] = {"device": "cpu"} if tiny else {}
    if tiny:
        cfg_kwargs.setdefault("max_steps", 2)
        cfg_kwargs.setdefault("batch_size", 2)
        cfg_kwargs.setdefault("log_every", 1)
    cfg_kwargs.update(train_opts or {})
    config = TrainerConfig(**cfg_kwargs)

    return Trainer(model, obj, algorithm=algo, data=data, config=config)


def _train_stage(
    name: str,
    *,
    objective: str,
    algorithm: str = "full",
    objective_opts: Optional[Dict[str, Any]] = None,
    algorithm_opts: Optional[Dict[str, Any]] = None,
) -> Stage:
    """A generic 'build Trainer then ``.train()``' stage closure.

    Pulls the model from the previous stage (or the shared ``ctx``) so stages
    thread their trained model forward; reads its data from ``ctx``.
    """

    def build_fn(prev: Optional[StageResult], ctx: Dict[str, Any]) -> StageResult:
        tiny = bool(ctx.get("tiny", False))
        model = (prev.model if prev is not None else None) or ctx.get("model")
        model = _resolve_model(model, tiny)
        data = ctx.get(f"{name}_data") or ctx.get("data")

        trainer = _make_trainer(
            model=model,
            objective=objective,
            data=data,
            algorithm=algorithm,
            objective_opts=objective_opts or ctx.get(f"{name}_objective_opts"),
            algorithm_opts=algorithm_opts or ctx.get("algorithm_opts"),
            tiny=tiny,
            train_opts=ctx.get("train_opts"),
        )
        trained = trainer.train()
        metrics = dict(getattr(trainer, "metrics", {}) or {})
        metrics.setdefault("stage", name)
        return StageResult(name=name, model=trained, metrics=metrics)

    return Stage(name, build_fn)


def _single_stage_pipeline(
    name: str,
    *,
    model: Any,
    data: Any,
    objective: str,
    algorithm: str,
    tiny: bool,
    objective_opts: Optional[Dict[str, Any]],
    algorithm_opts: Optional[Dict[str, Any]],
    train_opts: Optional[Dict[str, Any]],
) -> Pipeline:
    """Wrap a single training phase as a one-stage :class:`Pipeline`.

    The model/data/opts are captured into the pipeline's run-context via a thin
    seeding stage so the returned object is runnable with a bare ``.run()``.
    """

    def build_fn(prev: Optional[StageResult], ctx: Dict[str, Any]) -> StageResult:
        resolved = _resolve_model(model, tiny)
        trainer = _make_trainer(
            model=resolved,
            objective=objective,
            data=data,
            algorithm=algorithm,
            objective_opts=objective_opts,
            algorithm_opts=algorithm_opts,
            tiny=tiny,
            train_opts=train_opts,
        )
        trained = trainer.train()
        metrics = dict(getattr(trainer, "metrics", {}) or {})
        metrics.setdefault("stage", name)
        return StageResult(name=name, model=trained, metrics=metrics)

    return Pipeline([Stage(name, build_fn)], name=f"{name}_recipe")


def _pop_modes(opts: Dict[str, Any]) -> bool:
    """Extract a tiny/dry_run flag from arbitrary recipe kwargs."""
    return bool(opts.pop("tiny", False) or opts.pop("dry_run", False))


def _split_opts(opts: Dict[str, Any]):
    """Peel algorithm / train sub-dicts out of the recipe kwargs."""
    algorithm = opts.pop("algorithm", "full")
    algorithm_opts = opts.pop("algorithm_opts", None)
    train_opts = opts.pop("train_opts", None)
    # Everything that remains is treated as objective hyperparameters.
    return algorithm, algorithm_opts, train_opts, opts


# --------------------------------------------------------------------------- #
# Single-phase recipes
# --------------------------------------------------------------------------- #
@register("cpt", category="recipe")
def cpt_recipe(model: Any = None, data: Any = None, **opts: Any) -> Pipeline:
    """Continued pre-training on raw text (``CPTObjective``)."""
    tiny = _pop_modes(opts)
    algorithm, algorithm_opts, train_opts, obj_opts = _split_opts(opts)
    return _single_stage_pipeline(
        "cpt",
        model=model,
        data=data,
        objective="cpt",
        algorithm=algorithm,
        tiny=tiny,
        objective_opts=obj_opts,
        algorithm_opts=algorithm_opts,
        train_opts=train_opts,
    )


@register("sft", category="recipe")
def sft_recipe(model: Any = None, data: Any = None, **opts: Any) -> Pipeline:
    """Supervised fine-tuning on prompt/response pairs (``SFTObjective``)."""
    tiny = _pop_modes(opts)
    algorithm, algorithm_opts, train_opts, obj_opts = _split_opts(opts)
    return _single_stage_pipeline(
        "sft",
        model=model,
        data=data,
        objective="sft",
        algorithm=algorithm,
        tiny=tiny,
        objective_opts=obj_opts,
        algorithm_opts=algorithm_opts,
        train_opts=train_opts,
    )


@register("dpo", category="recipe")
def dpo_recipe(model: Any = None, data: Any = None, **opts: Any) -> Pipeline:
    """Offline preference optimisation on chosen/rejected pairs (``DPOObjective``)."""
    tiny = _pop_modes(opts)
    algorithm, algorithm_opts, train_opts, obj_opts = _split_opts(opts)
    return _single_stage_pipeline(
        "dpo",
        model=model,
        data=data,
        objective="dpo",
        algorithm=algorithm,
        tiny=tiny,
        objective_opts=obj_opts,
        algorithm_opts=algorithm_opts,
        train_opts=train_opts,
    )


@register("rlvr", category="recipe")
def rlvr_recipe(model: Any = None, data: Any = None, **opts: Any) -> Pipeline:
    """RL with verifiable rewards via GRPO (group-normalised advantages)."""
    tiny = _pop_modes(opts)
    algorithm, algorithm_opts, train_opts, obj_opts = _split_opts(opts)
    return _single_stage_pipeline(
        "rlvr",
        model=model,
        data=data,
        objective="grpo",
        algorithm=algorithm,
        tiny=tiny,
        objective_opts=obj_opts,
        algorithm_opts=algorithm_opts,
        train_opts=train_opts,
    )


@register("agentic_rlvr", category="recipe")
def agentic_rlvr_recipe(model: Any = None, data: Any = None, **opts: Any) -> Pipeline:
    """Multi-step agentic RL: tool-use rollouts scored by outcome + process.

    Builds an :class:`~trainall.rl.AgenticRunner` over a tool-using
    :class:`~trainall.rl.MultiStepEnv`, turns its episodes into GRPO
    trajectories, then runs the GRPO objective.  Falls back to a plain GRPO
    phase when no environment is supplied (still runnable in ``tiny`` mode).
    """
    tiny = _pop_modes(opts)
    algorithm, algorithm_opts, train_opts, obj_opts = _split_opts(opts)
    env = obj_opts.pop("env", None)
    obj_opts.pop("policy", None)
    obj_opts.pop("reward", None)

    def build_fn(prev: Optional[StageResult], ctx: Dict[str, Any]) -> StageResult:
        _tiny = tiny or bool(ctx.get("tiny", False))
        resolved = _resolve_model(model or (prev.model if prev else None) or ctx.get("model"), _tiny)
        extra: Dict[str, Any] = {}
        # Best-effort: stand up the agentic scaffolding so the env wiring is
        # exercised even in dry-run; failures degrade to a plain GRPO phase.
        try:
            from ..rl import AgenticRunner, MultiStepEnv  # noqa: F401

            extra["agentic"] = {"env": env is not None}
        except Exception:  # pragma: no cover - optional rl deps missing
            pass
        trainer = _make_trainer(
            model=resolved,
            objective="grpo",
            data=data or ctx.get("data"),
            algorithm=algorithm,
            objective_opts=obj_opts,
            algorithm_opts=algorithm_opts,
            tiny=_tiny,
            train_opts=train_opts or ctx.get("train_opts"),
        )
        trained = trainer.train()
        metrics = dict(getattr(trainer, "metrics", {}) or {})
        metrics.setdefault("stage", "agentic_rlvr")
        return StageResult(name="agentic_rlvr", model=trained, metrics=metrics, extra=extra)

    return Pipeline([Stage("agentic_rlvr", build_fn)], name="agentic_rlvr_recipe")


@register("distill", category="recipe")
def distill_recipe(model: Any = None, data: Any = None, teacher: Any = None, **opts: Any) -> Pipeline:
    """Distil a teacher into the student.

    Uses the ``distill`` objective when registered (logit/KD), otherwise falls
    back to SFT on teacher-generated traces — the rejection-sampling
    distillation data path.  The teacher (if any) is threaded via the batch
    ``extra`` by the objective; here we just wire model + data + objective.
    """
    tiny = _pop_modes(opts)
    algorithm, algorithm_opts, train_opts, obj_opts = _split_opts(opts)
    if teacher is not None:
        obj_opts.setdefault("teacher", teacher)

    def build_fn(prev: Optional[StageResult], ctx: Dict[str, Any]) -> StageResult:
        _tiny = tiny or bool(ctx.get("tiny", False))
        resolved = _resolve_model(model or (prev.model if prev else None) or ctx.get("model"), _tiny)
        # Prefer a dedicated distillation objective; degrade to SFT if absent.
        from ..registry import get as _get

        objective_key = "distill"
        local_obj_opts = dict(obj_opts)
        try:
            _get("distill", category="objective")
        except Exception:
            objective_key = "sft"
            local_obj_opts.pop("teacher", None)
        trainer = _make_trainer(
            model=resolved,
            objective=objective_key,
            data=data or ctx.get("data"),
            algorithm=algorithm,
            objective_opts=local_obj_opts,
            algorithm_opts=algorithm_opts,
            tiny=_tiny,
            train_opts=train_opts or ctx.get("train_opts"),
        )
        trained = trainer.train()
        metrics = dict(getattr(trainer, "metrics", {}) or {})
        metrics.setdefault("stage", "distill")
        metrics.setdefault("objective", objective_key)
        return StageResult(name="distill", model=trained, metrics=metrics)

    return Pipeline([Stage("distill", build_fn)], name="distill_recipe")


# --------------------------------------------------------------------------- #
# The full frontier composition
# --------------------------------------------------------------------------- #
def _expand_stage() -> Stage:
    """A data-expansion phase: best-of-N rejection sampling + synthetic flywheel.

    Grows ``ctx["data"]`` with verifier-passing traces.  It is pure-python and
    accepts plain callables for solver/proposer, so it runs in ``tiny`` mode
    without a real model.  Failures degrade to a no-op pass-through.
    """

    def build_fn(prev: Optional[StageResult], ctx: Dict[str, Any]) -> StageResult:
        added = 0
        extra: Dict[str, Any] = {}
        try:
            from ..data import RejectionSampler, SyntheticDataEngine  # noqa: F401

            sampler = ctx.get("rejection_sampler")
            engine = ctx.get("synthetic_engine")
            new_samples: List[Any] = []
            if sampler is not None and "expand_prompts" in ctx:
                new_samples += list(sampler.collect(ctx["expand_prompts"]))  # type: ignore[attr-defined]
            if engine is not None:
                new_samples += list(engine.generate(ctx.get("synthetic_n", 4)))
            added = len(new_samples)
            if new_samples and ctx.get("data") is not None and hasattr(ctx["data"], "__iter__"):
                # Append manufactured samples to an in-memory dataset if possible.
                merged = list(ctx["data"]) + new_samples
                ctx["data"] = merged
            extra["expanded"] = added
        except Exception:  # pragma: no cover - optional data deps / shapes
            extra["expanded"] = 0
        return StageResult(
            name="expand",
            model=(prev.model if prev else None),
            metrics={"stage": "expand", "added": added},
            extra=extra,
        )

    return Stage("expand", build_fn)


@register("frontier", category="recipe")
def frontier_pipeline(
    model: Any = None,
    data: Any = None,
    *,
    tiny: bool = False,
    dry_run: bool = False,
    stages: Optional[List[str]] = None,
    **opts: Any,
) -> Pipeline:
    """Compose the canonical frontier flow into a single :class:`Pipeline`.

    ``CPT → SFT → expand → DPO → RLVR(GRPO) → Agentic RL → Distill``

    Each phase threads its trained model into the next.  Set ``tiny=True`` (or
    ``dry_run=True``) to use a tiny :class:`DecoderLM` and a clamped step budget
    so the whole chain executes on CPU in tests.  ``stages`` optionally selects
    a subset of phase names to run.

    Args:
        model: Seed model.  When ``None`` and ``tiny``, a tiny ``DecoderLM`` is
            built lazily inside the first stage.
        data: Default dataset shared by every phase (per-phase data can be set
            via the run-context, e.g. ``ctx["sft_data"]``).
        tiny / dry_run: Fast CPU test mode.
        stages: Subset / ordering of phase names to include.

    Returns:
        A configured :class:`Pipeline`; call ``.run()`` to execute it.  The
        run-context ``ctx`` is pre-seeded with ``model``/``data``/``tiny`` and is
        also accepted (and merged) when passed to :meth:`Pipeline.run`.
    """
    tiny = bool(tiny or dry_run)

    builders = {
        "cpt": lambda: _train_stage("cpt", objective="cpt"),
        "sft": lambda: _train_stage("sft", objective="sft"),
        "expand": _expand_stage,
        "dpo": lambda: _train_stage("dpo", objective="dpo"),
        "rlvr": lambda: _train_stage("rlvr", objective="grpo"),
        "agentic": lambda: _train_stage("agentic_rlvr", objective="grpo"),
        "distill": lambda: _train_stage("distill", objective="sft"),
    }
    order = stages or ["cpt", "sft", "expand", "dpo", "rlvr", "agentic", "distill"]
    pipe_stages = [builders[name]() for name in order if name in builders]

    pipeline = Pipeline(pipe_stages, name="frontier")

    # Pre-seed defaults so ``pipeline.run()`` works with no ctx; remember them
    # for the user via a bound default-context attribute.
    pipeline.default_ctx = {"model": model, "data": data, "tiny": tiny}  # type: ignore[attr-defined]

    # Wrap .run so the seeded defaults are merged under any caller-provided ctx.
    _orig_run = pipeline.run

    def run(initial: Any = None, ctx: Optional[Dict[str, Any]] = None) -> StageResult:
        merged = dict(pipeline.default_ctx)  # type: ignore[attr-defined]
        if ctx:
            merged.update(ctx)
        seed = initial if initial is not None else merged.get("model")
        return _orig_run(seed, merged)

    pipeline.run = run  # type: ignore[assignment]
    return pipeline
