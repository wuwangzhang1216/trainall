"""trainall — one library for the frontier LLM training stack.

    data → objective → algorithm → training

Pick *where samples come from* (``trainall.data``), *what the model is rewarded
to become* (``trainall.objectives``), and *how parameters update*
(``trainall.algorithms``); a :class:`~trainall.training.Trainer` ties them
together, and :mod:`trainall.pipelines` chains the whole frontier recipe::

    CPT → SFT → synthetic/RS expand → DPO → RLVR(GRPO) → Agentic RL → Distill

Everything is reachable through one config-driven factory::

    import trainall
    obj  = trainall.build("dpo", beta=0.1)
    algo = trainall.build("qlora", r=16)
    print(trainall.available())            # see every registered key

``import trainall`` is deliberately cheap and works with no ML stack installed.
The heavy subpackages (``models``, ``objectives``, ``algorithms`` ...) are
imported lazily on first attribute access, and torch/transformers are pulled in
only when you actually train.
"""
from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.1.0"

# --- always-cheap core ----------------------------------------------------- #
from . import base, types  # noqa: E402  (pure-python, no ML stack)
from .base import (  # noqa: E402
    Algorithm,
    DataSource,
    Environment,
    Objective,
    Reward,
    Verifier,
)
from .config import (  # noqa: E402
    AlgorithmConfig,
    Config,
    DataConfig,
    ModelConfig,
    ObjectiveConfig,
    OptimConfig,
    RLConfig,
    RunConfig,
    TrainConfig,
    load_config,
)
from .registry import available, build, get, register  # noqa: E402
from .types import (  # noqa: E402
    Batch,
    Episode,
    Message,
    PreferenceSample,
    Sample,
    Trajectory,
    Transition,
    VerifierResult,
)
from .utils import get_logger, seed_everything  # noqa: E402

# Subpackages loaded on demand (some are torch-required).
_LAZY_SUBMODULES = {
    "models",
    "objectives",
    "algorithms",
    "verifiers",
    "rewards",
    "rl",
    "data",
    "training",
    "pipelines",
}


def __getattr__(name: str) -> Any:  # PEP 562 lazy submodule access
    if name in _LAZY_SUBMODULES:
        mod = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = mod
        return mod
    raise AttributeError(f"module 'trainall' has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_SUBMODULES))


def train(config: Any, **overrides: Any) -> Any:
    """Run one training job from a :class:`RunConfig`, dict, or YAML path.

    Thin convenience wrapper: build the objective/algorithm/data from the
    config and hand them to a :class:`~trainall.training.Trainer`.  Returns the
    trained model (or the :class:`~trainall.pipelines.StageResult` for a
    multi-stage pipeline config).
    """
    from .training import Trainer

    cfg = load_config(config)
    if overrides:
        cfg = cfg.merge(**overrides)
    return Trainer.from_config(cfg).train()


__all__ = [
    "__version__",
    # factory
    "build",
    "register",
    "get",
    "available",
    "train",
    # contracts
    "base",
    "types",
    "Objective",
    "Algorithm",
    "Verifier",
    "Reward",
    "DataSource",
    "Environment",
    # data types
    "Sample",
    "PreferenceSample",
    "Batch",
    "Message",
    "VerifierResult",
    "Trajectory",
    "Episode",
    "Transition",
    # config
    "Config",
    "RunConfig",
    "ModelConfig",
    "DataConfig",
    "ObjectiveConfig",
    "AlgorithmConfig",
    "OptimConfig",
    "TrainConfig",
    "RLConfig",
    "load_config",
    # utils
    "get_logger",
    "seed_everything",
]
