"""Config objects and a single ``RunConfig`` that drives an end-to-end job.

The library is *config-first*: any recipe can be expressed as a nested dict /
YAML and rebuilt with :func:`load_config`.  Each axis of the
data → objective → algorithm → training picture has its own small dataclass,
and :class:`RunConfig` is just the composition.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional, Type, TypeVar, get_type_hints

T = TypeVar("T", bound="Config")


@dataclass
class Config:
    """Base dataclass with dict/YAML round-tripping and shallow merging."""

    def to_dict(self) -> Dict[str, Any]:
        return _asdict(self)

    @classmethod
    def from_dict(cls: Type[T], data: Optional[Dict[str, Any]]) -> T:
        data = dict(data or {})
        kwargs: Dict[str, Any] = {}
        valid = {f.name for f in fields(cls)}  # type: ignore[arg-type]
        # ``from __future__ import annotations`` makes f.type a *string*, so we
        # resolve real types once here to detect nested Config fields.
        hints = get_type_hints(cls)
        for key, value in data.items():
            if key not in valid:
                # tolerate unknown keys so configs stay forward-compatible
                continue
            ftype = hints.get(key)
            if is_dataclass(ftype) and isinstance(value, dict):
                kwargs[key] = ftype.from_dict(value)  # type: ignore[union-attr]
            else:
                kwargs[key] = value
        return cls(**kwargs)  # type: ignore[arg-type]

    def merge(self: T, **overrides: Any) -> T:
        return dataclasses.replace(self, **overrides)


@dataclass
class DataConfig(Config):
    """Where samples come from + how they're tokenised."""

    source: str = "jsonl"
    path: Optional[str] = None
    split: Optional[str] = None
    max_seq_len: int = 2048
    pack: bool = False
    shuffle: bool = True
    seed: int = 0
    template: str = "chatml"
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ObjectiveConfig(Config):
    """Which loss + its hyperparameters (beta, clip range, group size...)."""

    name: str = "sft"
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlgorithmConfig(Config):
    """How parameters update: full / lora / qlora + adapter hyperparameters."""

    name: str = "full"
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimConfig(Config):
    lr: float = 1e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    scheduler: str = "cosine"
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    grad_clip: float = 1.0


@dataclass
class TrainConfig(Config):
    """The training loop knobs."""

    epochs: float = 1.0
    max_steps: Optional[int] = None
    batch_size: int = 8
    grad_accum: int = 1
    eval_every: Optional[int] = None
    save_every: Optional[int] = None
    output_dir: str = "./out"
    bf16: bool = True
    seed: int = 0
    log_every: int = 10


@dataclass
class RLConfig(Config):
    """Rollout / verification knobs for RLVR & agentic RL."""

    group_size: int = 8           # samples per prompt (GRPO)
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 1024
    kl_coef: float = 0.0
    clip_range: float = 0.2
    reward: str = "verifier"      # registry key of the Reward
    verifier: Optional[str] = None
    max_env_steps: int = 32
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig(Config):
    """Either a HF checkpoint name (``pretrained``) or a from-scratch spec."""

    pretrained: Optional[str] = None
    arch: str = "decoder_lm"
    dtype: str = "bfloat16"
    attn_impl: str = "sdpa"
    trust_remote_code: bool = False
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunConfig(Config):
    """The full description of one training run — the library's universal IR."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    name: str = "trainall-run"


# --------------------------------------------------------------------------- #
# (De)serialisation helpers
# --------------------------------------------------------------------------- #
def _asdict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _asdict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [_asdict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _asdict(v) for k, v in obj.items()}
    return obj


def load_config(path_or_dict: Any) -> RunConfig:
    """Build a :class:`RunConfig` from a YAML/JSON path or a plain dict."""
    if isinstance(path_or_dict, RunConfig):
        return path_or_dict
    if isinstance(path_or_dict, dict):
        return RunConfig.from_dict(path_or_dict)
    if isinstance(path_or_dict, str):
        import json

        with open(path_or_dict, "r", encoding="utf-8") as fh:
            text = fh.read()
        if path_or_dict.endswith((".yaml", ".yml")):
            import yaml

            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        return RunConfig.from_dict(data)
    raise TypeError(f"cannot load config from {type(path_or_dict)!r}")


__all__ = [
    "Config",
    "DataConfig",
    "ObjectiveConfig",
    "AlgorithmConfig",
    "OptimConfig",
    "TrainConfig",
    "RLConfig",
    "ModelConfig",
    "RunConfig",
    "load_config",
]
