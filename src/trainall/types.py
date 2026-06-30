"""Plain data contracts shared across the whole library.

These are the *nouns* of ``trainall``.  They are intentionally pure-python
dataclasses with no ML-stack dependency so that every layer — verifiers, data
sources, RL environments, pipelines — can speak the same vocabulary without
pulling in torch.

The three-axis mental model the library is built around:

* **Data** decides *what* the model learns           -> :class:`Sample` & friends
* **Objective** decides *what it is rewarded to be*   -> consumes a :class:`Batch`
* **Algorithm** decides *how parameters update*       -> wraps the model

Everything below is a transport format between those axes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence


# --------------------------------------------------------------------------- #
# Conversation primitives
# --------------------------------------------------------------------------- #
@dataclass
class Message:
    """A single turn in a chat conversation."""

    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    name: Optional[str] = None
    # Free-form metadata (tool call ids, citations, ...).
    meta: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Supervised / unsupervised samples
# --------------------------------------------------------------------------- #
@dataclass
class Sample:
    """A generic training example.

    Different objectives read different fields:

    * pretrain / CPT       -> ``text``
    * SFT                  -> ``prompt`` + ``response`` (or ``messages``)
    * reward / distill     -> ``prompt`` + ``response`` + ``meta``
    """

    text: Optional[str] = None
    prompt: Optional[str] = None
    response: Optional[str] = None
    messages: Optional[List[Message]] = None
    # Ground-truth answer / unit tests / schema used by verifiers.
    reference: Optional[Any] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreferenceSample:
    """A ``chosen`` vs ``rejected`` pair for offline preference optimisation.

    Used by DPO / IPO / KTO / ORPO / SimPO / CPO.  ``label`` is for the
    unpaired KTO case (a single response tagged desirable / undesirable).
    """

    prompt: str
    chosen: Optional[str] = None
    rejected: Optional[str] = None
    # KTO-style single-sided signal: True == desirable, False == undesirable.
    label: Optional[bool] = None
    margin: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Tensor batch contract
# --------------------------------------------------------------------------- #
@dataclass
class Batch:
    """A tokenised, collated batch handed to :meth:`Objective.compute_loss`.

    This is a *thin* typed wrapper over a dict of tensors.  Objectives should
    read the fields they need and ignore the rest.  Conventions:

    Causal LM / SFT
        ``input_ids``, ``attention_mask``, ``labels`` (``-100`` == ignore).
    Preference (DPO & friends)
        ``chosen_input_ids``, ``chosen_labels``,
        ``rejected_input_ids``, ``rejected_labels`` and, when available,
        ``ref_chosen_logps`` / ``ref_rejected_logps`` (precomputed reference
        log-probs).  When the reference logps are absent the objective is
        expected to obtain them from a frozen reference model.
    Policy-gradient (PPO / GRPO / RLOO)
        ``input_ids``, ``response_mask``, ``old_logps``, ``advantages`` /
        ``rewards``, optional ``group_ids`` for GRPO grouping.

    The raw payload lives in :attr:`tensors`; named properties are convenience
    accessors.  ``extra`` carries anything objective-specific.
    """

    tensors: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.tensors[key]

    def __contains__(self, key: str) -> bool:
        return key in self.tensors

    def get(self, key: str, default: Any = None) -> Any:
        return self.tensors.get(key, default)

    def keys(self):
        return self.tensors.keys()

    def to(self, device: Any) -> "Batch":
        """Move every tensor that supports ``.to`` onto ``device`` in place."""
        for k, v in self.tensors.items():
            if hasattr(v, "to"):
                self.tensors[k] = v.to(device)
        return self

    @classmethod
    def of(cls, **tensors: Any) -> "Batch":
        return cls(tensors=dict(tensors))


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
@dataclass
class VerifierResult:
    """Outcome of checking a model response against a reference.

    ``reward`` is the canonical scalar in ``[0, 1]`` consumed by RLVR.
    ``passed`` is the boolean view used by rejection sampling / pass@k.
    ``detail`` carries human-readable diagnostics (test output, parse error).
    """

    reward: float
    passed: bool
    detail: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def __float__(self) -> float:
        return float(self.reward)

    def __bool__(self) -> bool:
        return bool(self.passed)

    @classmethod
    def ok(cls, reward: float = 1.0, detail: str = "") -> "VerifierResult":
        return cls(reward=reward, passed=True, detail=detail)

    @classmethod
    def fail(cls, reward: float = 0.0, detail: str = "") -> "VerifierResult":
        return cls(reward=reward, passed=False, detail=detail)


# --------------------------------------------------------------------------- #
# RL / agentic primitives
# --------------------------------------------------------------------------- #
@dataclass
class Transition:
    """One ``observe -> act -> reward`` step inside an episode."""

    observation: Any
    action: Any
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Episode:
    """A full multi-step rollout produced by an :class:`Environment`."""

    transitions: List[Transition] = field(default_factory=list)
    total_reward: float = 0.0
    success: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)

    def add(self, t: Transition) -> None:
        self.transitions.append(t)
        self.total_reward += t.reward

    def __len__(self) -> int:
        return len(self.transitions)


@dataclass
class Trajectory:
    """A single sampled generation plus its bookkeeping for policy-gradient RL.

    Produced by the rollout layer, scored by a reward, then collated into a
    :class:`Batch` for GRPO / PPO / RLOO.
    """

    prompt: str
    response: str
    token_ids: Optional[Sequence[int]] = None
    logprobs: Optional[Sequence[float]] = None
    reward: float = 0.0
    # Group id ties trajectories sampled from the same prompt (GRPO).
    group_id: Optional[Any] = None
    advantage: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)


__all__ = [
    "Message",
    "Sample",
    "PreferenceSample",
    "Batch",
    "VerifierResult",
    "Transition",
    "Episode",
    "Trajectory",
]
