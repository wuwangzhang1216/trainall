"""Abstract contracts — the *verbs* of the library.

Every concrete piece in ``trainall`` implements one of these tiny interfaces.
Keeping them in one place (and free of any ML-stack import) is what lets the
pieces compose: a pipeline can wire any :class:`DataSource` into any
:class:`Objective` under any :class:`Algorithm`, scored by any :class:`Reward`
that may wrap any :class:`Verifier`.

Design rule for implementers
----------------------------
Do **not** import ``torch`` / ``transformers`` / ``peft`` / ``trl`` at module
top level.  Pull them in lazily inside the method that needs them via
``from ._optional import require`` so that ``import trainall`` stays cheap and
dependency-free.  The only sanctioned exception is the ``models`` and
``algorithms`` subpackages, whose classes subclass ``torch.nn.Module`` at
definition time and are therefore torch-required by design.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Iterator, List, Optional, Sequence, Tuple

from .types import Batch, Episode, Sample, Trajectory, VerifierResult


# --------------------------------------------------------------------------- #
# Objective: "what the model is rewarded to become"
# --------------------------------------------------------------------------- #
class Objective(ABC):
    """A training objective: maps ``(model, batch)`` to a scalar loss.

    Implementations are *stateless w.r.t. parameters* — they never own the
    model, they only score it.  This is the seam that makes "SFT vs DPO vs
    GRPO" a swap of one object while data and optimiser stay fixed.
    """

    #: Registry key, set by the ``@register`` decorator.
    name: str = "objective"

    @abstractmethod
    def compute_loss(self, model: Any, batch: Batch) -> Tuple[Any, dict]:
        """Return ``(loss, metrics)`` where ``loss`` is a backward-able scalar.

        ``metrics`` is a dict of python floats for logging.
        """

    def prepare_batch(self, batch: Batch) -> Batch:
        """Optional hook to massage a batch before :meth:`compute_loss`."""
        return batch

    # Some objectives (DPO, GRPO, ...) need a frozen reference / old policy.
    requires_reference_model: bool = False
    # Policy-gradient objectives consume sampled rollouts, not static data.
    is_on_policy: bool = False


# --------------------------------------------------------------------------- #
# Algorithm: "how parameters update"
# --------------------------------------------------------------------------- #
class Algorithm(ABC):
    """A parameter-update strategy (full finetune, LoRA, QLoRA, ...).

    Orthogonal to the objective: any algorithm can carry any objective.  Its
    job is to take a base model and return a model whose ``.parameters()`` (or a
    subset) are the ones the optimiser will move.
    """

    name: str = "algorithm"

    @abstractmethod
    def prepare_model(self, model: Any) -> Any:
        """Adapt ``model`` for training and return the trainable model."""

    def trainable_parameters(self, model: Any) -> Iterable[Any]:
        """Yield the parameters the optimiser should update."""
        return (p for p in model.parameters() if getattr(p, "requires_grad", False))

    def save(self, model: Any, path: str) -> None:
        """Persist whatever this algorithm considers the artifact."""
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(path)
        else:  # pragma: no cover - exercised with torch present
            from ._optional import require

            torch = require("torch", feature="model checkpointing")
            torch.save(model.state_dict(), path)


# --------------------------------------------------------------------------- #
# Verifier: "is this response correct?" (deterministic, pure-python)
# --------------------------------------------------------------------------- #
class Verifier(ABC):
    """Deterministically check a response, returning a :class:`VerifierResult`.

    Verifiers are the heart of RLVR: math equality, unit tests, SQL execution,
    JSON validity, citation existence.  They must be cheap, side-effect-free
    (or sandboxed), and never depend on a learned model.
    """

    name: str = "verifier"

    @abstractmethod
    def verify(
        self,
        response: str,
        reference: Any = None,
        *,
        prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> VerifierResult:
        ...

    def __call__(self, response: str, reference: Any = None, **kw: Any) -> VerifierResult:
        return self.verify(response, reference, **kw)

    def batch_verify(
        self, responses: Sequence[str], references: Optional[Sequence[Any]] = None, **kw: Any
    ) -> List[VerifierResult]:
        refs = references if references is not None else [None] * len(responses)
        return [self.verify(r, ref, **kw) for r, ref in zip(responses, refs)]


# --------------------------------------------------------------------------- #
# Reward: "score a rollout" (may wrap a Verifier or a reward model)
# --------------------------------------------------------------------------- #
class Reward(ABC):
    """Map sampled trajectories to scalar rewards for policy-gradient RL.

    A :class:`Reward` is the RL-facing adapter: it can wrap a deterministic
    :class:`Verifier` (RLVR), a learned reward model (RLHF), or shape several
    signals together (process supervision + outcome).
    """

    name: str = "reward"

    @abstractmethod
    def score(self, trajectories: Sequence[Trajectory]) -> List[float]:
        ...

    def __call__(self, trajectories: Sequence[Trajectory]) -> List[float]:
        return self.score(trajectories)


# --------------------------------------------------------------------------- #
# DataSource: "where samples come from"
# --------------------------------------------------------------------------- #
class DataSource(ABC):
    """An iterable of :class:`Sample` / :class:`PreferenceSample` objects.

    Concrete sources: JSONL files, HF datasets, in-memory lists, and the
    *generative* sources (synthetic flywheel, rejection sampling, self-play)
    that manufacture their own data.
    """

    name: str = "datasource"

    @abstractmethod
    def __iter__(self) -> Iterator[Any]:
        ...

    def __len__(self) -> int:  # pragma: no cover - optional
        raise TypeError(f"{type(self).__name__} has no defined length")

    def map(self, fn) -> "DataSource":
        return _MappedSource(self, fn)

    def take(self, n: int) -> List[Any]:
        out: List[Any] = []
        for i, item in enumerate(self):
            if i >= n:
                break
            out.append(item)
        return out


class _MappedSource(DataSource):
    def __init__(self, src: DataSource, fn) -> None:
        self._src = src
        self._fn = fn

    def __iter__(self) -> Iterator[Any]:
        return (self._fn(x) for x in self._src)


# --------------------------------------------------------------------------- #
# Environment: "a multi-step world for agentic RL"
# --------------------------------------------------------------------------- #
class Environment(ABC):
    """A resettable, steppable world the policy acts in.

    Mirrors the Gym contract but typed to :class:`trainall.types`.  An agentic
    rollout = ``reset()`` then ``step(action)`` until ``done``, yielding an
    :class:`Episode` whose reward blends outcome and process signals.
    """

    name: str = "environment"

    @abstractmethod
    def reset(self, sample: Optional[Sample] = None) -> Any:
        """Start a new episode; return the initial observation."""

    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, dict]:
        """Apply ``action``; return ``(observation, reward, done, info)``."""

    def rollout(self, policy, sample: Optional[Sample] = None, max_steps: int = 32) -> Episode:
        """Default loop: drive ``policy`` through the env into an Episode."""
        from .types import Transition

        obs = self.reset(sample)
        ep = Episode()
        for _ in range(max_steps):
            action = policy(obs)
            next_obs, reward, done, info = self.step(action)
            ep.add(Transition(observation=obs, action=action, reward=reward, done=done, info=info))
            obs = next_obs
            if done:
                ep.success = bool(info.get("success", reward > 0))
                break
        return ep


__all__ = [
    "Objective",
    "Algorithm",
    "Verifier",
    "Reward",
    "DataSource",
    "Environment",
]
