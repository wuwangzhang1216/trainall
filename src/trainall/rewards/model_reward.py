"""Learned reward-model reward — the RLHF / RLAIF signal.

Wraps a sequence-classification style reward model that maps a
``(prompt, response)`` pair to a scalar preference score (Christiano et al.
2017; Ouyang et al. 2022 "InstructGPT").  torch / transformers are imported
lazily so this module is importable without an ML stack, and a plain
``scorer(prompt, response) -> float`` callable is accepted for testing without
any model at all.
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence

from ..base import Reward
from ..registry import register
from ..types import Trajectory


@register("reward_model", category="reward", aliases=["rm"])
class RewardModelReward(Reward):
    """Score ``(prompt, response)`` pairs with a learned reward model.

    Parameters
    ----------
    model:
        A HF ``AutoModelForSequenceClassification``-style reward model (or any
        object whose forward returns ``.logits``).  Used together with
        ``tokenizer``.  Heavy torch work happens lazily inside :meth:`score`.
    tokenizer:
        Tokenizer paired with ``model``; turns the formatted pair into ids.
    scorer:
        Optional plain callable ``fn(prompt, response) -> float`` that bypasses
        the model entirely.  When given it takes precedence — this is the
        dependency-free path used in tests.
    batch_size:
        Mini-batch size for the model forward pass.
    template:
        Format string with ``{prompt}`` / ``{response}`` placeholders used to
        build the text fed to the reward model.
    """

    def __init__(
        self,
        model: Any = None,
        tokenizer: Any = None,
        scorer: Any = None,
        *,
        batch_size: int = 8,
        template: str = "{prompt}\n{response}",
        max_length: int = 2048,
    ) -> None:
        if scorer is None and model is None:
            raise ValueError("RewardModelReward needs either a `scorer` callable or a `model`.")
        if scorer is not None and not callable(scorer):
            raise TypeError("`scorer` must be callable: fn(prompt, response) -> float")
        if model is not None and scorer is None and tokenizer is None:
            raise ValueError("a reward `model` also requires a `tokenizer`.")
        self.model = model
        self.tokenizer = tokenizer
        self.scorer = scorer
        self.batch_size = int(batch_size)
        self.template = template
        self.max_length = int(max_length)

    def score(self, trajectories: Sequence[Trajectory]) -> List[float]:
        if self.scorer is not None:
            return [float(self.scorer(t.prompt, t.response)) for t in trajectories]
        return self._score_with_model(trajectories)

    def _score_with_model(self, trajectories: Sequence[Trajectory]) -> List[float]:
        torch = _torch()
        texts = [
            self.template.format(prompt=t.prompt or "", response=t.response or "")
            for t in trajectories
        ]
        device = self._device()
        rewards: List[float] = []
        was_training = getattr(self.model, "training", False)
        if hasattr(self.model, "eval"):
            self.model.eval()
        try:
            with torch.no_grad():
                for start in range(0, len(texts), self.batch_size):
                    chunk = texts[start : start + self.batch_size]
                    enc = self.tokenizer(
                        chunk,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                    )
                    enc = {k: v.to(device) for k, v in enc.items()}
                    out = self.model(**enc)
                    logits = out.logits if hasattr(out, "logits") else out
                    # (B, 1) scalar head -> (B,); multi-class -> use last column.
                    scores = logits.squeeze(-1) if logits.shape[-1] == 1 else logits[:, -1]
                    rewards.extend(scores.float().cpu().tolist())
        finally:
            if was_training and hasattr(self.model, "train"):
                self.model.train()
        return rewards

    def _device(self) -> Any:
        params = getattr(self.model, "parameters", None)
        if callable(params):
            try:
                return next(self.model.parameters()).device
            except StopIteration:  # pragma: no cover - empty model
                pass
        return "cpu"


def _torch() -> Any:
    from .._optional import require

    return require("torch", feature="reward-model scoring")


__all__ = ["RewardModelReward"]
