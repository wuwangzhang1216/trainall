"""Rollout sampling and the GRPO group-normalised advantage.

The rollout layer turns *prompts* into :class:`~trainall.types.Trajectory`
objects: a prompt, its sampled response, and (when a real model is given) the
generated token ids together with their per-token log-probabilities.  Those
log-probs are the ``old_logps`` a policy-gradient objective (PPO/GRPO) needs.

:func:`compute_group_advantages` implements the baseline that makes GRPO
critic-free (Shao et al., "DeepSeekMath", 2024): for every group of samples
drawn from the same prompt, the advantage of each sample is its reward
standardised within the group, ``(r - mean) / (std + eps)``.

The orchestration is pure-python — a "policy" may be a plain ``str -> str``
callable (used in tests) or an HF ``model`` plus ``tokenizer`` (torch is
imported lazily only on that path), so this module imports without torch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Union

from ..registry import register
from ..types import Trajectory

Policy = Union[Callable[[str], str], Any]


@register("rollout", category="environment")
@dataclass
class RolloutConfig:
    """Sampling hyperparameters for a :class:`Rollout`.

    Attributes mirror the usual decoding knobs.  ``group_size`` is the number
    of completions drawn per prompt for GRPO group sampling.
    """

    group_size: int = 8
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 512
    seed: int = 0


class Rollout:
    """Sample completions from a policy into :class:`Trajectory` objects.

    The ``policy`` is either

    * a plain callable ``str -> str`` (no tokenizer needed, used in tests), or
    * an HF causal-LM ``model`` together with ``tokenizer`` — generation then
      runs under torch, capturing the generated token ids and their per-token
      log-probs (the ``old_logps`` GRPO/PPO consume).

    The class is deliberately thin: it owns no training state, only the
    sampling configuration.
    """

    def __init__(
        self,
        policy: Policy,
        tokenizer: Any = None,
        config: Optional[RolloutConfig] = None,
    ) -> None:
        self.policy = policy
        self.tokenizer = tokenizer
        self.config = config or RolloutConfig()
        # A plain callable policy has no ``.generate``; that is the test path.
        self._is_callable = callable(policy) and not hasattr(policy, "generate")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def generate(self, prompts: Sequence[str]) -> List[Trajectory]:
        """Sample one completion per prompt; return a list of trajectories."""
        if self._is_callable:
            return [self._call_policy(p) for p in prompts]
        return self._generate_torch(list(prompts))

    def group_sample(
        self, prompts: Sequence[str], group_size: Optional[int] = None
    ) -> List[Trajectory]:
        """Sample ``group_size`` completions per prompt (GRPO group sampling).

        Each source prompt becomes a group: every trajectory drawn from it
        shares the same ``group_id`` (the prompt's index), which
        :func:`compute_group_advantages` later normalises over.
        """
        g = group_size or self.config.group_size
        prompts = list(prompts)
        # Replicate each prompt ``g`` times, remembering its source index.
        expanded: List[str] = []
        group_ids: List[int] = []
        for i, p in enumerate(prompts):
            expanded.extend([p] * g)
            group_ids.extend([i] * g)
        trajs = (
            [self._call_policy(p) for p in expanded]
            if self._is_callable
            else self._generate_torch(expanded)
        )
        for traj, gid in zip(trajs, group_ids):
            traj.group_id = gid
        return trajs

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _call_policy(self, prompt: str) -> Trajectory:
        """Run a plain callable policy ``str -> str`` into a Trajectory."""
        response = self.policy(prompt)
        return Trajectory(prompt=prompt, response=str(response))

    def _generate_torch(self, prompts: List[str]) -> List[Trajectory]:
        """Torch generation path: capture token ids + per-token logprobs.

        Generates one completion per (already expanded) prompt.  Per-token
        log-probs are read from the generation scores so they double as the
        ``old_logps`` for an importance-sampling policy-gradient objective.
        """
        from .._optional import require

        torch = require("torch", feature="model rollout generation")
        if self.tokenizer is None:
            raise ValueError("a tokenizer is required to generate from a model")

        model = self.policy
        cfg = self.config
        tok = self.tokenizer
        device = next(model.parameters()).device

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)

        trajectories: List[Trajectory] = []
        model_was_training = model.training
        model.eval()
        try:
            for prompt in prompts:
                enc = tok(prompt, return_tensors="pt")
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                prompt_len = input_ids.shape[1]
                with torch.no_grad():
                    out = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        do_sample=cfg.temperature > 0,
                        temperature=max(cfg.temperature, 1e-8),
                        top_p=cfg.top_p,
                        max_new_tokens=cfg.max_new_tokens,
                        return_dict_in_generate=True,
                        output_scores=True,
                        pad_token_id=getattr(tok, "pad_token_id", None)
                        or getattr(tok, "eos_token_id", None),
                    )
                seq = out.sequences[0]
                gen_ids = seq[prompt_len:]
                # Per-step log-probs of the chosen tokens from generation scores.
                logprobs: List[float] = []
                scores = getattr(out, "scores", None) or []
                for step, logits in enumerate(scores):
                    if step >= gen_ids.shape[0]:
                        break
                    logp = torch.log_softmax(logits[0].float(), dim=-1)
                    logprobs.append(float(logp[gen_ids[step]].item()))
                response = tok.decode(gen_ids, skip_special_tokens=True)
                trajectories.append(
                    Trajectory(
                        prompt=prompt,
                        response=response,
                        token_ids=[int(t) for t in gen_ids.tolist()],
                        logprobs=logprobs,
                    )
                )
        finally:
            if model_was_training:
                model.train()
        return trajectories


def compute_group_advantages(
    trajectories: Sequence[Trajectory], eps: float = 1e-6
) -> List[Trajectory]:
    """Set ``.advantage`` per trajectory using the GRPO group baseline.

    For each ``group_id`` (trajectories sampled from the same prompt) the
    advantage is the within-group standardised reward
    ``(reward - mean) / (std + eps)`` (Shao et al. 2024).  Trajectories with no
    ``group_id`` are treated as one global group.  Mutates and returns the same
    list, so it is a pure, importable helper with no torch dependency.
    """
    trajectories = list(trajectories)
    # Partition by group id (None -> a single shared group).
    groups: dict[Any, List[Trajectory]] = {}
    for traj in trajectories:
        groups.setdefault(traj.group_id, []).append(traj)

    for members in groups.values():
        rewards = [float(t.reward) for t in members]
        n = len(rewards)
        mean = sum(rewards) / n
        var = sum((r - mean) ** 2 for r in rewards) / n
        std = var ** 0.5
        for traj, r in zip(members, rewards):
            traj.advantage = (r - mean) / (std + eps)
    return trajectories


__all__ = ["Rollout", "RolloutConfig", "compute_group_advantages"]
