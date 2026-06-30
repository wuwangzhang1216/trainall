"""Proximal Policy Optimization — clipped policy-gradient (Schulman et al., 2017).

PPO maximises a clipped surrogate of the policy-gradient objective so a single
batch can be reused for several gradient steps without the policy moving too far
from the behaviour policy that generated it.  Per response token we form the
importance ratio ``r = exp(logp - old_logp)`` and minimise

    -E[min(r * A, clip(r, 1-eps, 1+eps) * A)]

optionally adding a value-function regression loss, an entropy bonus, and a KL
penalty to a reference policy.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

from ..base import Objective
from ..registry import register
from ..utils.tensorops import entropy_from_logits, gather_token_logps, masked_mean


def compute_gae(
    rewards: Any,
    values: Any,
    gamma: float = 1.0,
    lam: float = 0.95,
    mask: Any = None,
) -> Tuple[Any, Any]:
    """Generalised Advantage Estimation (Schulman et al., 2016).

    ``rewards`` / ``values`` are ``(B, T)`` per-token tensors aligned to the
    response.  Returns ``(advantages, returns)`` both ``(B, T)`` where
    ``returns = advantages + values``.  ``mask`` (``(B, T)``) zeroes padding so
    bootstrapping stops at the sequence boundary.
    """
    from .._optional import require

    torch = require("torch", feature="GAE")
    if mask is None:
        mask = torch.ones_like(rewards)
    mask = mask.to(rewards.dtype)
    rewards = rewards * mask
    values = values * mask

    bsz, length = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(bsz, dtype=rewards.dtype, device=rewards.device)
    for t in reversed(range(length)):
        next_values = values[:, t + 1] if t + 1 < length else torch.zeros_like(values[:, 0])
        next_mask = mask[:, t + 1] if t + 1 < length else torch.zeros_like(mask[:, 0])
        delta = rewards[:, t] + gamma * next_values * next_mask - values[:, t]
        last_gae = delta + gamma * lam * next_mask * last_gae
        advantages[:, t] = last_gae * mask[:, t]
    returns = advantages + values
    return advantages, returns


def _policy_logps(model: Any, input_ids: Any, attention_mask: Any) -> Tuple[Any, Any]:
    """Return ``(per_token_logps (B, T-1), logits (B, T, V))`` for ``input_ids``."""
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = getattr(out, "logits", out)
    # log p of the realised next token, shifted: position t scores token t+1.
    token_logp, _ = gather_token_logps(logits, input_ids, ignore_index=-100)
    return token_logp, logits


@register("ppo", category="objective")
class PPOObjective(Objective):
    """Clipped PPO surrogate over response tokens.

    Args:
        clip_range: PPO clip epsilon.
        vf_coef: weight of the value-function loss (used iff ``values`` present).
        ent_coef: weight of the entropy bonus.
        kl_coef: weight of a per-token KL penalty to ``ref_logps`` (if present).
    """

    is_on_policy = True

    def __init__(
        self,
        clip_range: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.0,
        kl_coef: float = 0.0,
    ) -> None:
        self.clip_range = float(clip_range)
        self.vf_coef = float(vf_coef)
        self.ent_coef = float(ent_coef)
        self.kl_coef = float(kl_coef)

    def compute_loss(self, model: Any, batch: Any) -> Tuple[Any, Dict[str, float]]:
        from .._optional import require

        torch = require("torch", feature="PPO loss")

        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        # response_mask is aligned to input_ids; shift to match the (T-1) logps.
        response_mask = batch["response_mask"][:, 1:].to(torch.float32)
        advantages = batch["advantages"]
        if advantages.dim() == 1:  # per-sequence -> broadcast over response tokens
            advantages = advantages.unsqueeze(-1)
        if advantages.shape[-1] != response_mask.shape[-1]:
            advantages = advantages[..., -response_mask.shape[-1]:] if advantages.shape[-1] > 1 else advantages

        logp, logits = _policy_logps(model, input_ids, attention_mask)
        old_logps = batch.get("old_logps")
        if old_logps is None:
            old_logps = logp.detach()
        else:
            if old_logps.shape[-1] == input_ids.shape[-1]:
                old_logps = old_logps[:, 1:]

        ratio = torch.exp(logp - old_logps)
        adv = advantages.to(ratio.dtype)
        if adv.shape[-1] == 1:
            adv = adv.expand_as(ratio)
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv
        pg_per_token = -torch.min(unclipped, clipped)
        pg_loss = masked_mean(pg_per_token, response_mask)

        metrics: Dict[str, float] = {}
        loss = pg_loss
        metrics["pg_loss"] = float(pg_loss.detach())

        # Entropy bonus (encourages exploration).
        if self.ent_coef != 0.0:
            ent = entropy_from_logits(logits)[:, :-1]
            ent_term = masked_mean(ent, response_mask)
            loss = loss - self.ent_coef * ent_term
            metrics["entropy"] = float(ent_term.detach())

        # KL penalty to a reference policy (per-token ref logps if provided).
        ref_logps = batch.get("ref_logps")
        if self.kl_coef != 0.0 and ref_logps is not None:
            if ref_logps.shape[-1] == input_ids.shape[-1]:
                ref_logps = ref_logps[:, 1:]
            kl = logp - ref_logps
            kl_term = masked_mean(kl, response_mask)
            loss = loss + self.kl_coef * kl_term
            metrics["kl"] = float(kl_term.detach())

        # Value-function regression (if a critic produced ``values``/``returns``).
        values = batch.get("values")
        returns = batch.get("returns")
        if values is not None and returns is not None:
            v = values
            r = returns
            if v.shape[-1] == input_ids.shape[-1]:
                v = v[:, 1:]
            if r.shape[-1] == input_ids.shape[-1]:
                r = r[:, 1:]
            vf_per_token = (v - r) ** 2
            vf_loss = 0.5 * masked_mean(vf_per_token, response_mask)
            loss = loss + self.vf_coef * vf_loss
            metrics["vf_loss"] = float(vf_loss.detach())

        metrics["loss"] = float(loss.detach())
        metrics["ratio"] = float(masked_mean(ratio, response_mask).detach())
        return loss, metrics


__all__ = ["PPOObjective", "compute_gae"]
