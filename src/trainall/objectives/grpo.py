"""GRPO — Group Relative Policy Optimization (Shao et al., 2024; DeepSeek-R1, 2025).

GRPO drops PPO's value network: for a group of ``k`` responses to the same
prompt it normalises rewards *within the group* to form the advantage

    A_i = (r_i - mean_g) / (std_g + eps)

then applies the same clipped policy-gradient surrogate as PPO over the response
tokens (token-mean), optionally with a KL penalty to a reference policy.  When no
``old_logps`` are present the ratio is 1 and the objective reduces to plain
REINFORCE on the group-relative advantage.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..base import Objective
from ..registry import register
from ..utils.tensorops import gather_token_logps, masked_mean


def _group_advantages(rewards: Any, group_ids: Any, norm: str, torch: Any) -> Any:
    """Group-relative advantage; ``norm='group'`` z-scores within each group."""
    rewards = rewards.to(torch.float32)
    adv = torch.zeros_like(rewards)
    groups: Dict[Any, List[int]] = {}
    gid_list = group_ids.tolist() if hasattr(group_ids, "tolist") else list(group_ids)
    for idx, gid in enumerate(gid_list):
        groups.setdefault(gid, []).append(idx)
    for members in groups.values():
        idx = torch.tensor(members, device=rewards.device)
        r = rewards[idx]
        mean = r.mean()
        if norm == "group":
            std = r.std(unbiased=False)
            adv[idx] = (r - mean) / (std + 1e-6)
        else:  # mean-only baseline
            adv[idx] = r - mean
    return adv


@register("grpo", category="objective")
class GRPOObjective(Objective):
    """Group-relative clipped policy gradient (DeepSeek GRPO).

    Args:
        clip_range: PPO-style clip epsilon.
        kl_coef: weight of a per-token KL penalty to ``ref_logps``.
        norm: ``"group"`` for within-group z-scored advantages, else mean-only.
    """

    is_on_policy = True

    def __init__(self, clip_range: float = 0.2, kl_coef: float = 0.0, norm: str = "group") -> None:
        self.clip_range = float(clip_range)
        self.kl_coef = float(kl_coef)
        self.norm = norm

    def prepare_batch(self, batch: Any) -> Any:
        """Compute group-relative ``advantages`` from ``rewards`` + ``group_ids``."""
        if "advantages" in batch:
            return batch
        rewards = batch.get("rewards")
        group_ids = batch.get("group_ids")
        if rewards is None or group_ids is None:
            return batch
        from .._optional import require

        torch = require("torch", feature="GRPO advantages")
        batch.tensors["advantages"] = _group_advantages(rewards, group_ids, self.norm, torch)
        return batch

    def compute_loss(self, model: Any, batch: Any) -> Tuple[Any, Dict[str, float]]:
        from .._optional import require

        torch = require("torch", feature="GRPO loss")

        batch = self.prepare_batch(batch)

        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        response_mask = batch["response_mask"][:, 1:].to(torch.float32)

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = getattr(out, "logits", out)
        logp, _ = gather_token_logps(logits, input_ids, ignore_index=-100)

        old_logps = batch.get("old_logps")
        if old_logps is None:
            # REINFORCE form: ratio is numerically 1 but keeps a gradient path
            # through ``logp`` (exp(logp - logp.detach()) == 1).
            old_logps = logp.detach()
        elif old_logps.shape[-1] == input_ids.shape[-1]:
            old_logps = old_logps[:, 1:]
        ratio = torch.exp(logp - old_logps)

        advantages = batch["advantages"].to(logp.dtype)
        if advantages.dim() == 1:  # per-sequence -> broadcast over tokens
            adv = advantages.unsqueeze(-1).expand_as(ratio)
        else:
            adv = advantages
            if adv.shape[-1] != ratio.shape[-1]:
                adv = adv[..., -ratio.shape[-1]:]

        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv
        pg_per_token = -torch.min(unclipped, clipped)
        loss = masked_mean(pg_per_token, response_mask)

        kl_val = 0.0
        ref_logps = batch.get("ref_logps")
        if self.kl_coef != 0.0 and ref_logps is not None:
            if ref_logps.shape[-1] == input_ids.shape[-1]:
                ref_logps = ref_logps[:, 1:]
            # k3 estimator (Schulman): exp(d) - d - 1, d = ref - logp; >= 0.
            d = ref_logps - logp
            kl_per_token = torch.exp(d) - d - 1.0
            kl_term = masked_mean(kl_per_token, response_mask)
            loss = loss + self.kl_coef * kl_term
            kl_val = float(kl_term.detach())

        return loss, {
            "loss": float(loss.detach()),
            "kl": kl_val,
            "reward_mean": float(batch["rewards"].float().mean()) if batch.get("rewards") is not None else 0.0,
            "adv_std": float(advantages.detach().std()),
        }


__all__ = ["GRPOObjective"]
