"""RLOO — REINFORCE Leave-One-Out (Ahmadian et al., 2024).

For each prompt we draw ``k`` samples sharing a ``group_id``.  The baseline for
sample ``i`` is the mean reward of the *other* ``k-1`` samples, so the advantage
``A_i = r_i - mean_{j!=i} r_j`` is an unbiased, variance-reduced REINFORCE
signal.  The loss is ``-mean(A_i * seq_logp_i)`` over the response tokens of
each sample — no critic, no clipping.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..base import Objective
from ..registry import register
from ..utils.tensorops import gather_token_logps, masked_sum


def _leave_one_out_advantages(rewards: Any, group_ids: Any, torch: Any) -> Any:
    """Per-sample leave-one-out advantage ``A_i = r_i - mean_{j!=i in g} r_j``."""
    rewards = rewards.to(torch.float32)
    adv = torch.zeros_like(rewards)
    # Group indices by their (hashable) id.
    groups: Dict[Any, List[int]] = {}
    gid_list = group_ids.tolist() if hasattr(group_ids, "tolist") else list(group_ids)
    for idx, gid in enumerate(gid_list):
        groups.setdefault(gid, []).append(idx)
    for members in groups.values():
        if len(members) == 1:
            adv[members[0]] = 0.0  # no baseline available -> no signal
            continue
        idx = torch.tensor(members, device=rewards.device)
        r = rewards[idx]
        total = r.sum()
        k = float(len(members))
        baseline = (total - r) / (k - 1.0)
        adv[idx] = r - baseline
    return adv


@register("rloo", category="objective")
class RLOOObjective(Objective):
    """REINFORCE with a leave-one-out baseline across ``k`` grouped samples."""

    is_on_policy = True

    def __init__(self) -> None:
        pass

    def prepare_batch(self, batch: Any) -> Any:
        """Compute leave-one-out ``advantages`` from ``rewards`` + ``group_ids``."""
        if "advantages" in batch:
            return batch
        rewards = batch.get("rewards")
        group_ids = batch.get("group_ids")
        if rewards is None or group_ids is None:
            return batch
        from .._optional import require

        torch = require("torch", feature="RLOO advantages")
        batch.tensors["advantages"] = _leave_one_out_advantages(rewards, group_ids, torch)
        return batch

    def compute_loss(self, model: Any, batch: Any) -> Tuple[Any, Dict[str, float]]:
        from .._optional import require

        torch = require("torch", feature="RLOO loss")

        batch = self.prepare_batch(batch)

        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        response_mask = batch["response_mask"][:, 1:].to(torch.float32)

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = getattr(out, "logits", out)
        token_logp, _ = gather_token_logps(logits, input_ids, ignore_index=-100)
        seq_logp = masked_sum(token_logp, response_mask, dim=-1)  # (B,)

        advantages = batch["advantages"].to(seq_logp.dtype)
        loss = -(advantages * seq_logp).mean()
        return loss, {
            "loss": float(loss.detach()),
            "adv_mean": float(advantages.detach().mean()),
            "adv_std": float(advantages.detach().std()),
        }


__all__ = ["RLOOObjective"]
