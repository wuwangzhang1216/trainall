"""Knowledge distillation — match a teacher's soft targets (Hinton et al., 2015).

The student is trained to reproduce the teacher's temperature-softened
distribution.  With temperature ``T`` the soft-target term is
``T^2 * KL(p_teacher^T || p_student^T)`` (the ``T^2`` keeps gradient magnitudes
comparable across temperatures), optionally combined with the ordinary
cross-entropy against hard ``labels``:

    total = alpha * KD + (1 - alpha) * CE

A ``reverse`` KL option (``KL(student || teacher)``) gives the mode-seeking
variant favoured by some on-policy distillation recipes.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

from ..base import Objective
from ..registry import register
from ..utils.tensorops import gather_token_logps, masked_mean


@register("distill", category="objective", aliases=["kd"])
class DistillObjective(Objective):
    """Soft-target knowledge distillation with an optional CE anchor.

    Args:
        temperature: softmax temperature ``T`` for the soft targets.
        alpha: weight on the KD term; ``(1 - alpha)`` weights the hard-CE term.
        kind: ``"forward"`` for ``KL(teacher || student)`` (default) or
            ``"reverse"`` for ``KL(student || teacher)``.
    """

    def __init__(self, temperature: float = 2.0, alpha: float = 0.5, kind: str = "forward") -> None:
        self.temperature = float(temperature)
        self.alpha = float(alpha)
        self.kind = kind

    def compute_loss(self, model: Any, batch: Any) -> Tuple[Any, Dict[str, float]]:
        from .._optional import require

        torch = require("torch", feature="distillation loss")

        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        student_logits = getattr(out, "logits", out)  # (B, T, V)

        teacher_logits = batch.extra.get("teacher_logits")
        if teacher_logits is None:
            raise ValueError("DistillObjective needs batch.extra['teacher_logits'] (B, T, V)")
        teacher_logits = teacher_logits.to(student_logits.dtype)

        # Mask over generated/response tokens; default to attention mask.
        if "response_mask" in batch:
            resp_mask = batch["response_mask"].to(torch.float32)
        else:
            resp_mask = attention_mask.to(torch.float32)

        T = self.temperature
        s_logp = torch.log_softmax(student_logits.float() / T, dim=-1)
        t_logp = torch.log_softmax(teacher_logits.float() / T, dim=-1)

        if self.kind == "reverse":
            # KL(student || teacher) = sum p_s (log p_s - log p_t)
            p = s_logp.exp()
            kl_tok = (p * (s_logp - t_logp)).sum(dim=-1)
        else:  # forward: KL(teacher || student) = sum p_t (log p_t - log p_s)
            p = t_logp.exp()
            kl_tok = (p * (t_logp - s_logp)).sum(dim=-1)

        kd = (T * T) * masked_mean(kl_tok, resp_mask)

        # Hard-label cross-entropy anchor (skipped if no labels / alpha == 1).
        ce_val = 0.0
        ce = torch.zeros((), dtype=kd.dtype, device=kd.device)
        labels = batch.get("labels")
        if labels is not None and self.alpha < 1.0:
            token_logp, mask = gather_token_logps(student_logits, labels, ignore_index=-100)
            ce = -masked_mean(token_logp, mask)
            ce_val = float(ce.detach())

        loss = self.alpha * kd + (1.0 - self.alpha) * ce
        return loss, {
            "loss": float(loss.detach()),
            "kd": float(kd.detach()),
            "ce": ce_val,
        }


__all__ = ["DistillObjective"]
