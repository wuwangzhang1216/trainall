"""LoRA — Low-Rank Adaptation of large language models (Hu et al., 2021).

LoRA freezes the pretrained weight ``W`` of a linear layer and learns a low-rank
update ``ΔW = (alpha / r) * B @ A`` where ``A`` is ``(r, in)`` and ``B`` is
``(out, r)``.  Only ``A`` and ``B`` are trained, slashing the optimiser/gradient
memory while leaving the dense forward path intact:

    y = W x + (alpha / r) * (x A^T) B^T

``B`` is initialised to zero so the adapter starts as a no-op and training begins
exactly at the base model.  After training the adapters can be *folded* back into
the base weights (:func:`merge_lora` / :meth:`LoRALinear.merge`) for zero-overhead
inference.

This subpackage is torch-required, so ``torch`` is imported at module top level
(per the build contract exception for ``algorithms/lora.py``).

Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", 2021.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List

import torch
import torch.nn as nn

from ..base import Algorithm
from ..registry import register

_DEFAULT_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


@dataclass
class LoRAConfig:
    """Hyperparameters for LoRA (Hu et al., 2021).

    Attributes:
        r: Rank of the low-rank update.
        alpha: Scaling numerator; the effective scale is ``alpha / r``.
        dropout: Dropout applied to the adapter input.
        target_modules: Attribute names of ``nn.Linear`` layers to adapt.
        bias: Bias handling — ``"none"`` (default), ``"all"`` or ``"lora_only"``.
    """

    r: int = 8
    alpha: int = 16
    dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: list(_DEFAULT_TARGETS))
    bias: str = "none"


class LoRALinear(nn.Module):
    """A frozen ``nn.Linear`` plus a trainable low-rank adapter.

    Wraps an existing ``base`` linear (weights frozen) and adds the LoRA update
    ``scaling * (dropout(x) @ A^T) @ B^T`` with ``scaling = alpha / r``.  ``A`` is
    kaiming-initialised and ``B`` is zero so the layer starts equal to ``base``.

    Reference: Hu et al., "LoRA", 2021.
    """

    def __init__(
        self,
        base: nn.Linear,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank r must be positive, got {r}")
        self.base = base
        # Freeze the pretrained weight + bias.
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # A: (r, in), B: (out, r).  Match the base weight's dtype/device.
        w = base.weight
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features, dtype=w.dtype, device=w.device))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, dtype=w.dtype, device=w.device))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        # B stays zero -> initial ΔW = 0.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora = self.lora_dropout(x) @ self.lora_A.transpose(0, 1)
        lora = lora @ self.lora_B.transpose(0, 1)
        return base_out + self.scaling * lora

    def delta_weight(self) -> torch.Tensor:
        """The folded update ``ΔW = scaling * (B @ A)`` shaped ``(out, in)``."""
        return self.scaling * (self.lora_B @ self.lora_A)

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        """Fold the adapter into the base weight and return the merged ``nn.Linear``.

        After merging, ``base.weight += scaling * (B @ A)`` and the adapter is
        reset to zero so a re-merge is idempotent.  The (now plain) base linear
        is returned for the caller to splice back into the parent module.
        """
        self.base.weight.add_(self.delta_weight().to(self.base.weight.dtype))
        self.lora_B.zero_()
        return self.base


def _iter_named_linear_parents(model: nn.Module):
    """Yield ``(parent_module, child_name, child_module)`` for every submodule."""
    for parent in model.modules():
        for child_name, child in parent.named_children():
            yield parent, child_name, child


@register("lora", category="algorithm")
class LoRA(Algorithm):
    """Apply LoRA adapters to a model's targeted linear layers (Hu et al., 2021).

    ``prepare_model`` walks the module tree, replaces every ``nn.Linear`` whose
    attribute name matches ``config.target_modules`` with a :class:`LoRALinear`
    (copying + freezing the original weights), and freezes all non-adapter
    parameters.  Only the LoRA ``A``/``B`` tensors are left trainable.
    """

    def __init__(self, config: LoRAConfig | None = None, use_peft: bool = False, **kwargs: Any) -> None:
        if config is None:
            config = LoRAConfig(**kwargs)
        self.config = config
        self.use_peft = use_peft

    # ------------------------------------------------------------------ #
    def prepare_model(self, model: Any) -> Any:
        if self.use_peft:
            return self._prepare_with_peft(model)
        return self._prepare_from_scratch(model)

    def _prepare_from_scratch(self, model: nn.Module) -> nn.Module:
        targets = set(self.config.target_modules)
        # First freeze everything; LoRALinear re-enables its own A/B params.
        for p in model.parameters():
            p.requires_grad_(False)

        for parent, child_name, child in list(_iter_named_linear_parents(model)):
            if child_name in targets and isinstance(child, nn.Linear):
                adapter = LoRALinear(
                    child,
                    r=self.config.r,
                    alpha=self.config.alpha,
                    dropout=self.config.dropout,
                )
                setattr(parent, child_name, adapter)

        if self.config.bias in ("all", "lora_only"):
            for name, p in model.named_parameters():
                if name.endswith("bias") and (
                    self.config.bias == "all" or ".base.bias" in name
                ):
                    p.requires_grad_(True)
        return model

    def _prepare_with_peft(self, model: Any) -> Any:
        from .._optional import require

        peft = require("peft", feature="LoRA via peft")
        lconf = peft.LoraConfig(
            r=self.config.r,
            lora_alpha=self.config.alpha,
            lora_dropout=self.config.dropout,
            target_modules=list(self.config.target_modules),
            bias=self.config.bias,
        )
        return peft.get_peft_model(model, lconf)

    def trainable_parameters(self, model: Any) -> Iterable[Any]:
        """Yield only the parameters with ``requires_grad`` (the LoRA A/B)."""
        return (p for p in model.parameters() if getattr(p, "requires_grad", False))


def merge_lora(model: nn.Module) -> nn.Module:
    """Fold every :class:`LoRALinear` in ``model`` back into a plain ``nn.Linear``.

    Returns the same ``model`` with adapters merged and removed, ready for
    deployment with no LoRA runtime overhead (Hu et al., 2021).
    """
    for parent, child_name, child in list(_iter_named_linear_parents(model)):
        if isinstance(child, LoRALinear):
            merged = child.merge()  # in-place fold into child.base
            setattr(parent, child_name, merged)
    return model


__all__ = ["LoRA", "LoRAConfig", "LoRALinear", "merge_lora"]
