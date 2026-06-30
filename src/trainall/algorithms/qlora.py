"""QLoRA — 4-bit quantised base + LoRA adapters (Dettmers et al., 2023).

QLoRA freezes the pretrained model in 4-bit NF4 precision (with optional double
quantisation of the quantisation constants) and trains LoRA adapters on top in
higher precision.  This makes fine-tuning large models feasible on a single GPU
while matching full-precision LoRA quality.

When ``bitsandbytes`` is installed, :meth:`QLoRA.prepare_model` 4-bit-quantises
the targeted base linears before attaching adapters.  When it is absent the
algorithm logs a clear warning via :func:`trainall.utils.get_logger` and falls
back to a full-precision base + LoRA — still fully functional, just without the
memory savings.

This subpackage is torch-required, so ``torch`` is imported at module top level
(per the build contract exception for ``algorithms/qlora.py``).

References:
    * Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs", 2023.
    * Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", 2021.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

import torch
import torch.nn as nn

from .._optional import has
from ..registry import register
from ..utils import get_logger
from .lora import LoRA, LoRAConfig, _DEFAULT_TARGETS, _iter_named_linear_parents

_logger = get_logger(__name__)


@dataclass
class QLoRAConfig(LoRAConfig):
    """LoRA hyperparameters plus 4-bit quantisation settings (Dettmers 2023).

    Attributes:
        bits: Quantisation bit-width of the frozen base (4 for NF4 QLoRA).
        quant_type: bitsandbytes 4-bit data type — ``"nf4"`` or ``"fp4"``.
        double_quant: Whether to quantise the quantisation constants too.
    """

    bits: int = 4
    quant_type: str = "nf4"
    double_quant: bool = True


@register("qlora", category="algorithm")
class QLoRA(LoRA):
    """Quantise the frozen base to 4-bit, then apply LoRA (Dettmers et al., 2023).

    Subclasses :class:`~trainall.algorithms.lora.LoRA` and adds a quantisation
    pre-pass.  The adapter attachment + trainable-parameter logic is inherited.
    """

    def __init__(self, config: QLoRAConfig | None = None, use_peft: bool = False, **kwargs: Any) -> None:
        if config is None:
            config = QLoRAConfig(**kwargs)
        super().__init__(config=config, use_peft=use_peft)
        self.config: QLoRAConfig = config

    def prepare_model(self, model: Any) -> Any:
        if self.use_peft:
            return self._prepare_with_peft(model)
        if has("bitsandbytes"):
            self._quantize_base(model)
        else:
            _logger.warning(
                "bitsandbytes is not installed; QLoRA cannot 4-bit-quantise the "
                "base model. Falling back to full-precision base + LoRA "
                "(functional, but without the QLoRA memory savings). "
                "Install it with: pip install 'trainall[quant]'."
            )
        # Attach LoRA adapters via the inherited from-scratch path.
        return self._prepare_from_scratch(model)

    def _quantize_base(self, model: nn.Module) -> None:
        """Replace targeted ``nn.Linear`` weights with bitsandbytes 4-bit linears."""
        import bitsandbytes as bnb  # local import: only when available

        targets = set(self.config.target_modules)
        for parent, child_name, child in list(_iter_named_linear_parents(model)):
            if child_name in targets and isinstance(child, nn.Linear):
                qlin = bnb.nn.Linear4bit(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    compute_dtype=child.weight.dtype,
                    quant_type=self.config.quant_type,
                    compress_statistics=self.config.double_quant,
                )
                with torch.no_grad():
                    qlin.weight = bnb.nn.Params4bit(
                        child.weight.data,
                        requires_grad=False,
                        quant_type=self.config.quant_type,
                    )
                    if child.bias is not None:
                        qlin.bias = nn.Parameter(child.bias.data, requires_grad=False)
                setattr(parent, child_name, qlin)


__all__ = ["QLoRA", "QLoRAConfig"]
