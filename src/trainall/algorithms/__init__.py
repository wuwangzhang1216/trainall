"""Parameter-update strategies — *how* the model updates, orthogonal to *what*.

* :class:`FullFinetune` — train every weight.
* :class:`LoRA`         — freeze the base, train low-rank adapters.
* :class:`QLoRA`        — quantise the frozen base to 4-bit, then LoRA on top.

LoRA/QLoRA wrap a model with adapters via ``prepare_model`` and can ``merge``
adapters back into the base for deployment.  This subpackage is torch-required.
"""
from __future__ import annotations

from .full import FullFinetune
from .lora import LoRA, LoRAConfig, LoRALinear, merge_lora
from .qlora import QLoRA, QLoRAConfig

__all__ = [
    "FullFinetune",
    "LoRA",
    "LoRAConfig",
    "LoRALinear",
    "merge_lora",
    "QLoRA",
    "QLoRAConfig",
]
