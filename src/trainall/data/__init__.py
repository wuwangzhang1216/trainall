"""Data sources + the generative data flywheel — *where samples come from*.

Static sources
* :class:`JsonlSource`, :class:`HFDatasetSource`, :class:`InMemorySource`

Templating / packing
* :class:`ChatTemplate`, :func:`apply_template`, :func:`mask_prompt`
* :class:`Packer`, :func:`pack_sequences`

Generative sources (self-improving)
* :class:`SyntheticDataEngine` — proposer → solver → verifier → keep.
* :class:`RejectionSampler`    — best-of-N, keep verifier-passing traces.
* :class:`SelfPlayLoop`        — difficulty-curriculum task generation.
"""
from __future__ import annotations

from .sources import HFDatasetSource, InMemorySource, JsonlSource
from .templates import ChatTemplate, apply_template, mask_prompt
from .packing import Packer, pack_sequences
from .synthetic import SyntheticDataEngine
from .rejection_sampling import RejectionSampler
from .selfplay import Curriculum, SelfPlayLoop, TaskProposer

__all__ = [
    "JsonlSource",
    "HFDatasetSource",
    "InMemorySource",
    "ChatTemplate",
    "apply_template",
    "mask_prompt",
    "Packer",
    "pack_sequences",
    "SyntheticDataEngine",
    "RejectionSampler",
    "SelfPlayLoop",
    "TaskProposer",
    "Curriculum",
]
