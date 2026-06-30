"""Composable, end-to-end recipes — the *frontier pipeline* as code.

A :class:`Pipeline` is an ordered list of :class:`Stage` objects, each one a
named training phase that consumes the previous stage's model.  The canonical
frontier flow

    CPT → SFT → (synthetic/rejection expand) → DPO → RLVR/GRPO → Agentic RL → Distill

is just ``frontier_pipeline()``.  Named single-phase recipes are provided for
the common cases.
"""
from __future__ import annotations

from .base import Pipeline, Stage, StageResult
from .recipes import (
    agentic_rlvr_recipe,
    cpt_recipe,
    distill_recipe,
    dpo_recipe,
    frontier_pipeline,
    rlvr_recipe,
    sft_recipe,
)

__all__ = [
    "Pipeline",
    "Stage",
    "StageResult",
    "cpt_recipe",
    "sft_recipe",
    "dpo_recipe",
    "rlvr_recipe",
    "agentic_rlvr_recipe",
    "distill_recipe",
    "frontier_pipeline",
]
