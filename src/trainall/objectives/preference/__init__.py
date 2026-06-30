"""Offline preference optimisation — learn from ``chosen`` > ``rejected``.

All variants share one idea (raise the relative log-prob of the preferred
response) and differ in the link function / regulariser:

* :class:`DPOObjective`   — sigmoid of beta-scaled implicit-reward margin
* :class:`IPOObjective`   — squared loss on the margin (less over-fitting)
* :class:`KTOObjective`   — unpaired, prospect-theory utility on single labels
* :class:`ORPOObjective`  — reference-free, odds-ratio penalty added to SFT
* :class:`SimPOObjective` — reference-free, length-normalised reward + margin
* :class:`CPOObjective`   — reference-free contrastive + SFT anchor
"""
from __future__ import annotations

from .dpo import DPOObjective
from .ipo import IPOObjective
from .kto import KTOObjective
from .orpo import ORPOObjective
from .simpo import SimPOObjective
from .cpo import CPOObjective

__all__ = [
    "DPOObjective",
    "IPOObjective",
    "KTOObjective",
    "ORPOObjective",
    "SimPOObjective",
    "CPOObjective",
]
