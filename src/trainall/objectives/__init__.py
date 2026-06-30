"""Training objectives — *what the model is rewarded to become*.

Every objective implements :class:`trainall.base.Objective` and registers under
a short key, so ``trainall.build("dpo")`` returns one.  None of these modules
import torch at top level (torch is pulled in lazily inside ``compute_loss``),
so importing this package is cheap.

Families
--------
* likelihood     : :class:`CausalLMObjective`, :class:`ContinuedPretrainObjective`, :class:`SFTObjective`
* preference     : DPO / IPO / KTO / ORPO / SimPO / CPO  (see ``.preference``)
* reward / RL    : :class:`BradleyTerryObjective`, :class:`PPOObjective`,
                   :class:`RLOOObjective`, :class:`GRPOObjective`
* supervision    : :class:`ProcessRewardObjective`, :class:`DistillObjective`
"""
from __future__ import annotations

from .pretrain import CausalLMObjective
from .cpt import ContinuedPretrainObjective
from .sft import SFTObjective
from .preference import (
    CPOObjective,
    DPOObjective,
    IPOObjective,
    KTOObjective,
    ORPOObjective,
    SimPOObjective,
)
from .reward_model import BradleyTerryObjective
from .ppo import PPOObjective
from .rloo import RLOOObjective
from .grpo import GRPOObjective
from .prm import ProcessRewardObjective
from .distill import DistillObjective

__all__ = [
    "CausalLMObjective",
    "ContinuedPretrainObjective",
    "SFTObjective",
    "DPOObjective",
    "IPOObjective",
    "KTOObjective",
    "ORPOObjective",
    "SimPOObjective",
    "CPOObjective",
    "BradleyTerryObjective",
    "PPOObjective",
    "RLOOObjective",
    "GRPOObjective",
    "ProcessRewardObjective",
    "DistillObjective",
]
