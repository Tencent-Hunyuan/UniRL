"""unirl stage-driven algorithms."""

from __future__ import annotations

from .bagel_flow_unigrpo import BagelFlowUniGRPO
from .base import AlgorithmStepResult, StageAlgorithm
from .cosmos3_sft import Cosmos3JointFlowMatchSFT
from .cppo import CPPO, CPPOConfig
from .diffusionnft import DiffusionNFT, DiffusionNFTConfig
from .diffusionopd import DiffusionOPD, TeacherSpec
from .dppo import DPPO, DPPOConfig
from .drpo import DRPO, DRPOConfig
from .flowdppo import FlowDPPO, FlowDPPOConfig
from .flowgrpo import FlowGRPO, FlowGRPOConfig
from .grpo import GRPO, GRPOConfig
from .gspo import GSPO, GSPOConfig
from .ppo import PPO, PPOConfig
from .sft import SFT, FlowMatchSFT

__all__ = [
    "SFT",
    "FlowMatchSFT",
    "GRPO",
    "GRPOConfig",
    "GSPO",
    "GSPOConfig",
    "PPO",
    "PPOConfig",
    "CPPO",
    "CPPOConfig",
    "Cosmos3JointFlowMatchSFT",
    "DPPO",
    "DPPOConfig",
    "DRPO",
    "DRPOConfig",
    "AlgorithmStepResult",
    "BagelFlowUniGRPO",
    "FlowGRPO",
    "FlowGRPOConfig",
    "DiffusionNFT",
    "DiffusionNFTConfig",
    "DiffusionOPD",
    "TeacherSpec",
    "FlowDPPO",
    "FlowDPPOConfig",
    "StageAlgorithm",
]
