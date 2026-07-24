"""unirl stage-driven algorithms.

Public surface for the ``models`` training contract.
"""

from __future__ import annotations

from .bagel_flow_unigrpo import BagelFlowUniGRPO
from .base import AlgorithmStepResult, StageAlgorithm
from .cppo import CPPO, CPPOConfig
from .diffusionnft import DiffusionNFT, DiffusionNFTConfig
from .dppo import DPPO, DPPOConfig
from .drpo import DRPO, DRPOConfig
from .flowdppo import FlowDPPO, FlowDPPOConfig
from .flowgrpo import FlowGRPO, FlowGRPOConfig
from .grpo import GRPO, GRPOConfig
from .gspo import GSPO, GSPOConfig
from .sft import SFT, FlowMatchSFT

__all__ = [
    "SFT",
    "FlowMatchSFT",
    "GRPO",
    "GRPOConfig",
    "GSPO",
    "GSPOConfig",
    "CPPO",
    "CPPOConfig",
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
    "FlowDPPO",
    "FlowDPPOConfig",
    "StageAlgorithm",
]
