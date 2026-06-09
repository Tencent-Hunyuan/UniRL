"""unirl stage-driven algorithms.

Public surface for the ``models`` training contract.
"""

from __future__ import annotations

from .ar_grpo import ARGRPO
from .base import AlgorithmStepResult, StageAlgorithm
from .flowgrpo import FlowGRPO, FlowGRPOConfig
from .drpo import ARDRPO
from .flowdppo import FlowDPPO, FlowDPPOConfig
from .nft import DiffusionNFT, DiffusionNFTConfig

__all__ = [
    "ARGRPO",
    "ARDRPO",
    "AlgorithmStepResult",
    "FlowGRPO",
    "FlowGRPOConfig",
    "DiffusionNFT",
    "DiffusionNFTConfig",
    "FlowDPPO",
    "FlowDPPOConfig",
    "StageAlgorithm",
]
