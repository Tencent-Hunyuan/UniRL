"""unirl stage-driven algorithms.

Public surface for the ``models`` training contract.
"""

from __future__ import annotations

from .ar_grpo import ARGRPO
from .base import AlgorithmStepResult, StageAlgorithm
from .diffusion_grpo import DiffusionGRPO
from .drpo import ARDRPO
from .flowdppo import FlowDPPO, FlowDPPOConfig
from .nft import DiffusionNFT, DiffusionNFTConfig

__all__ = [
    "ARGRPO",
    "ARDRPO",
    "AlgorithmStepResult",
    "DiffusionGRPO",
    "DiffusionNFT",
    "DiffusionNFTConfig",
    "FlowDPPO",
    "FlowDPPOConfig",
    "StageAlgorithm",
]
