"""unirl stage-driven algorithms; concrete algorithms resolve lazily (PEP 562) so ``base`` imports stay light."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_SYMBOL_MODULES = {
    "AlgorithmStepResult": "base",
    "StageAlgorithm": "base",
    "BagelFlowUniGRPO": "bagel_flow_unigrpo",
    "Cosmos3JointFlowMatchSFT": "cosmos3_sft",
    "CPPO": "cppo",
    "CPPOConfig": "cppo",
    "DiffusionNFT": "diffusionnft",
    "DiffusionNFTConfig": "diffusionnft",
    "DiffusionOPD": "diffusionopd",
    "TeacherSpec": "diffusionopd",
    "DPPO": "dppo",
    "DPPOConfig": "dppo",
    "DRPO": "drpo",
    "DRPOConfig": "drpo",
    "FlowDPPO": "flowdppo",
    "FlowDPPOConfig": "flowdppo",
    "FlowGRPO": "flowgrpo",
    "FlowGRPOConfig": "flowgrpo",
    "GRPO": "grpo",
    "GRPOConfig": "grpo",
    "GSPO": "gspo",
    "GSPOConfig": "gspo",
    "PPO": "ppo",
    "PPOConfig": "ppo",
    "SFT": "sft",
    "FlowMatchSFT": "sft",
}


def __getattr__(name: str) -> Any:
    try:
        module = _SYMBOL_MODULES[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    return getattr(import_module(f".{module}", __name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)


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
