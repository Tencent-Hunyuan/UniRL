"""unirl stage-driven algorithms; concrete algorithms resolve lazily (PEP 562) so ``base`` imports stay light."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = (
    ("SFT", "sft"),
    ("FlowMatchSFT", "sft"),
    ("GRPO", "grpo"),
    ("GRPOConfig", "grpo"),
    ("GSPO", "gspo"),
    ("GSPOConfig", "gspo"),
    ("PPO", "ppo"),
    ("PPOConfig", "ppo"),
    ("CPPO", "cppo"),
    ("CPPOConfig", "cppo"),
    ("Cosmos3JointFlowMatchSFT", "cosmos3_sft"),
    ("DPPO", "dppo"),
    ("DPPOConfig", "dppo"),
    ("DRPO", "drpo"),
    ("DRPOConfig", "drpo"),
    ("AlgorithmStepResult", "base"),
    ("BagelFlowUniGRPO", "bagel_flow_unigrpo"),
    ("FlowGRPO", "flowgrpo"),
    ("FlowGRPOConfig", "flowgrpo"),
    ("DiffusionNFT", "diffusionnft"),
    ("DiffusionNFTConfig", "diffusionnft"),
    ("DiffusionOPD", "diffusionopd"),
    ("TeacherSpec", "diffusionopd"),
    ("FlowDPPO", "flowdppo"),
    ("FlowDPPOConfig", "flowdppo"),
    ("StageAlgorithm", "base"),
)
_SYMBOL_MODULES = dict(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module = _SYMBOL_MODULES[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    return getattr(import_module(f".{module}", __name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [name for name, _ in _EXPORTS]
