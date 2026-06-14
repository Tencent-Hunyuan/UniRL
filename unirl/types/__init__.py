"""
Cross-module data types for unirl.

This package provides shared dataclasses and validation helpers used by:
- rollout control-plane
- ray actors
- samplers and losses
"""

from __future__ import annotations

import importlib
from typing import Dict, Tuple

_LAZY_ATTRS: Dict[str, Tuple[str, str]] = {
    "RewardRequest": ("unirl.types.reward", "RewardRequest"),
    "RewardResponse": ("unirl.types.reward", "RewardResponse"),
    "RewardType": ("unirl.types.reward", "RewardType"),
    "EngineConfig": ("unirl.types.engine", "EngineConfig"),
    "MediaPreview": ("unirl.types.media_preview", "MediaPreview"),
    "RolloutReq": ("unirl.types.rollout_req", "RolloutReq"),
    "RolloutResp": ("unirl.types.rollout_resp", "RolloutResp"),
    "RolloutTrack": ("unirl.types.rollout_resp", "RolloutTrack"),
    "ARSamplingParams": ("unirl.types.sampling", "ARSamplingParams"),
    "BaseSamplingParams": ("unirl.types.sampling", "BaseSamplingParams"),
    "ComposedSamplingParams": ("unirl.types.sampling", "ComposedSamplingParams"),
    "DiffusionSamplingParams": ("unirl.types.sampling", "DiffusionSamplingParams"),
    "get_diffusion_params": ("unirl.types.sampling", "get_diffusion_params"),
    "TrajectoryStore": ("unirl.types.trajectory_store", "TrajectoryStore"),
}

__all__ = list(_LAZY_ATTRS.keys())


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        module_name, attr_name = _LAZY_ATTRS[name]
        module = importlib.import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals().keys()) | set(__all__))
