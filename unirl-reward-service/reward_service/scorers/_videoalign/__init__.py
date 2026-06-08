"""Vendored VideoAlign / VideoReward inference subset.

Minimal slice of the upstream VideoAlign repo
(https://github.com/KwaiVGI/VideoAlign) needed to run the VideoReward
Qwen2-VL-2B reward model in inference mode. Training-only parts
(trainer, dataset converters, data collators, deepspeed helpers) are
intentionally stripped — see this directory's README.md for the full
provenance and the list of removals / changes.
"""

from __future__ import annotations

from reward_service.scorers._videoalign.configs import (
    DataConfig,
    ModelConfig,
    PEFTLoraConfig,
)
from reward_service.scorers._videoalign.inferencer import VideoRewardInferencer

__all__ = [
    "DataConfig",
    "ModelConfig",
    "PEFTLoraConfig",
    "VideoRewardInferencer",
]
