"""Cosmos3 (NVIDIA omnimodal world model) SFT support.

CPU-importable package top: the diffusers>=0.39 runtime imports live inside
``Cosmos3Bundle.from_config`` / ``Cosmos3JointStage.__init__`` bodies.
"""

from unirl.models.cosmos3.bundle import Cosmos3Bundle
from unirl.models.cosmos3.conditions import Cosmos3SFTCondition
from unirl.models.cosmos3.config import Cosmos3SFTConfig
from unirl.models.cosmos3.pipeline import Cosmos3JointStage, Cosmos3Pipeline
from unirl.models.cosmos3.track_builder import Cosmos3SupervisedTrackBuilder

__all__ = [
    "Cosmos3Bundle",
    "Cosmos3JointStage",
    "Cosmos3Pipeline",
    "Cosmos3SFTCondition",
    "Cosmos3SFTConfig",
    "Cosmos3SupervisedTrackBuilder",
]
