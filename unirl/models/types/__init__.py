"""Model type protocols and interfaces."""

from __future__ import annotations

from unirl.models.types.ar import ARSamplingParams, ARStage, ARStep
from unirl.models.types.batched_replay import BatchedStepReplayMixin
from unirl.models.types.bundle import Bundle
from unirl.models.types.codec import DecodeStage, EncodeStage
from unirl.models.types.diffusion import DiffusionStage, DiffusionStep
from unirl.models.types.embedding import EmbedStage, ImageConditionedEmbedStage
from unirl.models.types.pipeline import Pipeline
from unirl.models.types.replay_result import ReplayResult

__all__ = [
    "ARSamplingParams",
    "ARStage",
    "ARStep",
    "BatchedStepReplayMixin",
    "Bundle",
    "DecodeStage",
    "DiffusionStage",
    "DiffusionStep",
    "EmbedStage",
    "EncodeStage",
    "ImageConditionedEmbedStage",
    "Pipeline",
    "ReplayResult",
]
