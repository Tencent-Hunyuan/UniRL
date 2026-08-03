"""MiniMax-H3 -- 33B omni-modal joint video + stereo audio diffusion.

Currently covers the ``t2va`` (text -> video+audio) task of the FL2VA
checkpoint. ``fl2va`` (keyframe conditioning) shares these weights and needs
only conditioning rows; ``ref2va`` is a separate checkpoint and is not here.
"""

from .bundle import MiniMaxH3Bundle
from .conditions import MiniMaxH3Conditions
from .config import (
    MINIMAX_H3_LATENT_CHANNELS,
    MINIMAX_H3_PATCH_SIZE,
    MINIMAX_H3_SPATIAL_COMPRESSION,
    MINIMAX_H3_TEMPORAL_COMPRESSION,
    MiniMaxH3PipelineConfig,
)
from .diffusion import MiniMaxH3DiffusionStage, MiniMaxH3DiffusionStep
from .packing import MiniMaxH3Geometry, build_t2va_layout, row_timestep_plan
from .pipeline import MiniMaxH3Pipeline
from .text_embed import MiniMaxH3TextEmbedStage
from .vae import (
    MINIMAX_H3_AUDIO_SAMPLE_RATE,
    MiniMaxH3AudioDecodeStage,
    MiniMaxH3VideoDecodeStage,
)

__all__ = [
    "MINIMAX_H3_AUDIO_SAMPLE_RATE",
    "MINIMAX_H3_LATENT_CHANNELS",
    "MINIMAX_H3_PATCH_SIZE",
    "MINIMAX_H3_SPATIAL_COMPRESSION",
    "MINIMAX_H3_TEMPORAL_COMPRESSION",
    "MiniMaxH3AudioDecodeStage",
    "MiniMaxH3Bundle",
    "MiniMaxH3Conditions",
    "MiniMaxH3DiffusionStage",
    "MiniMaxH3DiffusionStep",
    "MiniMaxH3Geometry",
    "MiniMaxH3Pipeline",
    "MiniMaxH3PipelineConfig",
    "MiniMaxH3TextEmbedStage",
    "MiniMaxH3VideoDecodeStage",
    "build_t2va_layout",
    "row_timestep_plan",
]
