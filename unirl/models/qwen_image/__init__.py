"""Qwen-Image pipeline on the typed four-tier architecture.

Implements the typed ``Bundle`` / ``Pipeline`` / ``EmbedStage`` /
``DiffusionStage`` / ``DecodeStage`` protocols. Sibling of
:mod:`unirl.models.sd3` and :mod:`unirl.models.wan21`.

Importing this package re-exports its bundle / pipeline / config classes;
recipes wire them by ``_target_`` dotpath.
"""

from unirl.models.qwen_image.bundle import QwenImageBundle
from unirl.models.qwen_image.conditions import QwenImageConditions
from unirl.models.qwen_image.config import QwenImagePipelineConfig
from unirl.models.qwen_image.pipeline import QwenImagePipeline

__all__ = [
    "QwenImageBundle",
    "QwenImageConditions",
    "QwenImagePipeline",
    "QwenImagePipelineConfig",
]
