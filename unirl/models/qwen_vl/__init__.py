"""Qwen2.5-VL vision-language pipeline on the typed stage/pipeline architecture.

AR-only VLM pipeline: text+images in, text out. Supports GRPO training via
ARStage.autoregress + ARStage.replay.

Importing this package re-exports its bundle / pipeline / config classes;
recipes wire them by ``_target_`` dotpath.
"""

from unirl.models.qwen_vl.ar import (
    QwenVLARParams,
    QwenVLARStage,
    QwenVLARStep,
)
from unirl.models.qwen_vl.bundle import QwenVLBundle
from unirl.models.qwen_vl.chat_template import QwenVLChatTemplateStage
from unirl.models.qwen_vl.conditions import QwenVLARConditions
from unirl.models.qwen_vl.config import QwenVLPipelineConfig
from unirl.models.qwen_vl.pipeline import QwenVLPipeline

__all__ = [
    "QwenVLARConditions",
    "QwenVLARParams",
    "QwenVLARStage",
    "QwenVLARStep",
    "QwenVLBundle",
    "QwenVLChatTemplateStage",
    "QwenVLPipeline",
    "QwenVLPipelineConfig",
]
