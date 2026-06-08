"""Qwen3 causal LM pipeline on the typed stage/pipeline architecture.

Port of :class:`unirl.models.llm.bundle.LLMModelBundle` (the
generic HF causal LM adapter that ``unirl-pe`` tested with Qwen3)
into the typed ``Bundle`` / ``Pipeline`` / ``EmbedStage`` / ``ARStage``
protocols. Sibling of :mod:`unirl.models.qwen_image`.

Importing this package re-exports its bundle / pipeline / config classes;
recipes wire them by ``_target_`` dotpath.
"""

from unirl.models.qwen3.ar import (
    Qwen3ARParams,
    Qwen3ARStage,
    Qwen3ARStep,
)
from unirl.models.qwen3.bundle import Qwen3Bundle
from unirl.models.qwen3.chat_template import Qwen3ChatTemplateStage
from unirl.models.qwen3.conditions import Qwen3ARConditions
from unirl.models.qwen3.config import Qwen3PipelineConfig
from unirl.models.qwen3.pipeline import Qwen3Pipeline

__all__ = [
    "Qwen3ARConditions",
    "Qwen3ARParams",
    "Qwen3ARStage",
    "Qwen3ARStep",
    "Qwen3Bundle",
    "Qwen3ChatTemplateStage",
    "Qwen3Pipeline",
    "Qwen3PipelineConfig",
]
