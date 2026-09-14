"""Out-of-tree vLLM-Omni pipeline topology registrations."""

from __future__ import annotations

from vllm_omni.config.pipeline_registry import OMNI_PIPELINES, register_pipeline
from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

UNIRL_RL_IMAGE_DIFFUSION = PipelineConfig(
    model_type="unirl_rl_image_diffusion",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            final_output=True,
            final_output_type="image",
        ),
    ),
)

_HI3_MODEL_ARCH = "HunyuanImage3ForCausalMM"


def _hi3_text_pipeline(model_type: str, *, requires_multimodal_data: bool) -> PipelineConfig:
    return PipelineConfig(
        model_type=model_type,
        model_arch=_HI3_MODEL_ARCH,
        stages=(
            StagePipelineConfig(
                stage_id=0,
                model_stage="AR",
                execution_type=StageExecutionType.LLM_AR,
                final_output=True,
                final_output_type="text",
                owns_tokenizer=True,
                requires_multimodal_data=requires_multimodal_data,
                model_arch=_HI3_MODEL_ARCH,
                engine_output_type="text",
            ),
        ),
    )


UNIRL_HI3_AR_TEXT = _hi3_text_pipeline(
    "unirl_hi3_ar_text",
    requires_multimodal_data=False,
)
UNIRL_HI3_AR_MULTIMODAL_TEXT = _hi3_text_pipeline(
    "unirl_hi3_ar_multimodal_text",
    requires_multimodal_data=True,
)


def register_unirl_pipeline_configs() -> None:
    """Register UniRL's custom topologies once, rejecting key collisions."""
    for pipeline in (
        UNIRL_RL_IMAGE_DIFFUSION,
        UNIRL_HI3_AR_TEXT,
        UNIRL_HI3_AR_MULTIMODAL_TEXT,
    ):
        existing = OMNI_PIPELINES.get(pipeline.model_type)
        if existing is None:
            register_pipeline(pipeline)
        elif existing != pipeline:
            raise RuntimeError(
                f"vLLM-Omni pipeline key {pipeline.model_type!r} is already "
                f"registered to an incompatible pipeline: {existing!r}"
            )


__all__ = [
    "UNIRL_HI3_AR_MULTIMODAL_TEXT",
    "UNIRL_HI3_AR_TEXT",
    "UNIRL_RL_IMAGE_DIFFUSION",
    "register_unirl_pipeline_configs",
]
