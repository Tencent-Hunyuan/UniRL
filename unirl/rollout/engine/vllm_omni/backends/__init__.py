"""The backend seam — the stable public import surface."""

from unirl.rollout.engine.vllm_omni.backends.base import (
    STAGE_KIND_AR,
    STAGE_KIND_DIFFUSION,
    Backend,
    GenerateCall,
    OmniRawResult,
    StageSampling,
)
from unirl.rollout.engine.vllm_omni.backends.native import VLLMOmniBackend

__all__ = [
    "Backend",
    "GenerateCall",
    "OmniRawResult",
    "StageSampling",
    "STAGE_KIND_AR",
    "STAGE_KIND_DIFFUSION",
    "VLLMOmniBackend",
]
