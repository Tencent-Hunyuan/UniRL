"""The backend seam — the stable public import surface.

Consumers import from this package (``from ...backends import Backend,
VLLMOmniBackend``); the inner layout stays private so impl additions are
consumer-invisible. ``base`` is runtime-free; ``native`` lazy-imports the
vllm-omni runtime only inside ``boot`` and the verbs, so this package imports
on CPU.
"""

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
