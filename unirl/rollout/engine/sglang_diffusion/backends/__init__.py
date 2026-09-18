"""The backend seam package — the runtime boundary of the engine."""

from unirl.rollout.engine.sglang_diffusion.backends.base import (
    Backend,
    EncoderOutputs,
    MediaPayload,
    RawResult,
)
from unirl.rollout.engine.sglang_diffusion.backends.native import SGLangBackend

__all__ = ["Backend", "SGLangBackend", "RawResult", "EncoderOutputs", "MediaPayload"]
