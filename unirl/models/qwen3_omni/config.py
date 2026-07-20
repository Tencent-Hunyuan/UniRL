"""Construction config for the Qwen3-Omni thinker AR pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from unirl.config.validation import validate_precision_type


@dataclass
class Qwen3OmniPipelineConfig:
    """Arguments for constructing a bundle and pipeline."""

    pretrained_model_ckpt_path: str
    tokenizer_ckpt_path: Optional[str] = None
    trust_remote_code: bool = True

    model_precision: Any = "bf16"
    # Global HF attention backend used by both replay and autoregression.
    attn_implementation: Optional[str] = None
    device: Any = None

    autocast_precision: str = "bf16"
    logprob_precision: str = "fp32"

    use_gradient_checkpointing: bool = False

    # Weight-sync prefix for the thinker's decoder under ``model``.
    weight_sync_param_name_prefix: str = "model."

    use_lora: bool = False
    lora_target_modules: Optional[List[str]] = None

    # Freeze embedded encoders while training the decoder.
    freeze_vision_tower: bool = True
    freeze_audio_tower: bool = True

    max_prompt_length: int = 4096

    # Video sampling rate used to derive TMRoPE timing.
    video_fps: float = 1.0
    video_max_frames: Optional[int] = None
    # Per-frame pixel cap passed to the processor as ``size.longest_edge``.
    video_max_pixels: Optional[int] = None
    # Whether to include the video's audio track in TMRoPE inputs.
    use_audio_in_video: bool = False

    # Unsupported until the FSDP loader remaps checkpoint ``thinker.`` keys.
    meta_init_transformer: bool = False

    system_instruction: Optional[str] = None

    def __post_init__(self) -> None:
        validate_precision_type(self.model_precision, field="Qwen3OmniPipelineConfig.model_precision")


__all__ = ["Qwen3OmniPipelineConfig"]
