"""Construction config for the Qwen3-Omni thinker AR pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from unirl.config.validation import validate_precision_type


@dataclass
class Qwen3OmniPipelineConfig:
    """Arguments for constructing a bundle and pipeline."""

    pretrained_model_ckpt_path: str
    tokenizer_ckpt_path: Optional[str] = None
    trust_remote_code: bool = True

    model_precision: Any = "bf16"
    attn_implementation: Optional[str] = None
    device: Any = None

    autocast_precision: str = "bf16"
    logprob_precision: str = "fp32"

    use_gradient_checkpointing: bool = False

    weight_sync_param_name_prefix: str = "model."

    use_lora: bool = False
    lora_target_modules: Optional[List[str]] = None

    freeze_vision_tower: bool = True
    freeze_audio_tower: bool = True

    max_prompt_length: int = 4096

    video_fps: float = 1.0
    video_max_frames: Optional[int] = None
    video_max_pixels: Optional[int] = None
    use_audio_in_video: bool = False

    meta_init_transformer: bool = False

    system_instruction: Optional[str] = None
    chat_template_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_precision_type(self.model_precision, field="Qwen3OmniPipelineConfig.model_precision")
        if float(self.video_fps) <= 0.0:
            raise ValueError(f"Qwen3OmniPipelineConfig.video_fps must be > 0, got {self.video_fps!r}")
        if self.video_max_frames is not None and int(self.video_max_frames) < 1:
            raise ValueError(f"Qwen3OmniPipelineConfig.video_max_frames must be >= 1, got {self.video_max_frames!r}")
        if self.video_max_pixels is not None and int(self.video_max_pixels) < 1:
            raise ValueError(f"Qwen3OmniPipelineConfig.video_max_pixels must be >= 1, got {self.video_max_pixels!r}")


__all__ = ["Qwen3OmniPipelineConfig"]
