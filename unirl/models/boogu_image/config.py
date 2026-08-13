"""Construction config for the typed Boogu-Image pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from unirl.config.validation import validate_precision_type

BOOGU_IMAGE_BASE_STATIC_SHIFT = 3.158192909689768

_ATTENTION_BACKENDS = ("sdpa", "flash2_varlen")


@dataclass
class BooguImagePipelineConfig:
    """Construction args for ``BooguImagePipeline.from_config``."""

    pretrained_model_ckpt_path: str
    vae_ckpt_path: Optional[str] = None
    text_encoder_ckpt_path: Optional[str] = None
    vae_dtype: Any = None
    text_encoder_dtype: Any = None
    model_precision: Any = "bf16"
    device: Any = None

    autocast_precision: str = "bf16"
    trajectory_precision: str = "fp16"
    logprob_precision: str = "fp32"

    shift: float = BOOGU_IMAGE_BASE_STATIC_SHIFT

    weight_sync_param_name_prefix: str = "transformer."

    max_sequence_length: int = 1280

    use_lora: bool = False
    lora_target_modules: Optional[List[str]] = None

    load_text_encoder: bool = True

    load_vae: bool = True

    meta_init_transformer: bool = False

    attention_backend: str = "sdpa"

    def __post_init__(self) -> None:
        validate_precision_type(self.model_precision, field="BooguImagePipelineConfig.model_precision")
        if self.attention_backend not in _ATTENTION_BACKENDS:
            raise ValueError(
                f"BooguImagePipelineConfig.attention_backend must be one of "
                f"{_ATTENTION_BACKENDS}, got {self.attention_backend!r}"
            )


__all__ = ["BooguImagePipelineConfig", "BOOGU_IMAGE_BASE_STATIC_SHIFT"]
