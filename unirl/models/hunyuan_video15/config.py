"""Construction config for the typed HunyuanVideo-1.5 pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from unirl.config.validation import validate_precision_type


@dataclass
class HunyuanVideo15PipelineConfig:
    """Construction args for ``HunyuanVideo15Pipeline.from_config``."""

    pretrained_model_ckpt_path: str
    vae_ckpt_path: Optional[str] = None
    text_encoder_ckpt_path: Optional[str] = None
    text_encoder_2_ckpt_path: Optional[str] = None
    image_encoder_ckpt_path: Optional[str] = None
    vae_dtype: Any = None
    text_encoder_dtype: Any = None
    model_precision: Any = "bf16"
    device: Any = None

    autocast_precision: str = "bf16"
    trajectory_precision: str = "fp16"
    logprob_precision: str = "fp32"

    shift: float = 5.0

    weight_sync_param_name_prefix: str = "transformer."

    meta_init_transformer: bool = False

    latent_channels: Optional[int] = None

    mllm_max_length: int = 1000
    mllm_crop_start: int = 108
    mllm_skip_layers: int = 2
    byt5_max_length: int = 256

    vision_num_semantic_tokens: int = 729
    vision_states_dim: int = 1152
    load_vision_encoder: bool = False

    load_vae: bool = True

    use_lora: bool = False
    lora_target_modules: Optional[List[str]] = None

    def __post_init__(self) -> None:
        validate_precision_type(self.model_precision, field="HunyuanVideo15PipelineConfig.model_precision")


__all__ = ["HunyuanVideo15PipelineConfig"]
