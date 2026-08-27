"""Construction config for the typed HunyuanVideo-1.0 pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from unirl.config.validation import validate_precision_type


@dataclass
class HunyuanVideo10PipelineConfig:
    """Construction args for ``HunyuanVideo10Pipeline.from_config``."""

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

    shift: float = 5.0

    weight_sync_param_name_prefix: str = "transformer."

    meta_init_transformer: bool = False

    latent_channels: Optional[int] = None

    vae_use_tiling: bool = False

    llama_max_length: int = 256
    crop_start: int = 95
    clip_max_length: int = 77
    hidden_state_skip_layer: int = 2

    use_lora: bool = False
    lora_target_modules: Optional[List[str]] = None

    def __post_init__(self) -> None:
        validate_precision_type(self.model_precision, field="HunyuanVideo10PipelineConfig.model_precision")
        self.hidden_state_skip_layer = int(self.hidden_state_skip_layer)
        if self.hidden_state_skip_layer < 0:
            raise ValueError(
                f"HunyuanVideo10PipelineConfig.hidden_state_skip_layer must be >= 0, got {self.hidden_state_skip_layer}"
            )


__all__ = ["HunyuanVideo10PipelineConfig"]
