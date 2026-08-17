"""Construction config for the typed SD3 pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from unirl.config.validation import validate_precision_type


@dataclass
class SD3PipelineConfig:
    """Construction args for ``SD3Pipeline.from_config``."""

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
    batch_replay_steps: bool = False

    shift: float = 3.0

    weight_sync_param_name_prefix: str = "transformer."

    use_lora: bool = False
    lora_target_modules: Optional[List[str]] = None

    load_vae: bool = True

    meta_init_transformer: bool = False

    def __post_init__(self) -> None:
        validate_precision_type(self.model_precision, field="SD3PipelineConfig.model_precision")


__all__ = ["SD3PipelineConfig"]
