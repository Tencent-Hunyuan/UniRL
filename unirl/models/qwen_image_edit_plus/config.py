"""Construction config for the typed Qwen-Image-Edit-Plus pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from unirl.config.validation import validate_precision_type
from unirl.models.qwen_image.config import _qwen_image_dynamic_overrides


@dataclass
class QwenImageEditPlusPipelineConfig:
    """Construction args for :meth:`QwenImageEditPlusPipeline.from_config`."""

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

    max_sequence_length: int = 512

    use_condition_image_prompt: bool = True

    use_lora: bool = False
    lora_target_modules: Optional[List[str]] = None

    load_text_encoder: bool = True
    load_vae: bool = True
    meta_init_transformer: bool = False

    use_dynamic_shifting: bool = True
    dynamic_shift_overrides: Dict[str, Any] = field(default_factory=_qwen_image_dynamic_overrides)

    def __post_init__(self) -> None:
        validate_precision_type(self.model_precision, field="QwenImageEditPlusPipelineConfig.model_precision")


__all__ = ["QwenImageEditPlusPipelineConfig"]
