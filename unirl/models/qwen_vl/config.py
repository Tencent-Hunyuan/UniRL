from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from unirl.config.validation import validate_precision_type


@dataclass
class QwenVLPipelineConfig:
    pretrained_model_ckpt_path: str
    tokenizer_ckpt_path: Optional[str] = None
    trust_remote_code: bool = True

    model_precision: Any = "bf16"
    device: Any = None

    autocast_precision: str = "bf16"
    logprob_precision: str = "fp32"

    use_gradient_checkpointing: bool = False

    weight_sync_param_name_prefix: str = "model."

    use_lora: bool = False
    lora_target_modules: Optional[List[str]] = None

    freeze_vision_tower: bool = True
    max_prompt_length: int = 4096
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 1280 * 28 * 28

    def __post_init__(self) -> None:
        validate_precision_type(self.model_precision, field="QwenVLPipelineConfig.model_precision")


__all__ = ["QwenVLPipelineConfig"]
