from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from unirl.config.validation import validate_precision_type


@dataclass
class JanusProPipelineConfig:
    pretrained_model_ckpt_path: str

    model_precision: Any = "bf16"
    device: Any = None

    autocast_precision: str = "bf16"
    logprob_precision: str = "fp32"

    use_gradient_checkpointing: bool = False
    use_lora: bool = False
    cache_vision_embeddings: bool = False
    rollout_keep_params_unsharded: bool = False

    max_prompt_length: int = 4096

    def __post_init__(self) -> None:
        validate_precision_type(self.model_precision, field="JanusProPipelineConfig.model_precision")
        if self.max_prompt_length < 1:
            raise ValueError(f"JanusProPipelineConfig.max_prompt_length must be >= 1; got {self.max_prompt_length}.")


__all__ = ["JanusProPipelineConfig"]
