"""Construction config for the typed Qwen3 AR pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from unirl.config.validation import validate_precision_type


@dataclass
class Qwen3PipelineConfig:
    """Construction args for ``Qwen3Pipeline.from_config``."""

    pretrained_model_ckpt_path: str
    tokenizer_ckpt_path: Optional[str] = None
    trust_remote_code: bool = True

    model_precision: Any = "bf16"
    attn_implementation: Optional[str] = None
    device: Any = None

    autocast_precision: str = "bf16"
    logprob_precision: str = "fp32"

    use_gradient_checkpointing: bool = False

    weight_sync_param_name_prefix: str = "transformer."

    meta_init_transformer: bool = False

    lora_materialization: str = "merged_dense"

    use_lora: bool = False
    lora_target_modules: Optional[List[str]] = None

    use_value_head: bool = False

    system_instruction: Optional[str] = None
    enable_thinking: bool = False
    max_prompt_length: int = 4096

    def __post_init__(self) -> None:
        validate_precision_type(self.model_precision, field="Qwen3PipelineConfig.model_precision")


__all__ = ["Qwen3PipelineConfig"]
