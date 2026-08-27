"""Construction config for the new typed WAN 2.1 T2V / I2V pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from unirl.config.validation import validate_precision_type


@dataclass
class WAN21PipelineConfig:
    """Construction args for ``WAN21Pipeline.from_config``."""

    pretrained_model_ckpt_path: str
    vae_ckpt_path: Optional[str] = None
    text_encoder_ckpt_path: Optional[str] = None
    image_encoder_ckpt_path: Optional[str] = None
    vae_dtype: Any = None
    text_encoder_dtype: Any = None
    model_precision: Any = "bf16"
    device: Any = None

    autocast_precision: str = "bf16"
    trajectory_precision: str = "fp16"
    logprob_precision: str = "fp32"

    shift: float = 5.0

    # UniPC deterministic-solver contract for dedicated engines; must match the checkpoint scheduler.
    unipc_solver_order: int = 2
    unipc_solver_type: str = "bh2"
    unipc_lower_order_final: bool = True

    max_sequence_length: int = 512

    weight_sync_param_name_prefix: str = "transformer."

    use_lora: bool = False
    lora_target_modules: Optional[List[str]] = None

    meta_init_transformer: bool = False

    load_vae: bool = True

    def __post_init__(self) -> None:
        validate_precision_type(self.model_precision, field="WAN21PipelineConfig.model_precision")


__all__ = ["WAN21PipelineConfig"]
