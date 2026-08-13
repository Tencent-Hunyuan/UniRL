"""Construction config for the new typed HunyuanImage 3.0 pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

from unirl.config.validation import validate_precision_type


@dataclass
class HunyuanImage3PipelineConfig:
    """Construction args for ``HunyuanImage3Pipeline.from_config``."""

    pretrained_model_ckpt_path: str
    vit_ckpt_path: Optional[str] = None
    vae_dtype: Any = None
    text_encoder_dtype: Any = None
    model_precision: Any = "bf16"
    device: Any = None

    autocast_precision: str = "bf16"
    trajectory_precision: str = "fp16"
    logprob_precision: str = "fp32"

    shift: float = 3.0

    mrope_section: Tuple[int, int, int] = (0, 32, 32)

    guidance_scale: float = 2.5

    # Disable sampling KV cache when rollout log-probs must match replay.
    diffuse_kv_cache: bool = True

    # Prefix rollout weight names because training exposes the bare decoder.
    weight_sync_param_name_prefix: str = "model."

    def __post_init__(self) -> None:
        validate_precision_type(self.model_precision, field="HunyuanImage3PipelineConfig.model_precision")
        if not isinstance(self.mrope_section, tuple):
            self.mrope_section = tuple(self.mrope_section)
        if len(self.mrope_section) != 3:
            raise ValueError(
                f"HunyuanImage3PipelineConfig.mrope_section must be a 3-tuple "
                f"(text_axis, h_axis, w_axis); got {self.mrope_section!r}"
            )


__all__ = ["HunyuanImage3PipelineConfig"]
