"""Construction config for the new typed HunyuanImage 3.0 pipeline.

Mirrors :class:`unirl.models.sd3.config.SD3PipelineConfig` minus
SD3-specific knobs and plus HunyuanImage3-specific ones. The bundle is
weights+params only; LoRA / FSDP wrap / autocast / weight sync live
outside (in the training and rollout actors).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

from unirl.config.validation import validate_precision_type


@dataclass
class HunyuanImage3PipelineConfig:
    """Construction args for ``HunyuanImage3Pipeline.from_config``.

    The HunyuanImage3 backbone is a single shared MoE transformer that
    operates in either ``mode="gen_text"`` (AR) or ``mode="gen_image"`` (DiT)
    on the same weights. The bundle owns one transformer + one SigLIP2 ViT
    + one 3D-VAE + one tokenizer + one scheduler.

    ``device`` may be runtime-injected by the actor after compose; the
    other fields are set at compose time and read once during pipeline
    construction.
    """

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
