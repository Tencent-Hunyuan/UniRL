"""Construction config for the Bagel (BAGEL-7B-MoT) pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

from unirl.config.validation import validate_precision_type

BAGEL_MOE_GEN_LORA_TARGETS: Tuple[str, ...] = (
    "self_attn.q_proj_moe_gen",
    "self_attn.k_proj_moe_gen",
    "self_attn.v_proj_moe_gen",
    "self_attn.o_proj_moe_gen",
    "mlp_moe_gen.gate_proj",
    "mlp_moe_gen.up_proj",
    "mlp_moe_gen.down_proj",
)

BAGEL_UND_LORA_TARGETS: Tuple[str, ...] = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


@dataclass
class BagelPipelineConfig:
    """Construction args for ``BagelBundle.from_config``."""

    pretrained_model_ckpt_path: str
    model_precision: Any = "bf16"
    vae_dtype: Any = None
    device: Any = None

    autocast_precision: str = "bf16"
    trajectory_precision: str = "bf16"
    logprob_precision: str = "fp32"

    shift: float = 3.0

    latent_patch_size: int = 2
    max_latent_size: int = 64
    vae_downsample: int = 8
    latent_channels: int = 16

    enable_vit: bool = False

    weight_sync_param_name_prefix: str = "language_model."

    use_lora: bool = False
    lora_target_modules: Tuple[str, ...] = BAGEL_MOE_GEN_LORA_TARGETS

    cache_t2i_contexts: bool = True
    context_cache_size: int = 32

    def __post_init__(self) -> None:
        validate_precision_type(self.model_precision, field="BagelPipelineConfig.model_precision")
        if not isinstance(self.lora_target_modules, tuple):
            self.lora_target_modules = tuple(self.lora_target_modules)


__all__ = ["BAGEL_MOE_GEN_LORA_TARGETS", "BAGEL_UND_LORA_TARGETS", "BagelPipelineConfig"]
