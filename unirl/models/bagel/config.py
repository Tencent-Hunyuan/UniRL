"""Construction config for the Bagel (BAGEL-7B-MoT) pipeline.

Weights+params only — LoRA/FSDP/autocast and per-request sampling knobs live
elsewhere (the latter in ``BagelDiffusionParams``), mirroring the SD3 split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

from unirl.config.validation import validate_precision_type

# LoRA targets = the BAGEL MoT generation-expert projections (flow_grpo
# train_bagel.py:425-433); the only modules trained, und path stays frozen.
BAGEL_MOE_GEN_LORA_TARGETS: Tuple[str, ...] = (
    "self_attn.q_proj_moe_gen",
    "self_attn.k_proj_moe_gen",
    "self_attn.v_proj_moe_gen",
    "self_attn.o_proj_moe_gen",
    "mlp_moe_gen.gate_proj",
    "mlp_moe_gen.up_proj",
    "mlp_moe_gen.down_proj",
)


@dataclass
class BagelPipelineConfig:
    """Construction args for ``BagelBundle.from_config``.

    One MoT transformer (only the gen expert is trained) + one FLUX-style VAE +
    one tokenizer; for T2I the und ViT is disabled (``visual_und=False``).
    """

    pretrained_model_ckpt_path: str
    model_precision: Any = "bf16"
    vae_dtype: Any = None
    device: Any = None

    # Stage-level precision policy. bf16 trajectory matches flow_grpo's BAGEL setup;
    # the SDE log-prob step runs in fp32 inside the vendored kernel.
    autocast_precision: str = "bf16"
    trajectory_precision: str = "bf16"
    logprob_precision: str = "fp32"

    # FlowMatch time-shift (BAGEL uses 3.0). Consumed by the σ-schedule policy.
    shift: float = 3.0

    # Latent geometry (FLUX-style VAE + patchify): downsample 8, patch 2 ⇒ H/16 × W/16
    # grid, each token carrying latent_channels * latent_patch_size**2 values.
    latent_patch_size: int = 2
    max_latent_size: int = 64
    vae_downsample: int = 8
    latent_channels: int = 16

    # Trainable module is model.language_model; used only by dedicated-sync modes.
    weight_sync_param_name_prefix: str = "language_model."

    use_lora: bool = False
    lora_target_modules: Tuple[str, ...] = BAGEL_MOE_GEN_LORA_TARGETS

    def __post_init__(self) -> None:
        validate_precision_type(self.model_precision, field="BagelPipelineConfig.model_precision")
        if not isinstance(self.lora_target_modules, tuple):
            self.lora_target_modules = tuple(self.lora_target_modules)


__all__ = ["BAGEL_MOE_GEN_LORA_TARGETS", "BagelPipelineConfig"]
