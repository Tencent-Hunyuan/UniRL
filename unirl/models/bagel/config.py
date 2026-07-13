"""Construction config for the Bagel (BAGEL-7B-MoT) pipeline.

Weights+params only: LoRA injection, FSDP wrap, and autocast lifecycle live
outside the bundle (in the train / rollout actors), mirroring
:class:`unirl.models.sd3.config.SD3PipelineConfig` and
:class:`unirl.models.hunyuan_image3.config.HunyuanImage3PipelineConfig`.

Per-request sampling knobs (CFG scales, ``noise_level``, ``num_timesteps``,
SDE window) are intentionally NOT here — they live in ``BagelDiffusionParams``
consumed by the diffusion stage, the same split SD3 uses between its config and
``DiffusionSamplingParams``.

Fixed BAGEL topology constants (``qk_norm``, ``tie_word_embeddings``,
``layer_module``, ``connector_act``) are not exposed here — they are not tunable
for this checkpoint and live as constants in the bundle. ``visual_und`` IS
exposed (as ``enable_vit``): the und ViT tower is optional and only needed for
image-input tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

from unirl.config.validation import validate_precision_type

# LoRA targets = the BAGEL MoT generation-expert projections (flow_grpo
# train_bagel.py:425-433). These are the only modules trained in the gen-only
# FlowGRPO setup; the und (understanding) path stays frozen.
BAGEL_MOE_GEN_LORA_TARGETS: Tuple[str, ...] = (
    "self_attn.q_proj_moe_gen",
    "self_attn.k_proj_moe_gen",
    "self_attn.v_proj_moe_gen",
    "self_attn.o_proj_moe_gen",
    "mlp_moe_gen.gate_proj",
    "mlp_moe_gen.up_proj",
    "mlp_moe_gen.down_proj",
)

# LoRA targets for TEXT-out RL (t2t / i2t / it2t): the und/base projections the
# MoT routes text (and ViT) tokens through. The gen experts stay frozen.
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
    """Construction args for ``BagelBundle.from_config``.

    BAGEL-7B-MoT is a single MoT transformer that runs both the und
    (understanding) and gen (image-generation) paths on shared weights; only the
    gen expert is trained. The bundle owns one transformer + one FLUX-style VAE +
    one tokenizer; for pure text-to-image the und ViT stays disabled
    (``enable_vit=False`` → ``visual_und=False``), image-input tasks opt in.

    ``device`` may be runtime-injected by the actor after compose; the other
    fields are read once during bundle construction.
    """

    pretrained_model_ckpt_path: str
    model_precision: Any = "bf16"
    vae_dtype: Any = None
    device: Any = None

    # Stage-level precision / numerical policy (operator/runtime knobs, not
    # per-request shape). bf16 trajectory matches flow_grpo's BAGEL setup; the
    # SDE log-prob step itself runs in fp32 inside the vendored kernel.
    autocast_precision: str = "bf16"
    trajectory_precision: str = "bf16"
    logprob_precision: str = "fp32"

    # FlowMatch time-shift (BAGEL uses 3.0). Consumed by the σ-schedule policy.
    shift: float = 3.0

    # Latent geometry (FLUX-style VAE + patchify). With downsample 8 and patch 2,
    # the effective latent grid is H/16 x W/16 and each token carries
    # latent_channels * latent_patch_size**2 values. Needed by latent_shape and
    # the VAE decode/unpatchify.
    latent_patch_size: int = 2
    max_latent_size: int = 64
    vae_downsample: int = 8
    latent_channels: int = 16

    # Load the und ViT tower (``visual_und=True``): SiglipVisionConfig from
    # ``<ckpt>/vit_config.json`` + SiglipVisionModel/connector/vit_pos_embed
    # weights from the same ``ema.safetensors``. Required for image-INPUT tasks
    # (it2i editing / i2t / it2t); the default False keeps the T2I-only
    # construction byte-identical to before.
    enable_vit: bool = False

    # Trainable module is ``model.language_model`` (the MoT). Used only by
    # dedicated-sync modes; trainside performs no weight sync.
    weight_sync_param_name_prefix: str = "language_model."

    use_lora: bool = False
    lora_target_modules: Tuple[str, ...] = BAGEL_MOE_GEN_LORA_TARGETS

    # T2I prompt-context cache: dedup the prompt prefill across the N GRPO siblings
    # (BAGEL navit bs=1 sends them one per generate call). Safe ONLY when the
    # und/prefill path is frozen (gen-only LoRA); BagelPipeline additionally
    # auto-disables it at runtime if any non-gen param is trainable, so this flag
    # is just the opt-out / cache-size knob.
    cache_t2i_contexts: bool = True
    context_cache_size: int = 32

    def __post_init__(self) -> None:
        validate_precision_type(self.model_precision, field="BagelPipelineConfig.model_precision")
        if not isinstance(self.lora_target_modules, tuple):
            self.lora_target_modules = tuple(self.lora_target_modules)


__all__ = ["BAGEL_MOE_GEN_LORA_TARGETS", "BAGEL_UND_LORA_TARGETS", "BagelPipelineConfig"]
