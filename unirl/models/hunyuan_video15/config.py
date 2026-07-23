"""Construction config for the typed HunyuanVideo-1.5 pipeline.

Sibling of :class:`unirl.models.wan21.WAN21PipelineConfig`.
Carries weights+precision knobs only; LoRA injection, FSDP wrapping,
gradient checkpointing, and offload control all live in
``cfg.training.policies`` (``LoRAPolicy`` / ``FSDPPolicy``) — the bundle
is weights+params only.

Three knobs are HunyuanVideo-1.5-specific (vs the SD3 / Qwen-Image /
WAN21 sibling configs):

- ``text_encoder_2_ckpt_path``: ByT5 glyph encoder lives in a separate
  HuggingFace subfolder (``text_encoder_2`` / ``tokenizer_2``); recipes
  that load Qwen-VL + ByT5 from the same checkpoint dir can leave this
  ``None`` and the bundle falls back to ``pretrained_model_ckpt_path``.
- ``image_encoder_ckpt_path``: SigLIP vision encoder for I2V (only used
  when ``load_vision_encoder=True``).
- ``mllm_*`` / ``byt5_max_length`` / ``vision_*``: shape parameters
  copied verbatim from the upstream HunyuanVideo15 pipeline; tweaking
  these without also retraining the transformer breaks the model.

``shift`` defaults to 5.0 (the upstream HunyuanVideo-1.5 default; SD3
uses 3.0). Unlike Qwen-Image, HunyuanVideo-1.5 uses **static** shift,
not dynamic-mu — :class:`FlowMatchSchedulePolicy.from_pretrained` will
read ``use_dynamic_shifting=False`` from the checkpoint's
``scheduler_config.json`` and stick to the static branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from unirl.config.validation import validate_precision_type


@dataclass
class HunyuanVideo15PipelineConfig:
    """Construction args for ``HunyuanVideo15Pipeline.from_config``.

    ``device`` may be runtime-injected by the actor after compose; the
    other fields are set at compose time and read once during pipeline
    construction.
    """

    pretrained_model_ckpt_path: str
    vae_ckpt_path: Optional[str] = None
    text_encoder_ckpt_path: Optional[str] = None
    text_encoder_2_ckpt_path: Optional[str] = None
    image_encoder_ckpt_path: Optional[str] = None
    vae_dtype: Any = None
    text_encoder_dtype: Any = None
    model_precision: Any = "bf16"
    device: Any = None

    # Stage-level precision / numerical policy.
    autocast_precision: str = "bf16"
    trajectory_precision: str = "fp16"
    logprob_precision: str = "fp32"

    # FlowMatchSchedulePolicy shift — static (HunyuanVideo-1.5 does not
    # use dynamic shifting). Default 5.0 mirrors the upstream sampler.
    shift: float = 5.0

    # Trainer-side policy wraps the bare DiT, while vLLM-Omni loads it
    # under the pipeline's ``transformer.*`` namespace.
    weight_sync_param_name_prefix: str = "transformer."

    # Meta-init the transformer (build on the meta device; the backend loads
    # weights after sharding) instead of eager ``from_pretrained``. Avoids the
    # per-rank full-model GPU spike. Consumed by FSDPBackend / VeOmniBackend via
    # the stashed ``_transformer_weights_path``.
    meta_init_transformer: bool = False

    # VAE latent channel count. ``None`` lets both the driver and the
    # stage fall back to ``HunyuanVideo15DiffusionStage.DEFAULT_LATENT_CHANNELS``
    # (32) which matches the ``hunyuanvideo-community/HunyuanVideo-1.5-Diffusers``
    # variant in production. The stage will still cross-check against
    # ``vae.config.latent_channels`` at construction; the explicit
    # config-side value is the only handle the driver has before the
    # bundle is loaded, so set it here whenever the checkpoint's VAE
    # differs from the diffusers-community 32-channel default.
    latent_channels: Optional[int] = None

    # ------------------------------------------------------------------
    # Text-encoder shape parameters (copied verbatim from upstream)
    # ------------------------------------------------------------------
    # Qwen2.5-VL MLLM: chat-template prefix + tokenizer cap.
    mllm_max_length: int = 1000
    # Drop the chat-template prefix tokens after the encoder forward; this
    # value is the prefix length on the standard system prompt baked into
    # ``text_embed.PROMPT_TEMPLATE_SYSTEM_MESSAGE``.
    mllm_crop_start: int = 108
    # Use the (skip_layers + 1)-th-from-last hidden state, not the last
    # layer's output — matches the upstream pipeline.
    mllm_skip_layers: int = 2
    # ByT5 glyph encoder token cap (much shorter; only quoted text snippets).
    byt5_max_length: int = 256

    # ------------------------------------------------------------------
    # SigLIP vision-encoder shape parameters (used only when I2V lands)
    # ------------------------------------------------------------------
    # For T2V, the bundle emits a zero placeholder of shape
    # ``[B, vision_num_semantic_tokens, vision_states_dim]``; the
    # transformer cross-attends to it but the zero content is a no-op.
    vision_num_semantic_tokens: int = 729
    vision_states_dim: int = 1152
    # Set False in pure-T2V recipes to free ~1.6 GB of GPU memory.
    # When False, the bundle still emits the zero ``image_embeds``
    # placeholder so the transformer's input signature is satisfied; only
    # the SigLIP module itself is skipped.
    load_vision_encoder: bool = False

    # Trainer-side 3D VAE. False for separate-engine recipes (engine owns decode).
    load_vae: bool = True

    # LoRA hints for rollout-side engines (e.g. ``sglang``). Mirrors
    # SD3 / Qwen-Image config; the trainer-side LoRA injection lives in
    # ``cfg.training.policies`` via LoRAPolicy.
    use_lora: bool = False
    lora_target_modules: Optional[List[str]] = None

    def __post_init__(self) -> None:
        validate_precision_type(self.model_precision, field="HunyuanVideo15PipelineConfig.model_precision")


__all__ = ["HunyuanVideo15PipelineConfig"]
