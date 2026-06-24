"""Construction config for the typed HunyuanVideo-1.0 pipeline.

Sibling of :class:`unirl.models.hunyuan_video15.HunyuanVideo15PipelineConfig`.
Carries weights+precision knobs only; LoRA injection, FSDP wrapping,
gradient checkpointing, and offload control all live in
``cfg.training.policies`` (``LoRAPolicy`` / ``FSDPPolicy``) -- the bundle
is weights+params only.

HunyuanVideo-1.0-specific vs 1.5:

- No ``text_encoder_2_ckpt_path`` needed (CLIP is always co-located with
  the main checkpoint under ``text_encoder_2/`` + ``tokenizer_2/``).
- No ``byt5_*``, ``mllm_*``, ``vision_*`` fields (1.0 uses LLaMA + CLIP,
  not Qwen2.5-VL + ByT5 + SigLIP).
- ``llama_max_length`` / ``clip_max_length`` / ``crop_start`` shape the
  text encoding (LLaMA prompt template crops the first ``crop_start``
  tokens after encoding).
- ``guidance_embeds=True`` on the transformer -- guidance scale is passed
  as a tensor, NOT as CFG stacking.

``shift`` defaults to 5.0 (the upstream HunyuanVideo default). Static
shift only (``use_dynamic_shifting=False``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from unirl.config.validation import validate_precision_type


@dataclass
class HunyuanVideoPipelineConfig:
    """Construction args for ``HunyuanVideoPipeline.from_config``.

    ``device`` may be runtime-injected by the actor after compose; the
    other fields are set at compose time and read once during pipeline
    construction.
    """

    pretrained_model_ckpt_path: str
    vae_ckpt_path: Optional[str] = None
    text_encoder_ckpt_path: Optional[str] = None
    vae_dtype: Any = None
    text_encoder_dtype: Any = None
    model_precision: Any = "bf16"
    device: Any = None

    # Stage-level precision / numerical policy.
    autocast_precision: str = "bf16"
    trajectory_precision: str = "fp16"
    logprob_precision: str = "fp32"

    # FlowMatchSchedulePolicy shift -- static (HunyuanVideo-1.0 does not
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
    # stage fall back to ``HunyuanVideoDiffusionStage.DEFAULT_LATENT_CHANNELS``
    # (16) which matches the HunyuanVideo checkpoint. The stage will still
    # cross-check against ``vae.config.latent_channels`` at construction;
    # the explicit config-side value is the only handle the driver has
    # before the bundle is loaded.
    latent_channels: Optional[int] = None

    # ------------------------------------------------------------------
    # Text-encoder shape parameters
    # ------------------------------------------------------------------
    # LLaMA: tokenizer max_length (after the prompt template prefix is
    # prepended). Output is cropped by ``crop_start`` tokens.
    llama_max_length: int = 256
    # Number of prefix tokens to crop after the LLaMA encoder forward
    # (the prompt template system header length).
    crop_start: int = 95
    # CLIP: standard max_length for CLIPTokenizer.
    clip_max_length: int = 77

    # LoRA hints for rollout-side engines (e.g. ``sglang``). Mirrors
    # SD3 / HV15 config; the trainer-side LoRA injection lives in
    # ``cfg.training.policies`` via LoRAPolicy.
    use_lora: bool = False
    lora_target_modules: Optional[List[str]] = None

    def __post_init__(self) -> None:
        validate_precision_type(self.model_precision, field="HunyuanVideoPipelineConfig.model_precision")


__all__ = ["HunyuanVideoPipelineConfig"]
