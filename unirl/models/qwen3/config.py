"""Construction config for the typed Qwen3 AR pipeline.

Sibling of :class:`unirl.models.qwen_image.QwenImagePipelineConfig`
and :class:`unirl.models.hunyuan_image3.HunyuanImage3PipelineConfig`.
Carries weights+precision knobs only; LoRA injection, FSDP wrapping,
gradient checkpointing, and offload control all live in
``cfg.training.policies`` (``LoRAPolicy`` / ``FSDPPolicy``) — the bundle
is weights+params only.

Qwen3 is a pure causal LM (no diffusion / VAE / scheduler), so there is
no ``shift`` / ``vae_dtype`` / ``text_encoder_*`` / ``dynamic_shift_*``
field — the hosting engine's :func:`ensure_req_sigmas` is a no-op for
AR-only pipelines because :class:`Qwen3Pipeline.generate` never reads
``req.sigmas``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from unirl.config.validation import validate_precision_type


@dataclass
class Qwen3PipelineConfig:
    """Construction args for ``Qwen3Pipeline.from_config``.

    ``device`` may be runtime-injected by the actor after compose; the
    other fields are set at compose time and read once during pipeline
    construction.
    """

    pretrained_model_ckpt_path: str
    tokenizer_ckpt_path: Optional[str] = None
    trust_remote_code: bool = True

    model_precision: Any = "bf16"
    # HF attention backend for the TRAIN-side model, set on from_pretrained — so it
    # is the model's GLOBAL backend and governs EVERY forward: replay teacher-forcing
    # AND the HF autoregress() decode loop (the *_sglang recipes roll out in SGLang,
    # so only replay is exercised there; flex targets full-sequence forwards and may
    # recompile per step under HF incremental decode).
    # 'flex_attention' makes the packed varlen replay fast: transformers builds a
    # BlockMask from the restarting position_ids and the flex kernel skips the
    # fully-masked cross-sequence blocks (sdpa falls back to the math kernel on
    # packed masks — ~3x slower and memory-bound). None = HF default (sdpa).
    attn_implementation: Optional[str] = None
    device: Any = None

    autocast_precision: str = "bf16"
    logprob_precision: str = "fp32"

    use_gradient_checkpointing: bool = False

    weight_sync_param_name_prefix: str = "transformer."

    # Meta-init the transformer (build on the meta device; the backend loads
    # weights after sharding from the checkpoint root) instead of eager
    # ``from_pretrained``. Avoids the per-rank full-model GPU spike. Consumed by
    # FSDPBackend / VeOmniBackend via the stashed ``_transformer_weights_path``.
    meta_init_transformer: bool = False

    # ``merged_dense`` is the only LoRA-materialization path that survives
    # the SGLang LLM LoRA-pool deadlock under composed (PE) rollouts, so it
    # is the default for Qwen3.
    lora_materialization: str = "merged_dense"

    use_lora: bool = False
    lora_target_modules: Optional[List[str]] = None

    system_instruction: Optional[str] = None
    # Chat-template thinking switch; MUST agree with the rollout engine's
    # chat_template_kwargs.enable_thinking or train/rollout prompts diverge.
    enable_thinking: bool = False

    def __post_init__(self) -> None:
        validate_precision_type(self.model_precision, field="Qwen3PipelineConfig.model_precision")


__all__ = ["Qwen3PipelineConfig"]
