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
    device: Any = None

    autocast_precision: str = "bf16"
    logprob_precision: str = "fp32"

    use_gradient_checkpointing: bool = False

    weight_sync_param_name_prefix: str = "transformer."

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
