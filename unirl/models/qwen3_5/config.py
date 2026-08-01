"""Construction config for the Qwen3.5 VL AR pipeline.

Mirror of :class:`unirl.models.qwen_vl.QwenVLPipelineConfig` (vision tower,
min/max pixels, meta-init) plus the chat-template knobs from
:class:`unirl.models.qwen3.Qwen3PipelineConfig` (``system_instruction``,
``enable_thinking``). Qwen3.5 degrades to text-only when no image is
supplied, so a single config covers both supported recipe shapes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from unirl.config.validation import validate_precision_type


@dataclass
class Qwen3_5PipelineConfig:
    """Construction args for ``Qwen3_5Pipeline.from_config``."""

    pretrained_model_ckpt_path: str
    tokenizer_ckpt_path: Optional[str] = None
    trust_remote_code: bool = False  # Qwen3.5 is natively in transformers >= 5.0

    model_precision: Any = "bf16"
    # HF attention backend for the TRAIN-side model. Qwen3.5 has hybrid
    # attention (3 GDN + 1 full per 4 layers); the GDN layers do not support
    # flash/flex, so packed-varlen replay is NOT safe. Leave None (sdpa) and
    # use padding_replay (the AR stage forces it via _SPARSE_PACKED_ATTN=()).
    attn_implementation: Optional[str] = None
    device: Any = None

    autocast_precision: str = "bf16"
    logprob_precision: str = "fp32"

    use_gradient_checkpointing: bool = False

    weight_sync_param_name_prefix: str = "model."

    use_lora: bool = False
    lora_target_modules: Optional[List[str]] = None

    freeze_vision_tower: bool = True
    max_prompt_length: int = 4096
    min_pixels: int = 256 * 28 * 28
    max_pixels: int = 1280 * 28 * 28

    # Meta-init the transformer (build on the meta device; the backend loads
    # weights after sharding from the checkpoint root) instead of eager
    # ``from_pretrained``. Avoids the per-rank full-model GPU spike. Consumed
    # by FSDPBackend / VeOmniBackend via the stashed ``_transformer_weights_path``.
    meta_init_transformer: bool = False

    system_instruction: Optional[str] = None
    # Chat-template thinking switch; MUST agree with the rollout engine's
    # chat_template_kwargs.enable_thinking or train/rollout prompts diverge.
    enable_thinking: bool = False

    def __post_init__(self) -> None:
        validate_precision_type(self.model_precision, field="Qwen3_5PipelineConfig.model_precision")


__all__ = ["Qwen3_5PipelineConfig"]
