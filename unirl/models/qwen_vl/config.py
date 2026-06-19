from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from unirl.config.validation import validate_precision_type


@dataclass
class QwenVLPipelineConfig:
    pretrained_model_ckpt_path: str
    tokenizer_ckpt_path: Optional[str] = None
    trust_remote_code: bool = True

    model_precision: Any = "bf16"
    # HF attention backend for the TRAIN-side model, set on from_pretrained — so it
    # is the model's GLOBAL backend and governs EVERY forward: replay teacher-forcing
    # AND the HF autoregress() decode loop (the *_sglang recipes roll out in SGLang,
    # so only replay is exercised there).
    # Qwen2.5-VL has NO flex_attention support, so packed-varlen replay needs a
    # FlashAttention backend ('flash_attention_4' for the pinned flash-attn-4, or
    # 'flash_attention_2'/'flash_attention_3' if those packages are installed):
    # transformers derives per-sequence cu_seqlens from the restarting position_ids
    # so cross-sequence blocks are skipped. None = HF default (sdpa) -> dense replay.
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
    # ``from_pretrained``. Avoids the per-rank full-model GPU spike. Consumed by
    # FSDPBackend / VeOmniBackend via the stashed ``_transformer_weights_path``.
    meta_init_transformer: bool = False

    def __post_init__(self) -> None:
        validate_precision_type(self.model_precision, field="QwenVLPipelineConfig.model_precision")


__all__ = ["QwenVLPipelineConfig"]
