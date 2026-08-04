from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from unirl.models.types.bundle import Bundle
from unirl.models.types.meta_init import build_meta_init_transformer
from unirl.utils.dtypes import parse_torch_dtype

from .config import QwenVLPipelineConfig

logger = logging.getLogger(__name__)


class QwenVLBundle(Bundle):
    def __init__(
        self,
        *,
        transformer: nn.Module,
        processor: Any,
        tokenizer: Any,
        dtype: torch.dtype,
        device: torch.device,
        pretrained_path: str,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.processor = processor
        self.tokenizer = tokenizer
        self.dtype = dtype
        self.device = device
        self.pretrained_path = pretrained_path

    @classmethod
    def from_config(cls, config: QwenVLPipelineConfig) -> "QwenVLBundle":
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        path = config.pretrained_model_ckpt_path

        device = config.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, str):
            device = torch.device(device)

        dtype = parse_torch_dtype(config.model_precision, field_name="model_precision")

        load_kwargs = {}
        if getattr(config, "attn_implementation", None):
            load_kwargs["attn_implementation"] = str(config.attn_implementation)

        meta_init_state = None
        if config.meta_init_transformer:
            # Restore non-persistent RoPE buffers after meta initialization.
            from transformers import AutoConfig

            hf_config = AutoConfig.from_pretrained(path, trust_remote_code=bool(config.trust_remote_code))
            transformer, meta_init_state = build_meta_init_transformer(
                lambda: Qwen2_5_VLForConditionalGeneration(hf_config), dtype=dtype
            )
        else:
            transformer = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                path,
                torch_dtype=dtype,
                trust_remote_code=bool(config.trust_remote_code),
                **load_kwargs,
            ).to(device)

        if config.freeze_vision_tower:
            transformer.model.visual.requires_grad_(False)
            logger.info("Froze vision tower (%s parameters).", sum(1 for _ in transformer.model.visual.parameters()))

        if config.use_gradient_checkpointing:
            if hasattr(transformer, "gradient_checkpointing_enable"):
                transformer.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            else:
                logger.warning(
                    "QwenVL transformer %s does not expose gradient_checkpointing_enable; skipping.",
                    type(transformer).__name__,
                )

        processor = AutoProcessor.from_pretrained(
            path,
            trust_remote_code=bool(config.trust_remote_code),
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
        )

        tokenizer = processor.tokenizer
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        bundle = cls(
            transformer=transformer,
            processor=processor,
            tokenizer=tokenizer,
            dtype=dtype,
            device=device,
            pretrained_path=path,
        )
        if config.meta_init_transformer:
            bundle._transformer_weights_path = path
            bundle._meta_init_state = meta_init_state
        return bundle


__all__ = ["QwenVLBundle"]
