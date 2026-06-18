from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from unirl.models.types.bundle import Bundle
from unirl.models.types.meta_init import finalize_meta_init, stamp_init_state_restore
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

        if config.meta_init_transformer:
            # Meta-init (FSDP / VeOmni load_sharded path): parameters on the meta
            # device, materialized + loaded by the backend from the checkpoint
            # root after sharding (the embedded ViT is part of the trainable tree,
            # loaded with it — not a separate aux). init_empty_weights(
            # include_buffers=False) keeps buffers/attrs real on CPU: HF rotary
            # inv_freq is a non-persistent buffer computed in __init__ and absent
            # from the checkpoint, so to_empty later clobbers it -> garbage RoPE.
            # Capture it straight off the model and stamp a deferred restore
            # (drained post-load).
            from accelerate import init_empty_weights
            from transformers import AutoConfig

            hf_config = AutoConfig.from_pretrained(path, trust_remote_code=bool(config.trust_remote_code))
            with init_empty_weights(include_buffers=False):
                transformer = Qwen2_5_VLForConditionalGeneration(hf_config)
            stamp_init_state_restore(transformer)
            transformer = finalize_meta_init(transformer, dtype=dtype)
        else:
            transformer = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                path,
                torch_dtype=dtype,
                trust_remote_code=bool(config.trust_remote_code),
            ).to(device)

        # Structural (sets requires_grad / checkpointing flags, no weight
        # access); runs on both builds and persists through to_empty + load.
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
            # VL checkpoints store *.safetensors at the root (no subfolder).
            bundle._transformer_weights_path = path
        return bundle


__all__ = ["QwenVLBundle"]
