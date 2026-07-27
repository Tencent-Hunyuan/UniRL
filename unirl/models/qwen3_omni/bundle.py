"""Weights, processor, and tokenizer for the Qwen3-Omni thinker."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from unirl.models.types.bundle import Bundle
from unirl.utils.dtypes import parse_torch_dtype

from .config import Qwen3OmniPipelineConfig

logger = logging.getLogger(__name__)


class Qwen3OmniBundle(Bundle):
    """Qwen3-Omni thinker bundle: thinker transformer + processor + tokenizer."""

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
    def from_config(cls, config: Qwen3OmniPipelineConfig) -> "Qwen3OmniBundle":
        from transformers import AutoConfig, AutoProcessor
        from transformers.models.qwen3_omni_moe import Qwen3OmniMoeThinkerForConditionalGeneration

        path = config.pretrained_model_ckpt_path

        device = config.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, str):
            device = torch.device(device)

        dtype = parse_torch_dtype(config.model_precision, field_name="model_precision")

        # Build the standalone thinker from its nested config.
        full_cfg = AutoConfig.from_pretrained(path, trust_remote_code=bool(config.trust_remote_code))
        thinker_cfg = full_cfg.thinker_config

        if config.meta_init_transformer:
            # FSDP sharded loading requires remapping checkpoint ``thinker.`` keys.
            raise NotImplementedError(
                "Qwen3OmniBundle: meta_init_transformer=True is not yet supported "
                "(needs a strip-'thinker.' key remap in the sharded loader). Use "
                "meta_init_transformer=False (eager from_pretrained auto-strips the "
                "prefix via base_model_prefix='thinker')."
            )

        load_kwargs = {}
        if getattr(config, "attn_implementation", None):
            load_kwargs["attn_implementation"] = str(config.attn_implementation)

        # ``base_model_prefix`` maps full-checkpoint keys to the standalone thinker.
        transformer = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
            path,
            config=thinker_cfg,
            dtype=dtype,
            low_cpu_mem_usage=True,
            **load_kwargs,
        ).to(device)

        # Freeze the top-level encoders while leaving the decoder trainable.
        if config.freeze_vision_tower and hasattr(transformer, "visual"):
            transformer.visual.requires_grad_(False)
            logger.info("Froze thinker vision tower (%d params).", sum(1 for _ in transformer.visual.parameters()))
        if config.freeze_audio_tower and hasattr(transformer, "audio_tower"):
            transformer.audio_tower.requires_grad_(False)
            logger.info("Froze thinker audio tower (%d params).", sum(1 for _ in transformer.audio_tower.parameters()))

        if config.use_gradient_checkpointing:
            if hasattr(transformer, "gradient_checkpointing_enable"):
                transformer.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            else:
                logger.warning(
                    "Qwen3-Omni thinker %s does not expose gradient_checkpointing_enable; skipping.",
                    type(transformer).__name__,
                )

        # Load multimodal preprocessing assets from the checkpoint root.
        processor = AutoProcessor.from_pretrained(
            path,
            trust_remote_code=bool(config.trust_remote_code),
        )
        tokenizer = getattr(processor, "tokenizer", None) or processor
        if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token

        return cls(
            transformer=transformer,
            processor=processor,
            tokenizer=tokenizer,
            dtype=dtype,
            device=device,
            pretrained_path=path,
        )


__all__ = ["Qwen3OmniBundle"]
