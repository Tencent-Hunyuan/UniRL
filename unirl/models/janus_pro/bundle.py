from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from unirl.models.types.bundle import Bundle
from unirl.utils.dtypes import parse_torch_dtype

from .config import JanusProPipelineConfig

logger = logging.getLogger(__name__)


class JanusProBundle(Bundle):
    def __init__(
        self,
        *,
        model: nn.Module,
        processor: Any,
        tokenizer: Any,
        dtype: torch.dtype,
        device: torch.device,
        pretrained_path: str,
    ) -> None:
        super().__init__()
        self.model = model
        # Trainable root for Text+Image -> Text. The vision / generation towers
        # stay on the multimodal wrapper and are frozen by default.
        self.transformer = model.language_model
        self.processor = processor
        self.tokenizer = tokenizer
        self.dtype = dtype
        self.device = device
        self.pretrained_path = pretrained_path

    @classmethod
    def from_config(cls, config: JanusProPipelineConfig) -> "JanusProBundle":
        try:
            from .vendor.models import VLChatProcessor
        except ImportError as exc:
            raise ImportError(
                "JanusProBundle requires the vendored DeepSeek Janus code under "
                "unirl.models.janus_pro.vendor and its runtime dependencies."
            ) from exc

        from transformers import AutoModelForCausalLM

        path = config.pretrained_model_ckpt_path
        device = config.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, str):
            device = torch.device(device)

        dtype = parse_torch_dtype(config.model_precision, field_name="model_precision")

        processor = VLChatProcessor.from_pretrained(path)
        tokenizer = processor.tokenizer
        if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            path,
            trust_remote_code=bool(config.trust_remote_code),
            torch_dtype=dtype,
        ).to(device=device, dtype=dtype)
        model.eval()

        if config.freeze_vision_tower and hasattr(model, "vision_model"):
            model.vision_model.requires_grad_(False)
            logger.info("Froze Janus-Pro vision tower.")
        if config.freeze_aligner and hasattr(model, "aligner"):
            model.aligner.requires_grad_(False)
            logger.info("Froze Janus-Pro understanding aligner.")
        if config.freeze_generation_tower:
            for name in ("gen_vision_model", "gen_aligner", "gen_head", "gen_embed"):
                module = getattr(model, name, None)
                if hasattr(module, "requires_grad_"):
                    module.requires_grad_(False)
            logger.info("Froze Janus-Pro image-generation tower.")

        if config.use_gradient_checkpointing:
            lm = getattr(model, "language_model", None)
            if hasattr(lm, "gradient_checkpointing_enable"):
                lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            elif hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            else:
                logger.warning("Janus-Pro model does not expose gradient_checkpointing_enable; skipping.")

        return cls(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            dtype=dtype,
            device=device,
            pretrained_path=path,
        )

    def trainable_module(self) -> nn.Module:
        return self.transformer

    @property
    def pad_token_id(self) -> int:
        pad_id = getattr(self.processor, "pad_id", None)
        if pad_id is not None:
            return int(pad_id)
        tok_pad = getattr(self.tokenizer, "pad_token_id", None)
        if tok_pad is not None:
            return int(tok_pad)
        eos = getattr(self.tokenizer, "eos_token_id", None)
        if eos is not None:
            return int(eos)
        return 0


__all__ = ["JanusProBundle"]
