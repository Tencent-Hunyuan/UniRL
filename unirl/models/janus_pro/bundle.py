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
    requires_unwrapped_trainable_root = True

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
            # Importing the vendored package is also what registers
            # MultiModalityCausalLM with AutoModelForCausalLM below.
            from .vendor.models import MultiModalityCausalLM, VLChatProcessor
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
            trust_remote_code=config.trust_remote_code,
            torch_dtype=dtype,
        ).to(device=device, dtype=dtype)
        # Keep this as a backstop for explicit trust_remote_code overrides: remote
        # modeling code must not silently replace the reviewed vendored runtime.
        if not isinstance(model, MultiModalityCausalLM):
            raise TypeError(
                f"JanusProBundle: {path} resolved to {type(model).__module__}.{type(model).__name__}, "
                "not the vendored MultiModalityCausalLM. Re-vendor the checkpoint's modeling code "
                "instead of enabling an unreviewed remote implementation."
            )
        model.eval()

        # Janus stages call the replicated embedding/norm/head children directly,
        # so only decoder blocks may be trainable and FSDP-sharded. LoRA injection
        # happens later in the backend and creates fresh trainable adapter params.
        model.language_model.requires_grad_(False)
        if config.use_lora:
            logger.info("Froze Janus-Pro language-model base weights before LoRA injection.")
        else:
            decoder_blocks = [
                module for module in model.language_model.modules() if type(module).__name__ == "LlamaDecoderLayer"
            ]
            if not decoder_blocks:
                raise RuntimeError("JanusProBundle full fine-tuning found no LlamaDecoderLayer modules.")
            for module in decoder_blocks:
                module.requires_grad_(True)
            logger.info(
                "Enabled Janus-Pro decoder-block full fine-tuning (%d blocks); "
                "embedding, final norm, and LM head remain frozen and replicated.",
                len(decoder_blocks),
            )

        # JanusProPipelineConfig rejects unsupported unfreezing, so these towers
        # are unconditional rather than re-testing validated flags.
        model.vision_model.requires_grad_(False)
        model.aligner.requires_grad_(False)
        for name in ("gen_vision_model", "gen_aligner", "gen_head", "gen_embed"):
            getattr(model, name).requires_grad_(False)
        logger.info("Froze Janus-Pro vision, understanding-aligner, and image-generation towers.")

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
