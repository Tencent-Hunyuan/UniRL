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
        pad_token_id: int,
        cache_vision_embeddings: bool,
        rollout_keep_params_unsharded: bool,
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
        self.pad_token_id = pad_token_id
        self.cache_vision_embeddings = cache_vision_embeddings
        self.rollout_keep_params_unsharded = rollout_keep_params_unsharded

    @classmethod
    def from_config(cls, config: JanusProPipelineConfig) -> "JanusProBundle":
        # Importing the vendor registers MultiModalityCausalLM with Transformers.
        from transformers import AutoModelForCausalLM

        from .vendor.models import MultiModalityCausalLM, VLChatProcessor

        path = config.pretrained_model_ckpt_path
        device = config.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, str):
            device = torch.device(device)

        dtype = parse_torch_dtype(config.model_precision, field_name="model_precision")

        processor = VLChatProcessor.from_pretrained(path)
        tokenizer = processor.tokenizer
        if processor.pad_id is None:
            raise ValueError(f"JanusProBundle: {path} tokenizer is missing the Janus pad token.")

        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype).to(device=device, dtype=dtype)
        if not isinstance(model, MultiModalityCausalLM):
            raise TypeError(
                f"JanusProBundle: {path} resolved to {type(model).__module__}.{type(model).__name__}, "
                "not the vendored MultiModalityCausalLM."
            )
        # The train stack toggles only language_model; frozen sibling towers stay in eval mode.
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

        model.vision_model.requires_grad_(False)
        model.aligner.requires_grad_(False)
        for module in (model.gen_vision_model, model.gen_aligner, model.gen_head, model.gen_embed):
            module.requires_grad_(False)
        logger.info("Froze Janus-Pro vision, understanding-aligner, and image-generation towers.")

        if config.use_gradient_checkpointing:
            model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        return cls(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            dtype=dtype,
            device=device,
            pretrained_path=path,
            pad_token_id=processor.pad_id,
            cache_vision_embeddings=config.cache_vision_embeddings,
            rollout_keep_params_unsharded=config.rollout_keep_params_unsharded,
        )


__all__ = ["JanusProBundle"]
