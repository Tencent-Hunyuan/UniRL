"""BooguImageBundle — concrete weights+params holder for Boogu-Image."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import torch
import torch.nn as nn

from unirl.models.types.bundle import Bundle
from unirl.models.types.meta_init import build_meta_init_transformer
from unirl.utils.dtypes import parse_torch_dtype

from .config import BooguImagePipelineConfig
from .vendor import BooguImageTransformer2DModel

logger = logging.getLogger(__name__)


def _swap_attention_processors_to_flash(transformer: nn.Module) -> int:
    """Swap the pinned SDPA attention processors for their Flash2Varlen twins; returns the swap count."""
    from .vendor.attention_processor import (
        BooguImageAttnProcessor,
        BooguImageAttnProcessorFlash2Varlen,
        BooguImageDoubleStreamSelfAttnProcessor,
        BooguImageDoubleStreamSelfAttnProcessorFlash2Varlen,
    )

    count = 0
    for module in transformer.modules():
        processor = getattr(module, "processor", None)
        if isinstance(processor, BooguImageDoubleStreamSelfAttnProcessor):
            fresh = BooguImageDoubleStreamSelfAttnProcessorFlash2Varlen(
                head_dim=processor.head_dim,
                num_attention_heads=processor.num_attention_heads,
                num_kv_heads=processor.num_kv_heads,
                qkv_bias=False,
            )
            params = list(processor.parameters())
            if params and not params[0].is_meta:
                fresh.load_state_dict(processor.state_dict())
                fresh = fresh.to(device=params[0].device, dtype=params[0].dtype)
            module.set_processor(fresh)
            count += 1
        elif isinstance(processor, BooguImageAttnProcessor):
            module.set_processor(BooguImageAttnProcessorFlash2Varlen())
            count += 1
    return count


class BooguImageBundle(Bundle):
    """Boogu-Image bundle: vendored DiT + FLUX VAE + Qwen3-VL encoder + processor."""

    def __init__(
        self,
        *,
        transformer: nn.Module,
        vae: Optional[nn.Module],
        text_encoder: Optional[nn.Module],
        processor: Any,
        dtype: torch.dtype,
        device: torch.device,
        pretrained_path: str,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.vae = vae
        self.text_encoder = text_encoder
        self.processor = processor
        self.dtype = dtype
        self.device = device
        self.pretrained_path = pretrained_path

    @classmethod
    def from_config(cls, config: BooguImagePipelineConfig) -> "BooguImageBundle":
        """Load all Boogu-Image components from a HuggingFace-layout checkpoint."""

        import fcntl

        serialize = os.environ.get("DIFFRL_MODEL_LOAD_SERIALIZE", "1") != "0"
        lock_file = open("/tmp/diffrl_model_load.lock", "a+") if serialize else None
        if lock_file is not None:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            return cls._from_config_locked(config)
        finally:
            if lock_file is not None:
                import gc

                gc.collect()
                fcntl.flock(lock_file, fcntl.LOCK_UN)
                lock_file.close()

    @classmethod
    def _from_config_locked(cls, config: BooguImagePipelineConfig) -> "BooguImageBundle":
        from diffusers import AutoencoderKL
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        path = config.pretrained_model_ckpt_path
        vae_path = config.vae_ckpt_path or path
        text_encoder_path = config.text_encoder_ckpt_path or path

        device = config.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, str):
            device = torch.device(device)

        dtype = parse_torch_dtype(config.model_precision, field_name="model_precision")
        vae_raw = config.vae_dtype if config.vae_dtype is not None else config.model_precision
        vae_dtype = parse_torch_dtype(vae_raw, field_name="vae_dtype")
        te_raw = config.text_encoder_dtype if config.text_encoder_dtype is not None else config.model_precision
        te_dtype = parse_torch_dtype(te_raw, field_name="text_encoder_dtype")

        meta_init_state = None
        if config.meta_init_transformer:
            transformer_config = BooguImageTransformer2DModel.load_config(path, subfolder="transformer")
            transformer, meta_init_state = build_meta_init_transformer(
                lambda: BooguImageTransformer2DModel.from_config(transformer_config), dtype=dtype
            )
        else:
            transformer = BooguImageTransformer2DModel.from_pretrained(
                path, subfolder="transformer", torch_dtype=dtype
            ).to(device)

        if bool(getattr(transformer, "enable_teacache", False)) or bool(
            getattr(transformer, "enable_taylorseer", False)
        ):
            raise RuntimeError(
                "BooguImageBundle: transformer loaded with TeaCache/TaylorSeer "
                "enabled — RL rollout/replay must be cache-free."
            )
        if config.attention_backend == "flash2_varlen":
            swapped = _swap_attention_processors_to_flash(transformer)
            logger.info("attention_backend=flash2_varlen: swapped %d processor(s)", swapped)

        vae = None
        if config.load_vae:
            vae = AutoencoderKL.from_pretrained(vae_path, subfolder="vae", torch_dtype=vae_dtype).to(device).eval()
            vae.requires_grad_(False)

        text_encoder = None
        processor = None
        if config.load_text_encoder:
            wrapper = Qwen3VLForConditionalGeneration.from_pretrained(
                text_encoder_path, subfolder="mllm", torch_dtype=te_dtype
            )
            text_encoder = wrapper.model if hasattr(wrapper, "lm_head") else wrapper
            del wrapper
            text_encoder = text_encoder.to(device).eval()
            text_encoder.requires_grad_(False)

            processor = AutoProcessor.from_pretrained(text_encoder_path, subfolder="processor")

        bundle = cls(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            processor=processor,
            dtype=dtype,
            device=device,
            pretrained_path=path,
        )
        if config.meta_init_transformer:
            bundle._transformer_weights_path = os.path.join(path, "transformer")
            bundle._meta_init_state = meta_init_state
        return bundle


__all__ = ["BooguImageBundle"]
