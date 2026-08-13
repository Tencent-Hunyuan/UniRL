"""HunyuanVideoBundle -- concrete weights+params holder for HunyuanVideo-1.0."""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn

from unirl.models.types.bundle import Bundle
from unirl.models.types.meta_init import build_meta_init_transformer
from unirl.utils.dtypes import parse_torch_dtype

from .config import HunyuanVideoPipelineConfig


class HunyuanVideoBundle(Bundle):
    """HunyuanVideo-1.0 bundle: transformer + 3D VAE + dual text encoders + scheduler."""

    def __init__(
        self,
        *,
        transformer: nn.Module,
        vae: nn.Module,
        text_encoder: nn.Module,
        tokenizer: Any,
        text_encoder_2: nn.Module,
        tokenizer_2: Any,
        scheduler: Any,
        dtype: torch.dtype,
        device: torch.device,
        pretrained_path: str,
    ) -> None:
        self.transformer = transformer
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.text_encoder_2 = text_encoder_2
        self.tokenizer_2 = tokenizer_2
        self.scheduler = scheduler
        self.dtype = dtype
        self.device = device
        self.pretrained_path = pretrained_path

    @classmethod
    def from_config(cls, config: HunyuanVideoPipelineConfig) -> "HunyuanVideoBundle":
        """Load all HunyuanVideo-1.0 components from a checkpoint."""
        from diffusers import AutoencoderKLHunyuanVideo, HunyuanVideoTransformer3DModel
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
        from transformers import (
            CLIPTextModel,
            CLIPTokenizer,
            LlamaModel,
            LlamaTokenizerFast,
        )

        path = config.pretrained_model_ckpt_path
        vae_path = config.vae_ckpt_path or path
        te_path = config.text_encoder_ckpt_path or path

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
            transformer_config = HunyuanVideoTransformer3DModel.load_config(path, subfolder="transformer")
            transformer, meta_init_state = build_meta_init_transformer(
                lambda: HunyuanVideoTransformer3DModel.from_config(transformer_config), dtype=dtype
            )
        else:
            transformer = HunyuanVideoTransformer3DModel.from_pretrained(
                path, subfolder="transformer", torch_dtype=dtype
            )
            transformer = transformer.to(device=device, dtype=dtype)

        vae = (
            AutoencoderKLHunyuanVideo.from_pretrained(vae_path, subfolder="vae", torch_dtype=vae_dtype)
            .to(device)
            .eval()
        )
        if config.vae_use_tiling:
            vae.enable_tiling()
        vae.requires_grad_(False)

        text_encoder = (
            LlamaModel.from_pretrained(te_path, subfolder="text_encoder", torch_dtype=te_dtype).to(device).eval()
        )
        text_encoder.requires_grad_(False)
        tokenizer = LlamaTokenizerFast.from_pretrained(te_path, subfolder="tokenizer")

        text_encoder_2 = (
            CLIPTextModel.from_pretrained(te_path, subfolder="text_encoder_2", torch_dtype=te_dtype).to(device).eval()
        )
        text_encoder_2.requires_grad_(False)
        tokenizer_2 = CLIPTokenizer.from_pretrained(te_path, subfolder="tokenizer_2")

        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(path, subfolder="scheduler")

        bundle = cls(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            scheduler=scheduler,
            dtype=dtype,
            device=device,
            pretrained_path=path,
        )
        if config.meta_init_transformer:
            bundle._transformer_weights_path = os.path.join(path, "transformer")
            bundle._meta_init_state = meta_init_state
        return bundle


__all__ = ["HunyuanVideoBundle"]
