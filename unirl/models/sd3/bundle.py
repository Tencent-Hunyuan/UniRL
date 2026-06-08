"""SD3Bundle — concrete weights+params holder for SD3.

Implements the empty :class:`Bundle` Protocol. Pure container of the
modules SD3 ships with: 1× transformer, 1× VAE, 3× text encoders +
tokenizers (CLIP1, CLIP2, T5), 1× scheduler. No LoRA injection, FSDP
wrap, adapter switching, autocast helpers, or weight‑sync logic — those
are lifecycle concerns owned outside the bundle.

Use :meth:`SD3Bundle.from_config` to load a HuggingFace checkpoint.
:meth:`SD3Bundle.no_split_modules` exposes the FSDP wrap‑policy hint
that future training backends will read directly off the bundle.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from unirl.models.types.bundle import Bundle
from unirl.utils.dtypes import parse_torch_dtype

from .config import SD3PipelineConfig


class SD3Bundle(Bundle):
    """SD3-family bundle: transformer + VAE + 3 text encoders + scheduler."""

    def __init__(
        self,
        *,
        transformer: nn.Module,
        vae: nn.Module,
        text_encoder: nn.Module,
        text_encoder_2: nn.Module,
        text_encoder_3: nn.Module,
        tokenizer: Any,
        tokenizer_2: Any,
        tokenizer_3: Any,
        scheduler: Any,
        dtype: torch.dtype,
        device: torch.device,
        pretrained_path: str,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.vae = vae
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.text_encoder_3 = text_encoder_3
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.tokenizer_3 = tokenizer_3
        self.scheduler = scheduler
        self.dtype = dtype
        self.device = device
        self.pretrained_path = pretrained_path

    @classmethod
    def from_config(cls, config: SD3PipelineConfig) -> "SD3Bundle":
        """Load all SD3 components from a HuggingFace checkpoint."""
        from diffusers import AutoencoderKL, SD3Transformer2DModel
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
        from transformers import (
            CLIPTextModelWithProjection,
            CLIPTokenizer,
            T5EncoderModel,
            T5TokenizerFast,
        )

        path = config.pretrained_model_ckpt_path
        device = config.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, str):
            device = torch.device(device)

        dtype = parse_torch_dtype(config.model_precision, field_name="model_precision")
        vae_raw = config.vae_dtype if config.vae_dtype is not None else config.model_precision
        vae_dtype = parse_torch_dtype(vae_raw, field_name="vae_dtype")
        te_raw = config.text_encoder_dtype if config.text_encoder_dtype is not None else config.model_precision
        te_dtype = parse_torch_dtype(te_raw, field_name="text_encoder_dtype")

        transformer = SD3Transformer2DModel.from_pretrained(path, subfolder="transformer", torch_dtype=dtype).to(device)

        vae = AutoencoderKL.from_pretrained(path, subfolder="vae", torch_dtype=vae_dtype).to(device).eval()
        vae.requires_grad_(False)

        text_encoder = (
            CLIPTextModelWithProjection.from_pretrained(path, subfolder="text_encoder", torch_dtype=te_dtype)
            .to(device)
            .eval()
        )
        text_encoder.requires_grad_(False)
        text_encoder_2 = (
            CLIPTextModelWithProjection.from_pretrained(path, subfolder="text_encoder_2", torch_dtype=te_dtype)
            .to(device)
            .eval()
        )
        text_encoder_2.requires_grad_(False)
        text_encoder_3 = (
            T5EncoderModel.from_pretrained(path, subfolder="text_encoder_3", torch_dtype=te_dtype).to(device).eval()
        )
        text_encoder_3.requires_grad_(False)

        tokenizer = CLIPTokenizer.from_pretrained(path, subfolder="tokenizer")
        tokenizer_2 = CLIPTokenizer.from_pretrained(path, subfolder="tokenizer_2")
        tokenizer_3 = T5TokenizerFast.from_pretrained(path, subfolder="tokenizer_3")

        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(path, subfolder="scheduler")

        return cls(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            text_encoder_3=text_encoder_3,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            tokenizer_3=tokenizer_3,
            scheduler=scheduler,
            dtype=dtype,
            device=device,
            pretrained_path=path,
        )


__all__ = ["SD3Bundle"]
