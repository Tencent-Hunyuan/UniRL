"""QwenImage21Bundle - weights for Qwen-Image-2.1; the VAE is 5D RGBA, latents ``[B, 64, T=1, H/16, W/16]``."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from unirl.models.types.bundle import Bundle
from unirl.utils.dtypes import parse_torch_dtype

from .config import QwenImage21PipelineConfig


class QwenImage21Bundle(Bundle):
    """Qwen-Image-2.1 bundle: single-stream transformer + RGBA VAE + Qwen3-VL text encoder + processor."""

    def __init__(
        self,
        *,
        transformer: nn.Module,
        vae: nn.Module,
        text_encoder: nn.Module,
        processor: Any,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.vae = vae
        self.text_encoder = text_encoder
        self.processor = processor
        self.device = device

    @classmethod
    def from_config(cls, config: QwenImage21PipelineConfig) -> "QwenImage21Bundle":
        """Load all Qwen-Image-2.1 components from a diffusers-layout checkpoint."""
        from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

        from .vendor.autoencoder_kl_qwenimage21 import AutoencoderKLQwenImage21
        from .vendor.transformer_qwenimage21 import QwenImage21Transformer2DModel

        path = config.pretrained_model_ckpt_path
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = parse_torch_dtype(config.model_precision, field_name="model_precision")

        transformer = QwenImage21Transformer2DModel.from_pretrained(
            path, subfolder="transformer", torch_dtype=dtype
        ).to(device)
        vae = AutoencoderKLQwenImage21.from_pretrained(path, subfolder="vae", torch_dtype=torch.float32)
        vae = vae.to(device).eval().requires_grad_(False)
        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            path, subfolder="text_encoder", torch_dtype=dtype
        )
        text_encoder = text_encoder.to(device).eval().requires_grad_(False)
        processor = Qwen3VLProcessor.from_pretrained(path, subfolder="processor")
        return cls(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            processor=processor,
            device=device,
        )


__all__ = ["QwenImage21Bundle"]
