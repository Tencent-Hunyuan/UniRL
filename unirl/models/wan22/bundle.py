"""WAN22Bundle — concrete weights+params holder for WAN 2.2 T2V / I2V."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from unirl.models.types.bundle import Bundle
from unirl.models.wan21.bundle import WAN21Bundle
from unirl.utils.dtypes import parse_torch_dtype

from .config import WAN22PipelineConfig


class WanDualTransformer(nn.Module):
    """Thin ``nn.Module`` wrapper presenting two WAN transformers as one model."""

    def __init__(self, high_noise: nn.Module, low_noise: nn.Module) -> None:
        super().__init__()
        self.high_noise = high_noise
        self.low_noise = low_noise

    def forward(self, *, use_high_noise: bool, **kwargs: Any) -> Any:
        """Dispatch to the high- or low-noise sub-transformer."""
        target = self.high_noise if use_high_noise else self.low_noise
        return target(**kwargs)


class WAN22Bundle(Bundle):
    """WAN 2.2 T2V bundle: dual transformer + VAE + UMT5 text encoder."""

    def __init__(
        self,
        *,
        transformer: WanDualTransformer,
        high_noise_transformer: nn.Module,
        low_noise_transformer: nn.Module,
        vae: Optional[nn.Module],
        text_encoder: nn.Module,
        tokenizer: Any,
        dtype: torch.dtype,
        device: torch.device,
        pretrained_path: str,
        max_sequence_length: int,
        boundary_ratio: float,
        guidance_scale_2: Any,
        num_train_timesteps: int,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.high_noise_transformer = high_noise_transformer
        self.low_noise_transformer = low_noise_transformer
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.dtype = dtype
        self.device = device
        self.pretrained_path = pretrained_path
        self.max_sequence_length = max_sequence_length
        self.boundary_ratio = float(boundary_ratio)
        self.guidance_scale_2 = guidance_scale_2
        self.num_train_timesteps = int(num_train_timesteps)

    @classmethod
    def from_config(cls, config: WAN22PipelineConfig) -> "WAN22Bundle":
        """Load both WAN 2.2 transformers + reuse WAN 2.1 VAE / text loaders."""
        try:
            from diffusers import WanTransformer3DModel
        except ImportError:
            from diffusers import AutoModel

            WanTransformer3DModel = AutoModel

        aux = WAN21Bundle.from_config(config)
        high_noise_transformer = aux.transformer

        transformer_2_path = config.transformer_2_pretrained_path or config.pretrained_model_ckpt_path
        dtype = parse_torch_dtype(config.model_precision, field_name="model_precision")
        low_noise_transformer = WanTransformer3DModel.from_pretrained(
            transformer_2_path,
            subfolder="transformer_2",
            torch_dtype=dtype,
        )
        low_noise_transformer = low_noise_transformer.to(aux.device, dtype=dtype)

        transformer = WanDualTransformer(
            high_noise=high_noise_transformer,
            low_noise=low_noise_transformer,
        )

        return cls(
            transformer=transformer,
            high_noise_transformer=high_noise_transformer,
            low_noise_transformer=low_noise_transformer,
            vae=aux.vae,
            text_encoder=aux.text_encoder,
            tokenizer=aux.tokenizer,
            dtype=aux.dtype,
            device=aux.device,
            pretrained_path=aux.pretrained_path,
            max_sequence_length=aux.max_sequence_length,
            boundary_ratio=float(config.boundary_ratio),
            guidance_scale_2=config.guidance_scale_2,
            num_train_timesteps=int(config.num_train_timesteps),
        )

    def weight_sync_name_map(self) -> Dict[str, str]:
        """Return the prefix-substitution map for cross-process weight sync."""
        return {
            "high_noise.": "transformer.",
            "low_noise.": "transformer_2.",
        }


__all__ = ["WanDualTransformer", "WAN22Bundle"]
