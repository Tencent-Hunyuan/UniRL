"""HunyuanVideoBundle -- concrete weights+params holder for HunyuanVideo-1.0.

Implements the empty :class:`Bundle` Protocol. Pure container of the
modules HunyuanVideo-1.0 ships with:

- 1x ``HunyuanVideoTransformer3DModel`` (MMDiT with ``guidance_embeds=True``,
  ``in_channels=16`` -- no channel-dim packing)
- 1x ``AutoencoderKLHunyuanVideo`` (3D VAE; spatial=8x, temporal=4x,
  latent_channels=16, scaling_factor=0.476986)
- 2x text encoders + tokenizers:
    - ``LlamaModel`` + ``LlamaTokenizerFast`` (primary, 4096-dim hidden)
    - ``CLIPTextModel`` + ``CLIPTokenizer`` (pooled, 768-dim)
- 1x ``FlowMatchEulerDiscreteScheduler``

Diverges from :class:`unirl.models.hunyuan_video15.HunyuanVideo15Bundle`
mainly in the text-encoder pair (LLaMA + CLIP vs Qwen-VL + ByT5), the
absence of a vision encoder (no I2V slot), and the transformer signature
(no channel-dim packing, uses ``guidance`` kwarg instead of CFG).

No LoRA injection, FSDP wrap, adapter switching, or weight-sync logic
-- those are lifecycle concerns owned outside the bundle
(``cfg.training.policies``). The bundle exposes attributes by name so
the diffusion stage and downstream FSDPPolicy can address them without
indirection.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from unirl.models.types.bundle import Bundle
from unirl.utils.dtypes import parse_torch_dtype

from .config import HunyuanVideoPipelineConfig


class HunyuanVideoBundle(Bundle):
    """HunyuanVideo-1.0 bundle: transformer + 3D VAE + dual text encoders +
    scheduler."""

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
        # LLaMA stream (primary, 4096-dim hidden states).
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        # CLIP stream (pooled, 768-dim).
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

        transformer = HunyuanVideoTransformer3DModel.from_pretrained(path, subfolder="transformer", torch_dtype=dtype)
        transformer = transformer.to(device=device, dtype=dtype)

        vae = (
            AutoencoderKLHunyuanVideo.from_pretrained(vae_path, subfolder="vae", torch_dtype=vae_dtype)
            .to(device)
            .eval()
        )
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

        return cls(
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


__all__ = ["HunyuanVideoBundle"]
