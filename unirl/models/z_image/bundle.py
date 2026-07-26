"""ZImageBundle — concrete weights+params holder for Z-Image.

Loads either variant (architecture-identical): the base
``Tongyi-MAI/Z-Image`` (CFG, ``shift: 6.0``) or the distilled
``Z-Image-Turbo`` (no CFG, ``shift: 3.0``). Implements the empty
:class:`Bundle` Protocol. Pure container of the modules Z-Image ships with:
1× ``ZImageTransformer2DModel`` (the S3-DiT single-stream backbone), 1×
``AutoencoderKL`` (the 16-channel flux-style 2D VAE), 1× Qwen3 text encoder
(loaded via ``AutoModel``) + matching tokenizer, 1×
``FlowMatchEulerDiscreteScheduler``.

Diverges from :class:`unirl.models.qwen_image.QwenImageBundle` in two ways:

- **Standard 2D VAE** (``AutoencoderKL``, 4D latents ``[B, C, H, W]``)
  rather than Qwen-Image's video VAE — so the decode stage is the plain
  SD3-style ``scaling_factor`` / ``shift_factor`` round trip, no temporal
  squeeze/expand.
- **Qwen3 text encoder** (a pure causal LM, ``Qwen3Model``) consumed via a
  chat template, taking ``hidden_states[-2]`` (the second-to-last layer);
  no fixed-prefix strip and no pooled vector.

No LoRA injection, FSDP wrap, adapter switching, autocast helpers, or
weight-sync logic — those are lifecycle concerns owned outside the bundle
(``cfg.training.policies``).

Use :meth:`ZImageBundle.from_config` to load a checkpoint.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from unirl.models.types.bundle import Bundle
from unirl.utils.dtypes import parse_torch_dtype

from .config import ZImagePipelineConfig


class ZImageBundle(Bundle):
    """Z-Image bundle: S3-DiT transformer + AutoencoderKL + Qwen3 text encoder + scheduler."""

    def __init__(
        self,
        *,
        transformer: nn.Module,
        vae: Optional[nn.Module],
        text_encoder: nn.Module,
        tokenizer: Any,
        scheduler: Any,
        dtype: torch.dtype,
        device: torch.device,
        pretrained_path: str,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.dtype = dtype
        self.device = device
        self.pretrained_path = pretrained_path

    @classmethod
    def from_config(cls, config: ZImagePipelineConfig) -> "ZImageBundle":
        """Load all Z-Image components from a HuggingFace-layout checkpoint.

        Honors per-component path overrides (``vae_ckpt_path`` /
        ``text_encoder_ckpt_path``) so fine-tuning recipes can swap in
        alternate VAE / text-encoder checkpoints without re-downloading
        the transformer. Both default to ``pretrained_model_ckpt_path``.
        """
        from diffusers import AutoencoderKL, ZImageTransformer2DModel
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
        from transformers import AutoModel, AutoTokenizer

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

        transformer = ZImageTransformer2DModel.from_pretrained(path, subfolder="transformer", torch_dtype=dtype).to(
            device
        )

        vae = None
        if config.load_vae:
            vae = AutoencoderKL.from_pretrained(vae_path, subfolder="vae", torch_dtype=vae_dtype).to(device).eval()
            vae.requires_grad_(False)

        # AutoModel (not a hardcoded Qwen3Model) mirrors the official Z-Image
        # loader: it instantiates the base encoder named in the checkpoint's
        # text_encoder/config.json, so a future text-encoder swap just works.
        text_encoder = (
            AutoModel.from_pretrained(text_encoder_path, subfolder="text_encoder", torch_dtype=te_dtype)
            .to(device)
            .eval()
        )
        text_encoder.requires_grad_(False)

        tokenizer = AutoTokenizer.from_pretrained(text_encoder_path, subfolder="tokenizer")

        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(path, subfolder="scheduler")

        return cls(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            dtype=dtype,
            device=device,
            pretrained_path=path,
        )


__all__ = ["ZImageBundle"]
