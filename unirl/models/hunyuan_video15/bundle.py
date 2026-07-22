"""HunyuanVideo15Bundle — concrete weights+params holder for HunyuanVideo-1.5.

Implements the empty :class:`Bundle` Protocol. Pure container of the
modules HunyuanVideo-1.5 ships with:

- 1× ``HunyuanVideo15Transformer3DModel`` (MMDiT, expects packed input
  ``cat([latents, cond_latents, cond_mask], dim=1)``)
- 1× ``AutoencoderKLHunyuanVideo15`` (3D VAE; spatial=16×, temporal=4×)
- 2× text encoders + tokenizers:
    - ``Qwen2_5_VLTextModel`` + ``Qwen2Tokenizer`` (MLLM, chat-template)
    - ``T5EncoderModel`` + ``ByT5Tokenizer`` (glyph)
- (optional) ``SiglipVisionModel`` + ``SiglipImageProcessor``
  — only loaded when ``load_vision_encoder=True`` (for I2V).
- 1× ``FlowMatchEulerDiscreteScheduler``

Diverges from :class:`unirl.models.wan21.WAN21Bundle` mainly
in the dual text-encoder pair + the optional SigLIP path. Diverges from
:class:`unirl.models.hunyuan_image3.HunyuanImage3Bundle` in
that HunyuanImage3 uses a single fused-multimodal transformer with its
own embedding table, while HunyuanVideo-1.5 keeps the dual text streams
separate as cross-attention KV.

No LoRA injection, FSDP wrap, adapter switching, or weight-sync logic
— those are lifecycle concerns owned outside the bundle
(``cfg.training.policies``). The bundle exposes attributes by name so
the diffusion stage and downstream FSDPPolicy can address them without
indirection.

Aborts loudly on transformers configured with ``use_meanflow=True``
(the OLD bundle does the same): the meanflow branch needs ``timestep_r``
threaded through the training forward, which the pipeline
does not yet expose. Failing fast here beats silent shape mismatches at
the first forward.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import torch
import torch.nn as nn

from unirl.models.types.bundle import Bundle
from unirl.models.types.meta_init import build_meta_init_transformer
from unirl.utils.dtypes import parse_torch_dtype

from .config import HunyuanVideo15PipelineConfig


class HunyuanVideo15Bundle(Bundle):
    """HunyuanVideo-1.5 bundle: transformer + 3D VAE + dual text encoders +
    optional SigLIP + scheduler."""

    def __init__(
        self,
        *,
        transformer: nn.Module,
        vae: nn.Module,
        text_encoder: nn.Module,
        tokenizer: Any,
        text_encoder_2: nn.Module,
        tokenizer_2: Any,
        vision_encoder: Optional[nn.Module],
        image_processor: Optional[Any],
        scheduler: Any,
        dtype: torch.dtype,
        device: torch.device,
        pretrained_path: str,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.vae = vae
        # MLLM (Qwen2.5-VL) stream.
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        # ByT5 glyph stream.
        self.text_encoder_2 = text_encoder_2
        self.tokenizer_2 = tokenizer_2
        # SigLIP (optional, I2V only).
        self.vision_encoder = vision_encoder
        self.image_processor = image_processor
        self.scheduler = scheduler
        self.dtype = dtype
        self.device = device
        self.pretrained_path = pretrained_path

    @classmethod
    def from_config(cls, config: HunyuanVideo15PipelineConfig) -> "HunyuanVideo15Bundle":
        """Load all HunyuanVideo-1.5 components from a checkpoint."""
        from diffusers import AutoencoderKLHunyuanVideo15, HunyuanVideo15Transformer3DModel
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
        from transformers import (
            ByT5Tokenizer,
            Qwen2_5_VLTextModel,
            Qwen2Tokenizer,
            T5EncoderModel,
        )

        path = config.pretrained_model_ckpt_path
        vae_path = config.vae_ckpt_path or path
        te1_path = config.text_encoder_ckpt_path or path
        te2_path = config.text_encoder_2_ckpt_path or path
        vis_path = config.image_encoder_ckpt_path or path

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
            # Meta-init (FSDP / VeOmni load_sharded path): architecture only,
            # no per-rank weight allocation; the backend materializes + loads
            # from the stashed dir after sharding. build_meta_init_transformer
            # keeps init-computed non-persistent buffers (rope tables) real and
            # captures them into meta_init_state (stashed on the bundle below).
            transformer_config = HunyuanVideo15Transformer3DModel.load_config(path, subfolder="transformer")
            transformer, meta_init_state = build_meta_init_transformer(
                lambda: HunyuanVideo15Transformer3DModel.from_config(transformer_config), dtype=dtype
            )
        else:
            transformer = HunyuanVideo15Transformer3DModel.from_pretrained(
                path, subfolder="transformer", torch_dtype=dtype
            )
            transformer = transformer.to(device=device, dtype=dtype)
        # Reject meanflow checkpoints — replay path doesn't thread timestep_r yet.
        # (config is metadata, present on both the meta and eager builds.)
        if bool(getattr(getattr(transformer, "config", None), "use_meanflow", False)):
            raise NotImplementedError(
                "HunyuanVideo15Bundle does not support transformers with "
                "use_meanflow=True; timestep_r is not threaded through the "
                "forward path."
            )

        vae = (
            AutoencoderKLHunyuanVideo15.from_pretrained(vae_path, subfolder="vae", torch_dtype=vae_dtype)
            .to(device)
            .eval()
        )
        vae.requires_grad_(False)

        text_encoder = (
            Qwen2_5_VLTextModel.from_pretrained(te1_path, subfolder="text_encoder", torch_dtype=te_dtype)
            .to(device)
            .eval()
        )
        text_encoder.requires_grad_(False)
        tokenizer = Qwen2Tokenizer.from_pretrained(te1_path, subfolder="tokenizer")

        text_encoder_2 = (
            T5EncoderModel.from_pretrained(te2_path, subfolder="text_encoder_2", torch_dtype=te_dtype).to(device).eval()
        )
        text_encoder_2.requires_grad_(False)
        tokenizer_2 = ByT5Tokenizer.from_pretrained(te2_path, subfolder="tokenizer_2")

        vision_encoder: Optional[nn.Module] = None
        image_processor: Optional[Any] = None
        if bool(config.load_vision_encoder):
            from transformers import SiglipImageProcessor, SiglipVisionModel

            vision_encoder = (
                SiglipVisionModel.from_pretrained(vis_path, subfolder="image_encoder", torch_dtype=dtype)
                .to(device)
                .eval()
            )
            vision_encoder.requires_grad_(False)
            image_processor = SiglipImageProcessor.from_pretrained(vis_path, subfolder="feature_extractor")

        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(path, subfolder="scheduler")

        bundle = cls(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            vision_encoder=vision_encoder,
            image_processor=image_processor,
            scheduler=scheduler,
            dtype=dtype,
            device=device,
            pretrained_path=path,
        )
        if config.meta_init_transformer:
            # Consumed by the backend's post-shard weight load.
            bundle._transformer_weights_path = os.path.join(path, "transformer")
            # Ray-robust restore carrier for init-computed non-persistent state.
            bundle._meta_init_state = meta_init_state
        return bundle


__all__ = ["HunyuanVideo15Bundle"]
