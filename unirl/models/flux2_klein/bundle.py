"""Flux2KleinBundle — concrete weights+params holder for FLUX.2-klein-9B.

Implements the empty :class:`Bundle` Protocol. Pure container of the
modules FLUX.2-klein-9B ships with:

- 1× ``Flux2Transformer2DModel`` (9B params, joint_attention_dim=15360)
- 1× ``AutoencoderKLFlux2`` (32 latent channels, BN-normalized
  patchified latents)
- 1× Qwen3 text encoder (``AutoModelForCausalLM`` →
  ``Qwen3ForCausalLM``) + ``Qwen2TokenizerFast``
- 1× ``FlowMatchEulerDiscreteScheduler`` (empirical-mu schedule)

Diverges from :class:`unirl.models.sd3.SD3Bundle` and
:class:`unirl.models.qwen_image.QwenImageBundle` in two ways:

- **Klein-specific guidance-embedder materialization**. Older
  ``diffusers`` builds construct ``time_guidance_embed.guidance_embedder``
  on the transformer even when ``transformer/config.json`` sets
  ``guidance_embeds: false`` (Klein has no guidance distillation).
  ``from_pretrained`` then leaves those tensors on the ``meta`` device,
  which crashes the first forward with ``NotImplementedError`` from
  the FSDP all-gather. We zero-init any leftover ``meta`` tensors here
  so the bundle is fully materialized before the FSDP wrap.
- **Qwen3 text encoder via ``AutoModelForCausalLM``** (vs Qwen-Image's
  ``Qwen2_5_VLForConditionalGeneration``). Klein uses the language-only
  Qwen3 LLM as the text encoder; the chat-template + intermediate-layer
  concatenation lives in :class:`Flux2KleinTextEmbedStage`.

No LoRA injection, FSDP wrap, adapter switching, autocast helpers, or
weight-sync logic — those are lifecycle concerns owned outside the
bundle (``cfg.training.policies``).

Use :meth:`Flux2KleinBundle.from_config` to load a checkpoint.
"""

from __future__ import annotations

import logging
from typing import Any, List, Tuple

import torch
import torch.nn as nn

from unirl.models.types.bundle import Bundle
from unirl.utils.dtypes import parse_torch_dtype

from .config import Flux2KleinPipelineConfig

logger = logging.getLogger(__name__)


def _materialize_meta_tensors(module: nn.Module) -> List[str]:
    """Replace any remaining ``meta``-device parameters/buffers in
    ``module`` with zero-initialized real tensors on CPU.

    Used to recover from a ``from_pretrained`` call that left some
    submodules un-loaded because their weights are absent from the
    checkpoint (e.g. FLUX.2-klein-9B's
    ``time_guidance_embed.guidance_embedder`` when running against an
    older ``diffusers`` build that always constructs the module even
    though ``transformer/config.json`` sets ``guidance_embeds: false``).

    Returns the qualified names of every tensor that was materialized.
    """
    materialized: List[str] = []

    def _resolve_parent(root: nn.Module, qualified_name: str) -> Tuple[nn.Module, str]:
        parts = qualified_name.split(".")
        parent = root
        for piece in parts[:-1]:
            parent = getattr(parent, piece)
        return parent, parts[-1]

    for name, param in list(module.named_parameters()):
        if param.is_meta:
            parent, attr = _resolve_parent(module, name)
            new_param = nn.Parameter(
                torch.zeros(param.shape, dtype=param.dtype, device="cpu"),
                requires_grad=param.requires_grad,
            )
            setattr(parent, attr, new_param)
            materialized.append(name)

    for name, buf in list(module.named_buffers()):
        if buf.is_meta:
            parent, attr = _resolve_parent(module, name)
            persistent = name not in getattr(parent, "_non_persistent_buffers_set", set())
            parent.register_buffer(
                attr,
                torch.zeros(buf.shape, dtype=buf.dtype, device="cpu"),
                persistent=persistent,
            )
            materialized.append(name + " (buffer)")

    return materialized


class Flux2KleinBundle(Bundle):
    """FLUX.2-klein-9B bundle: transformer + VAE + Qwen3 text encoder + scheduler."""

    def __init__(
        self,
        *,
        transformer: nn.Module,
        vae: nn.Module,
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
    def from_config(cls, config: Flux2KleinPipelineConfig) -> "Flux2KleinBundle":
        """Load all FLUX.2-klein-9B components from a HuggingFace-layout checkpoint.

        Honors per-component path overrides (``vae_ckpt_path`` /
        ``text_encoder_ckpt_path``) so fine-tuning recipes can swap in
        alternate VAE / text-encoder checkpoints without re-downloading
        the 9B transformer.
        """
        from diffusers import AutoencoderKLFlux2, Flux2Transformer2DModel
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
        from transformers import AutoModelForCausalLM, AutoTokenizer

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

        # --- Transformer (9B, with meta-tensor materialization) ---
        transformer = Flux2Transformer2DModel.from_pretrained(
            path,
            subfolder="transformer",
            torch_dtype=dtype,
        )
        materialized = _materialize_meta_tensors(transformer)
        if materialized:
            preview = materialized[:5] + (["..."] if len(materialized) > 5 else [])
            logger.warning(
                "FLUX.2-klein transformer: zero-initialized %d parameter(s)/buffer(s) "
                "missing from the checkpoint (e.g. klein-9B guidance embedder): %s",
                len(materialized),
                preview,
            )
        transformer = transformer.to(device)

        # --- VAE (frozen, eval) ---
        vae = AutoencoderKLFlux2.from_pretrained(vae_path, subfolder="vae", torch_dtype=vae_dtype).to(device).eval()
        vae.requires_grad_(False)

        # --- Qwen3 text encoder (frozen, eval) ---
        tokenizer = AutoTokenizer.from_pretrained(text_encoder_path, subfolder="tokenizer")
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token

        text_encoder = (
            AutoModelForCausalLM.from_pretrained(text_encoder_path, subfolder="text_encoder", torch_dtype=te_dtype)
            .to(device)
            .eval()
        )
        text_encoder.requires_grad_(False)

        # --- Scheduler (FlowMatchEulerDiscreteScheduler with empirical mu) ---
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


__all__ = ["Flux2KleinBundle"]
