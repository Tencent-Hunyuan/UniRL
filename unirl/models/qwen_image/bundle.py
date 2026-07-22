"""QwenImageBundle — concrete weights+params holder for Qwen-Image.

Implements the empty :class:`Bundle` Protocol. Pure container of the
modules Qwen-Image ships with: 1× ``QwenImageTransformer2DModel``, 1×
``AutoencoderKLQwenImage``, 1× ``Qwen2_5_VLForConditionalGeneration``
text encoder + ``Qwen2Tokenizer``, 1× ``FlowMatchEulerDiscreteScheduler``.

Diverges from :class:`unirl.models.sd3.SD3Bundle` in two ways:

- **Single text encoder** (vs SD3's CLIP1 + CLIP2 + T5 stack). Qwen-Image
  uses a multimodal LLM (Qwen-2.5-VL) as a text encoder; the tokenizer
  is the matching ``Qwen2Tokenizer``. Pooled vectors are not produced —
  the receiving transformer reads token-level hidden states only.
- **5D VAE latents** ``[B, C, T=1, H, W]``. Qwen-Image's VAE is the
  video VAE (``AutoencoderKLQwenImage``) used with a single frame; the
  decode/encode stages handle the temporal squeeze/expand at the
  boundary.

No LoRA injection, FSDP wrap, adapter switching, autocast helpers, or
weight-sync logic — those are lifecycle concerns owned outside the
bundle (``cfg.training.policies``).

Use :meth:`QwenImageBundle.from_config` to load a checkpoint.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import torch
import torch.nn as nn

from unirl.models.types.bundle import Bundle
from unirl.models.types.meta_init import build_meta_init_transformer
from unirl.utils.dtypes import parse_torch_dtype

from .config import QwenImagePipelineConfig

logger = logging.getLogger(__name__)


class QwenImageBundle(Bundle):
    """Qwen-Image bundle: transformer + VAE + Qwen-VL text encoder + scheduler."""

    def __init__(
        self,
        *,
        transformer: nn.Module,
        vae: Optional[nn.Module],
        text_encoder: Optional[nn.Module],
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
    def from_config(cls, config: QwenImagePipelineConfig) -> "QwenImageBundle":
        """Load all Qwen-Image components from a HuggingFace-layout checkpoint.

        Honors per-component path overrides (``vae_ckpt_path`` /
        ``text_encoder_ckpt_path``) so fine-tuning recipes can swap in
        alternate VAE / text-encoder checkpoints without re-downloading
        the transformer. Both default to ``pretrained_model_ckpt_path``.
        """

        import fcntl

        # Node-local load serialization: 8 colocated ranks each hold ~20 GiB
        # anon RSS while materializing the 20B transformer (safetensors ->
        # bf16 staging). The simultaneous burst blows the pod's k8s memcg
        # limit (~439 GiB incl. page cache) and the kernel OOM-kills
        # raylet/python (LIN-382 qwen probes b/d: "Memory cgroup out of
        # memory", anon-rss ~20-23 GiB per kill). Single-file the heavy
        # window; DIFFRL_MODEL_LOAD_SERIALIZE=0 opts out (single-rank runs).
        serialize = os.environ.get("DIFFRL_MODEL_LOAD_SERIALIZE", "1") != "0"
        lock_file = open("/tmp/diffrl_model_load.lock", "a+") if serialize else None
        if lock_file is not None:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            return cls._from_config_locked(config)
        finally:
            if lock_file is not None:
                # Return this rank's staging anon to the kernel before the
                # next rank starts its load, so the serialized peak holds.
                import gc

                gc.collect()
                fcntl.flock(lock_file, fcntl.LOCK_UN)
                lock_file.close()

    @classmethod
    def _from_config_locked(cls, config: QwenImagePipelineConfig) -> "QwenImageBundle":
        from diffusers import AutoencoderKLQwenImage, QwenImageTransformer2DModel
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
        from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer

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
            # Meta-init (VeOmni load_sharded path): params on meta, weights load
            # post-parallelize. Qwen-Image's QwenEmbedRope holds its complex rope
            # tables (pos_freqs / neg_freqs) as plain __dict__ tensors, so to_empty
            # never materializes them; build_meta_init_transformer keeps them real
            # on CPU and captures them into meta_init_state for load_trainable_weights
            # to restore after the sharded load.
            transformer_config = QwenImageTransformer2DModel.load_config(path, subfolder="transformer")
            transformer, meta_init_state = build_meta_init_transformer(
                lambda: QwenImageTransformer2DModel.from_config(transformer_config), dtype=dtype
            )
        else:
            transformer = QwenImageTransformer2DModel.from_pretrained(
                path, subfolder="transformer", torch_dtype=dtype
            ).to(device)

        vae = None
        if config.load_vae:
            vae = (
                AutoencoderKLQwenImage.from_pretrained(vae_path, subfolder="vae", torch_dtype=vae_dtype)
                .to(device)
                .eval()
            )
            vae.requires_grad_(False)

        # Skipped when load_text_encoder=False (separate-engine; see config).
        text_encoder = None
        if config.load_text_encoder:
            text_encoder = (
                Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    text_encoder_path, subfolder="text_encoder", torch_dtype=te_dtype
                )
                .to(device)
                .eval()
            )
            text_encoder.requires_grad_(False)

        tokenizer = Qwen2Tokenizer.from_pretrained(text_encoder_path, subfolder="tokenizer")

        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(path, subfolder="scheduler")

        bundle = cls(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            dtype=dtype,
            device=device,
            pretrained_path=path,
        )
        if config.meta_init_transformer:
            # Consumed by VeOmniBackend's post-parallelize weight load.
            # Kept as the raw join — the backend validates local-dir-ness
            # at load time (HF repo IDs need a local download first).
            bundle._transformer_weights_path = os.path.join(path, "transformer")
            # Ray-robust restore carrier for init-computed non-persistent state.
            bundle._meta_init_state = meta_init_state
        return bundle


__all__ = ["QwenImageBundle"]
