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
from unirl.utils.dtypes import parse_torch_dtype

from .config import QwenImagePipelineConfig

logger = logging.getLogger(__name__)


def _rebuild_meta_rope_modules(transformer: nn.Module) -> int:
    """Rebuild rope-embed modules whose tables were built on meta.

    diffusers' ``QwenEmbedRope`` keeps its complex rope tables
    (``pos_freqs`` / ``neg_freqs``) as PLAIN tensor attributes —
    deliberately not buffers (upstream comment: registering complex
    buffers drops the imaginary part) — so ``to_empty`` never
    materializes them and they stay on meta. The module holds no
    parameters, so re-instantiating it on CPU from its own ctor attrs
    and swapping it in is shard-exempt; ``forward`` moves the tables to
    the live device on use."""
    count = 0
    for name, mod in list(transformer.named_modules()):
        pos_freqs = getattr(mod, "pos_freqs", None)
        if not (isinstance(pos_freqs, torch.Tensor) and pos_freqs.is_meta):
            continue
        fresh = type(mod)(
            theta=mod.theta,
            axes_dim=list(mod.axes_dim),
            scale_rope=getattr(mod, "scale_rope", False),
        )
        if "." in name:
            parent_name, attr = name.rsplit(".", 1)
            setattr(transformer.get_submodule(parent_name), attr, fresh)
        else:
            setattr(transformer, name, fresh)
        count += 1
    return count


class QwenImageBundle(Bundle):
    """Qwen-Image bundle: transformer + VAE + Qwen-VL text encoder + scheduler."""

    def __init__(
        self,
        *,
        transformer: nn.Module,
        vae: nn.Module,
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

        if config.meta_init_transformer:
            # VeOmniBackend lifecycle: architecture only, on the meta device
            # (no weight allocation). VeOmni's parallelize asserts meta init,
            # materializes storage via ``to_empty``, and calls the model's
            # ``init_weights()`` unconditionally — stamped to a no-op here so
            # it cannot clobber anything; the backend loads the real weights
            # from the stashed path after sharding (rank0 read + broadcast).
            transformer_config = QwenImageTransformer2DModel.load_config(path, subfolder="transformer")
            with torch.device("meta"):
                transformer = QwenImageTransformer2DModel.from_config(transformer_config)
            rebuilt = _rebuild_meta_rope_modules(transformer)
            if rebuilt:
                logger.info("meta_init_transformer: rebuilt %d rope module(s) on CPU", rebuilt)
            transformer = transformer.to(dtype)
            transformer.init_weights = lambda: None
            non_persistent = sorted(set(n for n, _ in transformer.named_buffers()) - set(transformer.state_dict()))
            if non_persistent:
                # Non-persistent buffers are absent from checkpoints, so
                # ``to_empty`` leaves them uninitialized — if any module
                # relies on init-time buffer values, the parity gates will
                # surface it; this log names the suspects.
                logger.warning(
                    "meta_init_transformer: %d non-persistent buffer(s) will not be "
                    "restored by the checkpoint load: %s",
                    len(non_persistent),
                    non_persistent[:8],
                )
        else:
            transformer = QwenImageTransformer2DModel.from_pretrained(
                path, subfolder="transformer", torch_dtype=dtype
            ).to(device)

        vae = AutoencoderKLQwenImage.from_pretrained(vae_path, subfolder="vae", torch_dtype=vae_dtype).to(device).eval()
        vae.requires_grad_(False)

        # vllm-omni recipes skip the trainer-side copy (~15 GiB/rank dead
        # weight there — the engine encodes prompts in its own workers and
        # the trainer replays from captured conditions); see
        # QwenImagePipelineConfig.load_text_encoder.
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
        return bundle


__all__ = ["QwenImageBundle"]
