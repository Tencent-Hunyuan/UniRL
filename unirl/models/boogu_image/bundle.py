"""BooguImageBundle — concrete weights+params holder for Boogu-Image.

Implements the empty :class:`Bundle` Protocol. Pure container of the modules
Boogu-Image-0.1 ships with: 1× vendored ``BooguImageTransformer2DModel`` (the
10.29B Lumina-2-lineage single/double-stream DiT), 1× ``AutoencoderKL`` (the
16-channel FLUX.1 VAE), 1× Qwen3-VL instruction encoder (the checkpoint's
``mllm/`` subfolder) + its ``Qwen3VLProcessor`` (``processor/`` subfolder).

Diverges from :class:`unirl.models.z_image.ZImageBundle` /
:class:`unirl.models.qwen_image.QwenImageBundle` in three ways:

- **Non-standard checkpoint subfolders**: the encoder lives under ``mllm/``
  (not ``text_encoder/``) and the tokenizer side is a full processor under
  ``processor/`` (not ``tokenizer/``). There is no ``tokenizer`` attribute —
  the processor owns tokenization + chat templating.
- **lm_head strip**: the reference pipeline reuses the encoder checkpoint as
  an optional prompt rewriter, so ``mllm/`` may hold a full
  ``Qwen3VLForConditionalGeneration``. Mirroring ``pipeline_boogu.py:194-198``,
  the bundle keeps only the inner ``.model`` (``Qwen3VLModel``) when an
  ``lm_head`` is present — conditioning uses hidden states only.
- **No scheduler object**: the upstream time-shifting scheduler class is not
  vendored; its released static-v1 schedule is expressed through
  ``BooguImagePipeline.build_schedule_policy()`` (bagel precedent — the σ
  math lives in the policy, not a diffusers scheduler instance).

No LoRA injection, FSDP wrap, adapter switching, autocast helpers, or
weight-sync logic — those are lifecycle concerns owned outside the bundle.

Use :meth:`BooguImageBundle.from_config` to load a checkpoint.
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

from .config import BooguImagePipelineConfig
from .vendor import BooguImageTransformer2DModel

logger = logging.getLogger(__name__)


def _swap_attention_processors_to_flash(transformer: nn.Module) -> int:
    """Swap the pinned SDPA attention processors for their Flash2Varlen
    twins (identical parameter names, so ``load_state_dict`` transfers the
    double-stream QKV weights losslessly). Returns the swap count.

    Meta-built modules are swapped structurally with no state transfer —
    the post-shard checkpoint load fills the fresh processors.
    """
    from .vendor.attention_processor import (
        BooguImageAttnProcessor,
        BooguImageAttnProcessorFlash2Varlen,
        BooguImageDoubleStreamSelfAttnProcessor,
        BooguImageDoubleStreamSelfAttnProcessorFlash2Varlen,
    )

    count = 0
    for module in transformer.modules():
        processor = getattr(module, "processor", None)
        if isinstance(processor, BooguImageDoubleStreamSelfAttnProcessor):
            fresh = BooguImageDoubleStreamSelfAttnProcessorFlash2Varlen(
                head_dim=processor.head_dim,
                num_attention_heads=processor.num_attention_heads,
                num_kv_heads=processor.num_kv_heads,
                qkv_bias=False,
            )
            params = list(processor.parameters())
            if params and not params[0].is_meta:
                fresh.load_state_dict(processor.state_dict())
                fresh = fresh.to(device=params[0].device, dtype=params[0].dtype)
            module.set_processor(fresh)
            count += 1
        elif isinstance(processor, BooguImageAttnProcessor):
            module.set_processor(BooguImageAttnProcessorFlash2Varlen())
            count += 1
    return count


class BooguImageBundle(Bundle):
    """Boogu-Image bundle: vendored DiT + FLUX VAE + Qwen3-VL encoder + processor."""

    def __init__(
        self,
        *,
        transformer: nn.Module,
        vae: Optional[nn.Module],
        text_encoder: Optional[nn.Module],
        processor: Any,
        dtype: torch.dtype,
        device: torch.device,
        pretrained_path: str,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.vae = vae
        self.text_encoder = text_encoder
        self.processor = processor
        self.dtype = dtype
        self.device = device
        self.pretrained_path = pretrained_path

    @classmethod
    def from_config(cls, config: BooguImagePipelineConfig) -> "BooguImageBundle":
        """Load all Boogu-Image components from a HuggingFace-layout checkpoint.

        Honors per-component path overrides (``vae_ckpt_path`` /
        ``text_encoder_ckpt_path``); both default to
        ``pretrained_model_ckpt_path``.
        """

        import fcntl

        # Node-local load serialization (qwen_image precedent): 8 colocated
        # ranks each stage ~20 GiB (10.29B DiT) + ~17 GiB (Qwen3-VL-8B)
        # while materializing; the simultaneous burst can blow the pod memcg.
        # DIFFRL_MODEL_LOAD_SERIALIZE=0 opts out (single-rank runs).
        serialize = os.environ.get("DIFFRL_MODEL_LOAD_SERIALIZE", "1") != "0"
        lock_file = open("/tmp/diffrl_model_load.lock", "a+") if serialize else None
        if lock_file is not None:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            return cls._from_config_locked(config)
        finally:
            if lock_file is not None:
                import gc

                gc.collect()
                fcntl.flock(lock_file, fcntl.LOCK_UN)
                lock_file.close()

    @classmethod
    def _from_config_locked(cls, config: BooguImagePipelineConfig) -> "BooguImageBundle":
        from diffusers import AutoencoderKL
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

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
            # Architecture only, on the meta device; the backend materializes
            # per-rank shards after wrapping and loads weights from the
            # stashed path. Boogu's vendored tree has zero registered buffers
            # and zero init-computed plain-tensor attrs (rope freqs_cis is a
            # per-call input), so the captured init state must come out EMPTY;
            # if it ever grows after a re-vendor, that is the signal a quirk
            # restore became load-bearing.
            transformer_config = BooguImageTransformer2DModel.load_config(path, subfolder="transformer")
            transformer, meta_init_state = build_meta_init_transformer(
                lambda: BooguImageTransformer2DModel.from_config(transformer_config), dtype=dtype
            )
        else:
            transformer = BooguImageTransformer2DModel.from_pretrained(
                path, subfolder="transformer", torch_dtype=dtype
            ).to(device)

        if bool(getattr(transformer, "enable_teacache", False)) or bool(
            getattr(transformer, "enable_taylorseer", False)
        ):
            raise RuntimeError(
                "BooguImageBundle: transformer loaded with TeaCache/TaylorSeer "
                "enabled — RL rollout/replay must be cache-free."
            )
        if config.attention_backend == "flash2_varlen":
            swapped = _swap_attention_processors_to_flash(transformer)
            logger.info("attention_backend=flash2_varlen: swapped %d processor(s)", swapped)

        vae = None
        if config.load_vae:
            vae = AutoencoderKL.from_pretrained(vae_path, subfolder="vae", torch_dtype=vae_dtype).to(device).eval()
            vae.requires_grad_(False)

        # Separate-engine recipes can skip the frozen encoder copy on actors
        # that never embed text (qwen_image precedent).
        text_encoder = None
        processor = None
        if config.load_text_encoder:
            # ``mllm/`` may ship the full CausalLM wrapper (reused upstream as
            # a prompt rewriter). Conditioning uses hidden states only, so
            # keep the inner ``Qwen3VLModel`` and free the wrapper/lm_head
            # (mirrors pipeline_boogu.py:194-198).
            wrapper = Qwen3VLForConditionalGeneration.from_pretrained(
                text_encoder_path, subfolder="mllm", torch_dtype=te_dtype
            )
            text_encoder = wrapper.model if hasattr(wrapper, "lm_head") else wrapper
            del wrapper
            text_encoder = text_encoder.to(device).eval()
            text_encoder.requires_grad_(False)

            processor = AutoProcessor.from_pretrained(text_encoder_path, subfolder="processor")

        bundle = cls(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            processor=processor,
            dtype=dtype,
            device=device,
            pretrained_path=path,
        )
        if config.meta_init_transformer:
            # Consumed by the backends' post-wrap sharded weight load.
            bundle._transformer_weights_path = os.path.join(path, "transformer")
            bundle._meta_init_state = meta_init_state
        return bundle


__all__ = ["BooguImageBundle"]
