"""Flux2KleinBundle — concrete weights+params holder for FLUX.2-klein-9B."""

from __future__ import annotations

import glob
import logging
import os
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn

from unirl.models.types.bundle import Bundle
from unirl.models.types.meta_init import build_meta_init_transformer
from unirl.utils.dtypes import parse_torch_dtype

from .config import Flux2KleinPipelineConfig

logger = logging.getLogger(__name__)


def _materialize_meta_tensors(module: nn.Module) -> List[str]:
    """Replace any remaining ``meta``-device parameters/buffers in"""
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


def _stamp_zero_checkpoint_absent_params(transformer: nn.Module, weights_dir: str) -> None:
    """Zero-init (post-load, deferred) transformer params the checkpoint omits."""
    from safetensors import safe_open

    from unirl.train.deferred import _stamp

    ckpt_keys: set = set()
    if os.path.isdir(weights_dir):
        for shard in sorted(glob.glob(os.path.join(weights_dir, "*.safetensors"))):
            with safe_open(shard, framework="pt") as handle:
                ckpt_keys.update(handle.keys())
    absent = frozenset(name for name, _ in transformer.named_parameters() if name not in ckpt_keys)
    if not absent:
        return

    def _zero_absent(model: nn.Module) -> None:
        from unirl.train.backend.sharded_state import _canonical_param_name

        zeroed: List[str] = []
        with torch.no_grad():
            for name, param in model.named_parameters():
                # The deferred op runs post-wrap: activation checkpointing
                # interposes wrapper segments the captured names never had.
                if _canonical_param_name(name) in absent:
                    param.zero_()
                    zeroed.append(name)
        if len(zeroed) != len(absent):
            raise RuntimeError(
                f"FLUX.2-klein meta-init: zeroed {len(zeroed)} of {len(absent)} checkpoint-absent "
                f"param(s); the unmatched ones keep to_empty garbage. absent={sorted(absent)[:8]} "
                f"zeroed={sorted(zeroed)[:8]}"
            )
        if zeroed:
            logger.info(
                "FLUX.2-klein meta-init: zero-initialized %d checkpoint-absent param(s): %s",
                len(zeroed),
                zeroed[:8],
            )

    _stamp(transformer, _zero_absent)


class Flux2KleinBundle(Bundle):
    """FLUX.2-klein-9B bundle: transformer + VAE + Qwen3 text encoder + scheduler."""

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
    def from_config(cls, config: Flux2KleinPipelineConfig) -> "Flux2KleinBundle":
        """Load all FLUX.2-klein-9B components from a HuggingFace-layout checkpoint."""
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

        meta_init_state = None
        if config.meta_init_transformer:
            # Zero-init checkpoint-absent guidance parameters after meta materialization.
            transformer_config = Flux2Transformer2DModel.load_config(path, subfolder="transformer")
            transformer, meta_init_state = build_meta_init_transformer(
                lambda: Flux2Transformer2DModel.from_config(transformer_config), dtype=dtype
            )
            _stamp_zero_checkpoint_absent_params(transformer, os.path.join(path, "transformer"))
        else:
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

        vae = None
        if config.load_vae:
            vae = AutoencoderKLFlux2.from_pretrained(vae_path, subfolder="vae", torch_dtype=vae_dtype).to(device).eval()
            vae.requires_grad_(False)

        tokenizer = AutoTokenizer.from_pretrained(text_encoder_path, subfolder="tokenizer")
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token

        text_encoder = (
            AutoModelForCausalLM.from_pretrained(text_encoder_path, subfolder="text_encoder", torch_dtype=te_dtype)
            .to(device)
            .eval()
        )
        text_encoder.requires_grad_(False)

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
            bundle._transformer_weights_path = os.path.join(path, "transformer")
            bundle._meta_init_state = meta_init_state
        return bundle


__all__ = ["Flux2KleinBundle"]
