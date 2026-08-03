"""MiniMax-H3 bundle -- plain weight holder for the t2va pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from unirl.models.types.meta_init import build_meta_init_transformer
from unirl.utils.dtypes import parse_torch_dtype

from .config import MiniMaxH3PipelineConfig
from .vendor import (
    AutoencoderKLMiniMaxH3,
    AutoencoderKLMiniMaxH3Audio,
    MiniMaxH3Transformer3DModel,
)


@dataclass
class MiniMaxH3Bundle:
    """Loaded MiniMax-H3 components.

    No scheduler is held. MiniMax-H3's two rectified-flow grids are plain
    ``shift*t / (1 + (shift-1)*t)`` over ``linspace(1, 0, N)``, which UniRL's
    static ``FlowMatchSchedulePolicy`` reproduces exactly -- so sigma is pinned
    onto the generation Part by the hosting engine like every other model here,
    and a ``MiniMaxH3Scheduler`` instance would be a second source of truth.
    """

    transformer: nn.Module
    vae: nn.Module
    audio_vae: nn.Module
    text_encoder: nn.Module
    processor: Any
    tokenizer: Any
    dtype: torch.dtype
    device: torch.device
    pretrained_path: str
    text_encoder_hidden_layer: int
    max_sequence_length: int

    @classmethod
    def from_config(cls, config: MiniMaxH3PipelineConfig) -> "MiniMaxH3Bundle":
        """Load every MiniMax-H3 component from a HuggingFace checkpoint."""
        from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

        path = config.pretrained_model_ckpt_path
        vae_path = config.vae_ckpt_path or path
        te_path = config.text_encoder_ckpt_path or path

        device = config.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, str):
            device = torch.device(device)
        # The frozen aux here is unusually heavy -- the Qwen3-VL conditioner is
        # 32B (~64 GB bf16) on its own, more than the trainable DiT's shard on
        # any sane mesh. Parking it (and the two fp32 VAEs) on CPU is what makes
        # an 8-GPU trainside recipe fit at all.
        aux_device = torch.device("cpu") if config.aux_components_on_cpu else device

        dtype = parse_torch_dtype(config.model_precision, field_name="model_precision")
        vae_dtype = parse_torch_dtype(config.vae_dtype, field_name="vae_dtype")
        audio_vae_dtype = parse_torch_dtype(config.audio_vae_dtype, field_name="audio_vae_dtype")
        te_raw = config.text_encoder_dtype if config.text_encoder_dtype is not None else config.model_precision
        te_dtype = parse_torch_dtype(te_raw, field_name="text_encoder_dtype")

        # Transformer (trainable). MiniMax-H3 ships a MIXED-DTYPE checkpoint:
        # `_keep_in_fp32_modules` (patch projections, timestep MLP, output heads,
        # rope) stays fp32 while the block stack is bf16, and the forward aligns
        # every input to its projection's parameter dtype. `from_pretrained`
        # honours that natively; the meta path has to be told, hence the
        # explicit `keep_in_fp32` hand-off below.
        meta_init_state = None
        if config.meta_init_transformer:
            transformer_config = MiniMaxH3Transformer3DModel.load_config(path, subfolder="transformer")
            transformer, meta_init_state = build_meta_init_transformer(
                lambda: MiniMaxH3Transformer3DModel.from_config(transformer_config),
                dtype=dtype,
                keep_in_fp32=MiniMaxH3Transformer3DModel._keep_in_fp32_modules,
            )
        else:
            transformer = MiniMaxH3Transformer3DModel.from_pretrained(
                path, subfolder="transformer", torch_dtype=dtype
            ).to(device)

        # Video VAE (frozen, fp32).
        vae = AutoencoderKLMiniMaxH3.from_pretrained(vae_path, subfolder="vae", torch_dtype=vae_dtype)
        vae = vae.to(aux_device).eval()
        vae.requires_grad_(False)

        # Audio VAE (frozen, fp32). Do NOT let a global bf16 cast reach this:
        # the reference reports bf16 output roughly 20 dB too quiet.
        audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(
            vae_path, subfolder="audio_vae", torch_dtype=audio_vae_dtype
        )
        audio_vae = audio_vae.to(aux_device).eval()
        audio_vae.requires_grad_(False)

        # Conditioner -- Qwen3-VL-32B (frozen). H3 reads an intermediate hidden
        # state from it, so it must be loaded as the full LM, not a truncated
        # encoder.
        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            te_path, subfolder="text_encoder", torch_dtype=te_dtype
        )
        text_encoder = text_encoder.to(aux_device).eval()
        text_encoder.requires_grad_(False)

        processor = AutoProcessor.from_pretrained(te_path, subfolder="processor")
        tokenizer = AutoTokenizer.from_pretrained(te_path, subfolder="tokenizer")

        bundle = cls(
            transformer=transformer,
            vae=vae,
            audio_vae=audio_vae,
            text_encoder=text_encoder,
            processor=processor,
            tokenizer=tokenizer,
            dtype=dtype,
            device=device,
            pretrained_path=path,
            text_encoder_hidden_layer=int(config.text_encoder_hidden_layer),
            max_sequence_length=int(config.max_sequence_length),
        )
        if config.meta_init_transformer:
            # Diffusers layout: the backend's sharded loader reads the
            # safetensors under <ckpt>/transformer after `to_empty`.
            bundle._transformer_weights_path = os.path.join(path, "transformer")
            bundle._meta_init_state = meta_init_state
        return bundle

    def trainable_module(self) -> nn.Module:
        return self.transformer


__all__ = ["MiniMaxH3Bundle"]
