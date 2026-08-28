"""MiniMax-H3 bundle -- plain weight holder for the t2va pipeline."""

from __future__ import annotations

import os
from typing import Any

import torch
from torch import nn

from unirl.models.types.bundle import Bundle
from unirl.models.types.meta_init import build_meta_init_transformer
from unirl.utils.dtypes import parse_torch_dtype

from .config import MiniMaxH3PipelineConfig
from .vendor import (
    AutoencoderKLMiniMaxH3,
    AutoencoderKLMiniMaxH3Audio,
    MiniMaxH3Transformer3DModel,
)


class MiniMaxH3Bundle(Bundle):
    """Loaded MiniMax-H3 components."""

    def __init__(
        self,
        *,
        transformer: nn.Module,
        vae: nn.Module,
        audio_vae: nn.Module,
        text_encoder: nn.Module,
        processor: Any,
        tokenizer: Any,
        dtype: torch.dtype,
        device: torch.device,
        pretrained_path: str,
        max_sequence_length: int,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.vae = vae
        self.audio_vae = audio_vae
        self.text_encoder = text_encoder
        self.processor = processor
        self.tokenizer = tokenizer
        self.dtype = dtype
        # NOTE: ``Remote.setup`` later rebinds this to the worker's device
        # STRING (e.g. "cuda:0"), so never assume ``torch.device`` attributes
        # (``.type``) off it -- pass it straight to ``.to()`` / ``device=``.
        self.device = device
        self.pretrained_path = pretrained_path
        self.max_sequence_length = max_sequence_length

    @classmethod
    def from_config(cls, config: MiniMaxH3PipelineConfig) -> "MiniMaxH3Bundle":
        """Load every MiniMax-H3 component from a HuggingFace checkpoint."""
        from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

        path = config.pretrained_model_ckpt_path
        transformer_weights_path = os.path.join(path, "transformer")
        if config.meta_init_transformer and not os.path.isdir(transformer_weights_path):
            raise ValueError(
                "MiniMaxH3Bundle: meta-init requires pretrained_model_ckpt_path to be a local snapshot root "
                f"containing transformer/*.safetensors; got {path!r}. Set PRETRAINED_MODEL to a downloaded snapshot."
            )
        vae_path = config.vae_ckpt_path or path
        te_path = config.text_encoder_ckpt_path or path

        device = config.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, str):
            device = torch.device(device)
        # The frozen aux here is unusually heavy -- the Qwen3-VL conditioner is
        # 32B (~64 GB bf16) on its own, more than the trainable DiT's shard on
        # any sane mesh. Parking it on CPU is what makes an 8-GPU trainside
        # recipe fit at all; it runs once per rollout, so the cost is noise.
        #
        # The VAEs are a separate call and default to the train device: they are
        # ~10 GB fp32 together, but decoding 124 frames of 768x768 on CPU takes
        # MINUTES per sample and would dominate the rollout.
        aux_device = torch.device("cpu") if config.aux_components_on_cpu else device
        vae_device = torch.device("cpu") if config.vae_components_on_cpu else device

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
        vae = vae.to(vae_device).eval()
        vae.requires_grad_(False)

        # Audio VAE (frozen, fp32). Do NOT let a global bf16 cast reach this:
        # the reference reports bf16 output roughly 20 dB too quiet.
        audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(
            vae_path, subfolder="audio_vae", torch_dtype=audio_vae_dtype
        )
        audio_vae = audio_vae.to(vae_device).eval()
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
            max_sequence_length=int(config.max_sequence_length),
        )
        if config.meta_init_transformer:
            # Diffusers layout: the backend's sharded loader reads the
            # safetensors under <ckpt>/transformer after `to_empty`.
            bundle._transformer_weights_path = transformer_weights_path
            bundle._meta_init_state = meta_init_state
        return bundle

    def trainable_module(self) -> nn.Module:
        return self.transformer


__all__ = ["MiniMaxH3Bundle"]
