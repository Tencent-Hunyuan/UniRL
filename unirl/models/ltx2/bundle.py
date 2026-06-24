"""LTX2Bundle — weights holder for LTX-2 / LTX-2.3 video diffusion.

Loads the LTX2 transformer, video VAE, text encoder (Gemma3), connectors,
scheduler, and optionally audio VAE + vocoder for LTX-2.3.

No LoRA injection, FSDP wrap, or weight-sync logic — those are lifecycle
concerns owned outside the bundle.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from unirl.models.types.bundle import Bundle
from unirl.utils.dtypes import parse_torch_dtype

from .config import LTX2PipelineConfig


class LTX2Bundle(Bundle):
    """LTX-2/2.3 bundle: transformer + video VAE + text encoder + scheduler
    (+ optional audio VAE + vocoder for T2AV).
    """

    def __init__(
        self,
        *,
        transformer: nn.Module,
        vae: nn.Module,
        text_encoder: nn.Module,
        tokenizer: Any,
        connectors: Optional[nn.Module],
        scheduler: Any,
        dtype: torch.dtype,
        device: torch.device,
        pretrained_path: str,
        max_sequence_length: int,
        audio_vae: Optional[nn.Module] = None,
        vocoder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.connectors = connectors
        self.scheduler = scheduler
        self.dtype = dtype
        self.device = device
        self.pretrained_path = pretrained_path
        self.max_sequence_length = max_sequence_length
        self.audio_vae = audio_vae
        self.vocoder = vocoder

    @property
    def has_audio(self) -> bool:
        """True iff the bundle loaded audio components (LTX-2.3 T2AV mode)."""
        return self.audio_vae is not None

    @classmethod
    def from_config(cls, config: LTX2PipelineConfig) -> "LTX2Bundle":
        """Load all LTX-2 components from a HuggingFace checkpoint."""
        from diffusers import AutoencoderKLLTX2Video, LTX2VideoTransformer3DModel
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
        from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

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

        # Transformer (trainable)
        transformer = LTX2VideoTransformer3DModel.from_pretrained(path, subfolder="transformer", torch_dtype=dtype)
        transformer = transformer.to(device, dtype=dtype)

        # Video VAE (frozen)
        vae = AutoencoderKLLTX2Video.from_pretrained(vae_path, subfolder="vae", torch_dtype=vae_dtype).to(device).eval()
        vae.requires_grad_(False)

        # Text encoder — Gemma3 (frozen). LTX-2 uses Gemma-3-12B whose config is
        # nested (text_config/vision_config); loading it with the v1
        # GemmaForCausalLM class crashes in GenerationConfig.from_model_config
        # ('dict' has no attribute 'to_dict'). Match diffusers' LTX2 pipeline,
        # which uses Gemma3ForConditionalGeneration.
        text_encoder = (
            Gemma3ForConditionalGeneration.from_pretrained(te_path, subfolder="text_encoder", torch_dtype=te_dtype)
            .to(device)
            .eval()
        )
        text_encoder.requires_grad_(False)

        tokenizer = AutoTokenizer.from_pretrained(te_path, subfolder="tokenizer")

        # Connectors (Gemma per-layer hidden states → video/audio text
        # embeddings, frozen). REQUIRED for LTX-2.0: the DiT was trained on
        # connector outputs, not raw Gemma hidden states — diffusers declares
        # ``connectors`` in the pipeline's component sequence. The class is
        # ``LTX2TextConnectors`` (NOT ``LTX2Connector``, which does not exist).
        from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors

        connectors = (
            LTX2TextConnectors.from_pretrained(path, subfolder="connectors", torch_dtype=dtype).to(device).eval()
        )
        connectors.requires_grad_(False)

        # Scheduler
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(path, subfolder="scheduler")

        # Optional audio components (LTX-2.3)
        audio_vae: Optional[nn.Module] = None
        vocoder: Optional[nn.Module] = None
        if config.enable_audio:
            try:
                from diffusers import AutoencoderKLLTX2Audio
                from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder

                audio_vae = (
                    AutoencoderKLLTX2Audio.from_pretrained(path, subfolder="audio_vae", torch_dtype=vae_dtype)
                    .to(device)
                    .eval()
                )
                audio_vae.requires_grad_(False)

                vocoder = LTX2Vocoder.from_pretrained(path, subfolder="vocoder", torch_dtype=dtype).to(device).eval()
                vocoder.requires_grad_(False)
            except (ImportError, OSError) as e:
                raise RuntimeError(
                    f"LTX2Bundle: enable_audio=True but audio components not available: {e}. "
                    f"Ensure the checkpoint at {path} is an LTX-2.3 model with audio_vae/ and vocoder/ subfolders."
                ) from e

        return cls(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            connectors=connectors,
            scheduler=scheduler,
            dtype=dtype,
            device=device,
            pretrained_path=path,
            max_sequence_length=int(config.max_sequence_length),
            audio_vae=audio_vae,
            vocoder=vocoder,
        )


__all__ = ["LTX2Bundle"]
