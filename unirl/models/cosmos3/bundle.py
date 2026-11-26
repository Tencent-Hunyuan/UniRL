"""Cosmos3 SFT checkpoint bundle and freeze policy."""

from __future__ import annotations

import logging
import re

import torch

from unirl.models.cosmos3.config import Cosmos3SFTConfig
from unirl.models.types.bundle import Bundle
from unirl.models.types.meta_init import build_meta_init_transformer, resolve_meta_init_weights
from unirl.utils.dtypes import parse_torch_dtype

logger = logging.getLogger(__name__)

# Understanding (AR/text) parameter set; trailing ``\.`` anchors are load-bearing (README # Gotchas).
_UND_PARAM_PATTERNS = (
    r"^embed_tokens\.",
    r"^lm_head\.",
    r"^norm\.",
    r"^layers\.\d+\.self_attn\.(to_q|to_k|to_v|to_out|norm_q|norm_k)\.",
    r"^layers\.\d+\.mlp\.",
    r"^layers\.\d+\.(input_layernorm|post_attention_layernorm)\.",
)
_UND_PARAM_RE = re.compile("|".join(f"(?:{p})" for p in _UND_PARAM_PATTERNS))


def is_understanding_param(name: str) -> bool:
    """Return whether a Cosmos3 transformer parameter belongs to the understanding stream."""
    return _UND_PARAM_RE.match(name) is not None


def _import_diffusers_classes():
    try:
        from diffusers import AutoencoderKLWan, Cosmos3OmniTransformer, UniPCMultistepScheduler
        from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import Cosmos3OmniPipeline
    except ImportError as exc:  # pragma: no cover - version guard
        raise ImportError(
            "Cosmos3 support requires diffusers>=0.39 (Cosmos3OmniTransformer/"
            "Cosmos3OmniPipeline first shipped in 0.39.0)."
        ) from exc
    return Cosmos3OmniTransformer, AutoencoderKLWan, UniPCMultistepScheduler, Cosmos3OmniPipeline


class Cosmos3Bundle(Bundle):
    """Weights and frozen helpers for Cosmos3 SFT."""

    def __init__(self, *, transformer, vae, text_tokenizer, scheduler, config: Cosmos3SFTConfig) -> None:
        super().__init__()
        self.transformer = transformer
        self.vae = vae
        self.text_tokenizer = text_tokenizer
        self.scheduler = scheduler
        self.config = config
        self.dtype = parse_torch_dtype(config.model_precision, field_name="model_precision")
        self.device = torch.device(config.device)
        self.pretrained_path = config.pretrained_model_ckpt_path

    @classmethod
    def from_config(cls, config: Cosmos3SFTConfig) -> "Cosmos3Bundle":
        Cosmos3OmniTransformer, AutoencoderKLWan, UniPCMultistepScheduler, _ = _import_diffusers_classes()
        path = config.pretrained_model_ckpt_path
        device = torch.device(config.device)
        master_dtype = parse_torch_dtype(config.master_precision, field_name="master_precision")
        vae_dtype = parse_torch_dtype(config.vae_precision, field_name="vae_precision")

        # Uniform master-dtype storage (fp32 storage + bf16 FSDP compute: README # Gotchas).
        meta_init_state = None
        if config.meta_init_transformer:
            transformer_weights_path = resolve_meta_init_weights(path, component="transformer")
            transformer_config = Cosmos3OmniTransformer.load_config(path, subfolder="transformer")
            transformer, meta_init_state = build_meta_init_transformer(
                lambda: Cosmos3OmniTransformer.from_config(transformer_config),
                dtype=master_dtype,
                keep_in_fp32=Cosmos3OmniTransformer._keep_in_fp32_modules,
            )
        else:
            transformer = Cosmos3OmniTransformer.from_pretrained(
                path,
                subfolder="transformer",
                torch_dtype=master_dtype,
            )
            transformer = transformer.to(device)  # no dtype: from_pretrained kept time_embedder fp32 (README # Gotchas)
        vae = AutoencoderKLWan.from_pretrained(path, subfolder="vae", torch_dtype=vae_dtype).to(device)
        vae.requires_grad_(False)
        vae.eval()

        from transformers import AutoTokenizer

        text_tokenizer = AutoTokenizer.from_pretrained(path, subfolder="text_tokenizer")
        scheduler = UniPCMultistepScheduler.from_pretrained(path, subfolder="scheduler")

        if config.freeze_understanding:
            frozen = trainable = 0
            for name, param in transformer.named_parameters():
                if is_understanding_param(name):
                    param.requires_grad_(False)
                    frozen += param.numel()
                else:
                    trainable += param.numel()
            logger.info(
                "Cosmos3Bundle: froze und stream (%.2fB params); gen stream trainable (%.2fB params)",
                frozen / 1e9,
                trainable / 1e9,
            )

        bundle = cls(
            transformer=transformer,
            vae=vae,
            text_tokenizer=text_tokenizer,
            scheduler=scheduler,
            config=config,
        )
        if config.meta_init_transformer:
            bundle._transformer_weights_path = transformer_weights_path
            bundle._meta_init_state = meta_init_state
        return bundle

    @property
    def flow_shift(self) -> float:
        """Effective flow shift: config override, else the checkpoint scheduler's."""
        if self.config.flow_shift is not None:
            return self.config.flow_shift
        return float(getattr(self.scheduler.config, "flow_shift", 5.0))


__all__ = ["Cosmos3Bundle", "is_understanding_param"]
