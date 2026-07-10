"""Cosmos3Bundle — plain weight holder for Cosmos3 SFT.

Loads the diffusers-layout checkpoint (``transformer/``, ``vae/``,
``text_tokenizer/``, ``scheduler/``) and applies the freeze policy. FSDP
wrapping, optimizer, and checkpointing are owned by the backend
(:class:`unirl.train.backend.fsdp.FSDPBackend` over ``trainable_attr="transformer"``).

Requires diffusers >= 0.39 (first release with ``Cosmos3OmniTransformer`` /
``Cosmos3OmniPipeline``).
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import torch

from unirl.models.cosmos3.config import Cosmos3SFTConfig
from unirl.utils.dtypes import parse_torch_dtype

logger = logging.getLogger(__name__)

# Full-name patterns of the understanding (AR/text) parameter set inside
# Cosmos3OmniTransformer. Everything else — add_{q,k,v}_proj / to_add_out,
# mlp_moe_gen.*, *_moe_gen norms, proj_in/proj_out, time_embedder, and the
# action/audio modality heads — is the generation stream and stays trainable.
# NB ``to_out`` must not swallow ``to_add_out`` and ``norm.`` must not swallow
# ``norm_moe_gen.`` — hence the trailing ``\.`` anchors.
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
    """True if ``name`` (a ``named_parameters`` key of Cosmos3OmniTransformer)
    belongs to the frozen-by-default understanding stream."""
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


class Cosmos3Bundle:
    """Weights for Cosmos3 SFT: MoT transformer (trainable), WanVAE + tokenizer
    + scheduler (frozen helpers)."""

    def __init__(self, *, transformer, vae, text_tokenizer, scheduler, config: Cosmos3SFTConfig) -> None:
        self.transformer = transformer
        self.vae = vae
        self.text_tokenizer = text_tokenizer
        self.scheduler = scheduler
        self.config = config
        self._pipeline = None

    @classmethod
    def from_config(cls, config: Cosmos3SFTConfig) -> "Cosmos3Bundle":
        Cosmos3OmniTransformer, AutoencoderKLWan, UniPCMultistepScheduler, _ = _import_diffusers_classes()
        path = config.pretrained_model_ckpt_path
        device = torch.device(config.device)
        model_dtype = parse_torch_dtype(config.model_precision, field_name="model_precision")
        vae_dtype = parse_torch_dtype(config.vae_precision, field_name="vae_precision")

        transformer = Cosmos3OmniTransformer.from_pretrained(
            path, subfolder="transformer", torch_dtype=model_dtype
        ).to(device)
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

        return cls(
            transformer=transformer,
            vae=vae,
            text_tokenizer=text_tokenizer,
            scheduler=scheduler,
            config=config,
        )

    @property
    def flow_shift(self) -> float:
        """Effective flow shift: config override, else the checkpoint scheduler's."""
        if self.config.flow_shift is not None:
            return float(self.config.flow_shift)
        return float(getattr(self.scheduler.config, "flow_shift", 5.0))

    def build_pipeline(self):
        """The real ``Cosmos3OmniPipeline`` over this bundle's components.

        Used both for its packing helpers (tokenize_prompt / _prepare_*_segment /
        _encode_video — training mirrors inference bit-for-bit through them) and
        for eval sampling. Safety checker disabled: SFT prompts come from local
        datasets, and the guardrail model download/latency has no place in the
        training loop.
        """
        if self._pipeline is None:
            *_, Cosmos3OmniPipeline = _import_diffusers_classes()
            self._pipeline = Cosmos3OmniPipeline(
                transformer=self.transformer,
                text_tokenizer=self.text_tokenizer,
                vae=self.vae,
                scheduler=self.scheduler,
                sound_tokenizer=None,
                safety_checker=None,
                enable_safety_checker=False,
            )
        return self._pipeline


__all__ = ["Cosmos3Bundle", "is_understanding_param"]
