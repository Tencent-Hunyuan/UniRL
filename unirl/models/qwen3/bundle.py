"""Qwen3Bundle — concrete weights+tokenizer holder for a Qwen3 causal LM.

Implements the empty :class:`Bundle` Protocol
(:mod:`unirl.models.types.bundle`). Pure container of:

- ``transformer`` — HuggingFace :class:`AutoModelForCausalLM` loaded with
  ``trust_remote_code=True`` (required for Qwen3's custom modeling code).
- ``tokenizer`` — matching HuggingFace :class:`AutoTokenizer`. ``pad_token``
  is set to ``eos_token`` when absent (decoder-only models commonly skip
  defining a pad token; the chat-template stage right-pads in-batch and
  needs a valid pad id).

No VAE / text encoder / scheduler — Qwen3 is a pure causal LM with no
diffusion side. Lifecycle concerns (LoRA injection, FSDP wrapping,
adapter switching, autocast helpers, weight-sync logic) live outside the
bundle in ``cfg.training.policies`` per the new design.

Use :meth:`Qwen3Bundle.from_config` to load a checkpoint.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from unirl.models.types.bundle import Bundle
from unirl.models.types.meta_init import build_meta_init_transformer
from unirl.utils.dtypes import parse_torch_dtype

from .config import Qwen3PipelineConfig

logger = logging.getLogger(__name__)


class Qwen3Bundle(Bundle):
    """Qwen3 bundle: causal-LM transformer + matching tokenizer."""

    def __init__(
        self,
        *,
        transformer: nn.Module,
        tokenizer: Any,
        dtype: torch.dtype,
        device: torch.device,
        pretrained_path: str,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.dtype = dtype
        self.device = device
        self.pretrained_path = pretrained_path

    @classmethod
    def from_config(cls, config: Qwen3PipelineConfig) -> "Qwen3Bundle":
        """Load the Qwen3 transformer + tokenizer from a HuggingFace-layout checkpoint."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = config.pretrained_model_ckpt_path
        tokenizer_path = config.tokenizer_ckpt_path or path

        device = config.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, str):
            device = torch.device(device)

        dtype = parse_torch_dtype(config.model_precision, field_name="model_precision")

        if config.meta_init_transformer:
            # Meta-init (FSDP / VeOmni load_sharded path): parameters on the meta
            # device, materialized + loaded by the backend from the checkpoint
            # root after sharding (AR layout: no transformer/ subfolder, so the
            # stashed dir is the root). build_meta_init_transformer keeps
            # buffers/attrs real on CPU: HF rotary inv_freq / original_inv_freq
            # are non-persistent buffers computed in __init__ and absent from the
            # checkpoint, so to_empty later clobbers them -> garbage RoPE. It
            # captures them; meta_init_state is stashed on the BUNDLE below and
            # restored by load_trainable_weights after the sharded weight load.
            from transformers import AutoConfig

            hf_config = AutoConfig.from_pretrained(path, trust_remote_code=bool(config.trust_remote_code))
            transformer, meta_init_state = build_meta_init_transformer(
                lambda: AutoModelForCausalLM.from_config(hf_config, trust_remote_code=bool(config.trust_remote_code)),
                dtype=dtype,
            )
        else:
            load_kwargs = {}
            if getattr(config, "attn_implementation", None):
                load_kwargs["attn_implementation"] = str(config.attn_implementation)
            transformer = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=dtype,
                trust_remote_code=bool(config.trust_remote_code),
                **load_kwargs,
            ).to(device)

        # Structural (no weight access); runs on both the meta and eager builds
        # and persists through to_empty + load.
        if config.use_gradient_checkpointing:
            if hasattr(transformer, "gradient_checkpointing_enable"):
                transformer.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            else:
                logger.warning(
                    "Qwen3 transformer %s does not expose gradient_checkpointing_enable; "
                    "skipping use_gradient_checkpointing=True.",
                    type(transformer).__name__,
                )

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=bool(config.trust_remote_code),
        )
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        bundle = cls(
            transformer=transformer,
            tokenizer=tokenizer,
            dtype=dtype,
            device=device,
            pretrained_path=path,
        )
        if config.meta_init_transformer:
            # AR checkpoints store *.safetensors at the root (no subfolder).
            bundle._transformer_weights_path = path
            # Ray-robust restore carrier for init-computed non-persistent state.
            bundle._meta_init_state = meta_init_state
        return bundle


__all__ = ["Qwen3Bundle"]
