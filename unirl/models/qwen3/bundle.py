"""Qwen3Bundle — concrete weights+tokenizer holder for a Qwen3 causal LM."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from unirl.models.types.bundle import Bundle
from unirl.models.types.meta_init import build_meta_init_transformer
from unirl.models.types.value_head import ValueHead
from unirl.utils.dtypes import parse_torch_dtype

from .config import Qwen3PipelineConfig

logger = logging.getLogger(__name__)


def _stamp_value_head_reset(transformer: nn.Module) -> None:
    """Zero checkpoint-absent value-head params after meta materialization."""
    from unirl.models.types.post_materialize import defer_after_materialize

    def _reset(model: nn.Module) -> None:
        reset: list[str] = []
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name.startswith("value_head."):
                    param.zero_()
                    reset.append(name)
        if not reset:
            raise RuntimeError("Qwen3 meta-init: value_head parameters disappeared before post-load reset")
        logger.info("Qwen3 meta-init: zero-initialized checkpoint-absent value head: %s", reset)

    defer_after_materialize(transformer, _reset)


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
            # Restore non-persistent RoPE buffers after meta initialization.
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

        if config.use_value_head:
            hidden_size = int(getattr(transformer.config, "hidden_size"))
            transformer_device = next(transformer.parameters()).device
            transformer.value_head = ValueHead(hidden_size, device=transformer_device)
            if config.meta_init_transformer:
                _stamp_value_head_reset(transformer)

        bundle = cls(
            transformer=transformer,
            tokenizer=tokenizer,
            dtype=dtype,
            device=device,
            pretrained_path=path,
        )
        if config.meta_init_transformer:
            bundle._transformer_weights_path = path
            bundle._meta_init_state = meta_init_state
        return bundle


__all__ = ["Qwen3Bundle"]
