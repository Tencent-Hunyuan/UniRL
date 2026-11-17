"""Qwen3MoeBundle — VeOmni-patched Qwen3-MoE causal LM + tokenizer."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from unirl.models.qwen3.config import Qwen3PipelineConfig
from unirl.models.types.bundle import Bundle
from unirl.models.types.meta_init import capture_init_state, finalize_meta_init, resolve_meta_init_weights
from unirl.utils.dtypes import canonical_torch_dtype_name, parse_torch_dtype


class Qwen3MoeBundle(Bundle):
    """VeOmni Qwen3-MoE transformer + tokenizer (meta-init, EP-capable)."""

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

    def prepare_for_expert_parallel(self) -> None:
        """Backend hook when ``ep_size > 1``."""
        if not callable(getattr(self.transformer, "get_parallel_plan", None)):
            raise ValueError(
                "Qwen3MoeBundle.prepare_for_expert_parallel: transformer lacks "
                "get_parallel_plan(); rebuild with moe_implementation=fused_triton"
            )

    @classmethod
    def from_config(cls, config: Qwen3PipelineConfig) -> "Qwen3MoeBundle":
        """Build the VeOmni Qwen3-MoE transformer (on meta) + tokenizer."""
        path = config.pretrained_model_ckpt_path
        if not config.meta_init_transformer:
            raise ValueError(
                "Qwen3MoeBundle requires meta_init_transformer=true: VeOmniBackend "
                "materializes and loads this EP model only after parallelization."
            )
        transformer_weights_path = resolve_meta_init_weights(path)

        from unirl.train.backend.veomni import _compat

        _compat.ensure_qwen3_moe_installed()
        from veomni.arguments import OpsImplementationConfig
        from veomni.models.auto import build_foundation_model

        dtype = parse_torch_dtype(config.model_precision, field_name="Qwen3MoeBundle.model_precision")
        dtype_name = canonical_torch_dtype_name(dtype, field_name="Qwen3MoeBundle.model_precision")
        device = config.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, str):
            device = torch.device(device)

        ops = OpsImplementationConfig(
            attn_implementation=config.attn_implementation or "flash_attention_2",
            moe_implementation="fused_triton",
            cross_entropy_loss_implementation="eager",
            rms_norm_implementation="eager",
            swiglu_mlp_implementation="eager",
            rotary_pos_emb_implementation="eager",
            load_balancing_loss_implementation="eager",
        )
        transformer = build_foundation_model(
            config_path=path,
            weights_path=None,
            torch_dtype=dtype_name,
            init_device="meta",
            ops_implementation=ops,
        )
        meta_init_state = capture_init_state(transformer)
        transformer = finalize_meta_init(transformer, dtype=dtype)

        tok_path = config.tokenizer_ckpt_path or path
        tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=config.trust_remote_code)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        bundle = cls(
            transformer=transformer,
            tokenizer=tokenizer,
            dtype=dtype,
            device=device,
            pretrained_path=path,
        )
        bundle._transformer_weights_path = transformer_weights_path
        bundle._meta_init_state = meta_init_state
        return bundle


__all__ = ["Qwen3MoeBundle"]
