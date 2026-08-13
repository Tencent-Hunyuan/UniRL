"""Qwen3MoeBundle — VeOmni-patched Qwen3-MoE causal LM + tokenizer."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from unirl.models.types.bundle import Bundle
from unirl.models.types.meta_init import finalize_meta_init
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
    def from_config(
        cls,
        config: Any = None,
        *,
        pretrained_model_ckpt_path: Optional[str] = None,
        tokenizer_ckpt_path: Optional[str] = None,
        model_precision: str = "bf16",
        moe_implementation: str = "fused_triton",
        attn_implementation: str = "flash_attention_2",
        meta_init_transformer: bool = True,
        trust_remote_code: bool = True,
        tokenizer: Any = None,
    ) -> "Qwen3MoeBundle":
        """Build the VeOmni Qwen3-MoE transformer (on meta) + tokenizer."""
        if config is not None:
            pretrained_model_ckpt_path = config.pretrained_model_ckpt_path
            tokenizer_ckpt_path = getattr(config, "tokenizer_ckpt_path", None)
            model_precision = getattr(config, "model_precision", "bf16")
            attn_implementation = getattr(config, "attn_implementation", None) or attn_implementation
            moe_implementation = getattr(config, "moe_implementation", None) or moe_implementation
            meta_init_transformer = bool(getattr(config, "meta_init_transformer", True))
            trust_remote_code = bool(getattr(config, "trust_remote_code", True))
        if pretrained_model_ckpt_path is None:
            raise ValueError("Qwen3MoeBundle.from_config: pretrained_model_ckpt_path is required")
        if not meta_init_transformer:
            raise ValueError(
                "Qwen3MoeBundle requires meta_init_transformer=true: VeOmniBackend "
                "materializes and loads this EP model only after parallelization."
            )

        from unirl.train.backend.veomni import _compat

        _compat.ensure_qwen3_moe_installed()
        from veomni.arguments import OpsImplementationConfig
        from veomni.models.auto import build_foundation_model

        dtype = parse_torch_dtype(str(model_precision), field_name="Qwen3MoeBundle.model_precision")
        dtype_name = canonical_torch_dtype_name(dtype, field_name="Qwen3MoeBundle.model_precision")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ops = OpsImplementationConfig(
            attn_implementation=attn_implementation,
            moe_implementation=moe_implementation,
            cross_entropy_loss_implementation="eager",
            rms_norm_implementation="eager",
            swiglu_mlp_implementation="eager",
            rotary_pos_emb_implementation="eager",
            load_balancing_loss_implementation="eager",
        )
        transformer = build_foundation_model(
            config_path=pretrained_model_ckpt_path,
            weights_path=None,
            torch_dtype=dtype_name,
            init_device="meta",
            ops_implementation=ops,
        )
        transformer = finalize_meta_init(transformer, dtype=dtype)

        if tokenizer is None:
            from transformers import AutoTokenizer

            tok_path = tokenizer_ckpt_path or pretrained_model_ckpt_path
            tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=trust_remote_code)
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token

        bundle = cls(
            transformer=transformer,
            tokenizer=tokenizer,
            dtype=dtype,
            device=device,
            pretrained_path=pretrained_model_ckpt_path,
        )
        bundle._transformer_weights_path = pretrained_model_ckpt_path
        return bundle


__all__ = ["Qwen3MoeBundle"]
