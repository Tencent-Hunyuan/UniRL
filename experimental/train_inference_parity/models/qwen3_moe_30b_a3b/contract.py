"""Frozen structural contract for the first parity model profile."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Qwen3MoeContract:
    model_name: str
    hidden_size: int
    attention_heads: int
    key_value_heads: int
    head_dim: int
    experts: int
    topk: int
    intermediate_size: int
    vocab_size: int
    dtype: str


QWEN3_MOE_30B_A3B_CONTRACT = Qwen3MoeContract(
    model_name="Qwen3-30B-A3B",
    hidden_size=2048,
    attention_heads=32,
    key_value_heads=4,
    head_dim=128,
    experts=128,
    topk=8,
    intermediate_size=768,
    vocab_size=151936,
    dtype="bfloat16",
)


def _is_bfloat16(dtype: object) -> bool:
    if dtype is torch.bfloat16:
        return True
    return str(dtype).lower().removeprefix("torch.") in {"bf16", "bfloat16"}


def require_train_tp() -> int:
    """Read the explicit training TP contract; there is intentionally no default."""
    raw = os.environ.get("UNIRL_PARITY_TRAIN_TP")
    if raw is None or not raw.strip():
        raise RuntimeError("Qwen3 parity requires UNIRL_PARITY_TRAIN_TP to be set explicitly")
    try:
        train_tp = int(raw)
    except ValueError as error:
        raise ValueError(f"UNIRL_PARITY_TRAIN_TP must be a positive integer, got {raw!r}") from error
    if train_tp <= 0:
        raise ValueError(f"UNIRL_PARITY_TRAIN_TP must be a positive integer, got {raw!r}")
    return train_tp


def validate_parallel_dimensions(hidden_size: int, intermediate_size: int, train_tp: int) -> None:
    mismatches = {}
    if hidden_size % train_tp:
        mismatches["hidden_size"] = (hidden_size, train_tp)
    if intermediate_size % train_tp:
        mismatches["intermediate_size"] = (intermediate_size, train_tp)
    if mismatches:
        raise ValueError(f"Qwen3 parity dimensions must be divisible by train_tp: {mismatches}")


def validate_model_config(config, *, dtype: object | None = None) -> None:
    contract = QWEN3_MOE_30B_A3B_CONTRACT
    checks = {
        "hidden_size": contract.hidden_size,
        "num_attention_heads": contract.attention_heads,
        "num_key_value_heads": contract.key_value_heads,
        "head_dim": contract.head_dim,
        "num_experts": contract.experts,
        "num_experts_per_tok": contract.topk,
        "moe_intermediate_size": contract.intermediate_size,
        "vocab_size": contract.vocab_size,
    }
    mismatches = {
        name: (getattr(config, name, None), expected)
        for name, expected in checks.items()
        if getattr(config, name, None) != expected
    }
    if mismatches:
        raise ValueError(f"Qwen3-30B-A3B parity contract mismatch: {mismatches}")
    resolved_dtype = dtype
    if resolved_dtype is None:
        resolved_dtype = getattr(config, "dtype", getattr(config, "torch_dtype", None))
    if not _is_bfloat16(resolved_dtype):
        raise ValueError(f"Qwen3-30B-A3B parity requires BF16 model dtype, got {resolved_dtype!r}")


def validate_runtime_contract(config, *, dtype: object) -> int:
    validate_model_config(config, dtype=dtype)
    train_tp = require_train_tp()
    validate_parallel_dimensions(
        int(getattr(config, "hidden_size")),
        int(getattr(config, "moe_intermediate_size")),
        train_tp,
    )
    return train_tp


__all__ = [
    "QWEN3_MOE_30B_A3B_CONTRACT",
    "Qwen3MoeContract",
    "require_train_tp",
    "validate_model_config",
    "validate_parallel_dimensions",
    "validate_runtime_contract",
]
