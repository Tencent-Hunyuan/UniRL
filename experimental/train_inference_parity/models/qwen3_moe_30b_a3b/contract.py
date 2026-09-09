"""Frozen structural contract for the first parity model profile."""

from __future__ import annotations

from dataclasses import dataclass


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


def validate_model_config(config) -> None:
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


__all__ = [
    "QWEN3_MOE_30B_A3B_CONTRACT",
    "Qwen3MoeContract",
    "validate_model_config",
]
