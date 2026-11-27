"""Qwen3-30B-A3B vLLM model patch."""

from .patch import install_qwen3_moe_patch, preflight_qwen3_moe_patch

__all__ = ["install_qwen3_moe_patch", "preflight_qwen3_moe_patch"]
