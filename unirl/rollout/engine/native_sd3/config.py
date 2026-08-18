"""Configuration for the native SD3 rollout worker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

from unirl.config.require import require
from unirl.rollout.engine.base import BaseEngineConfig


@dataclass
class NativeSD3EngineConfig(BaseEngineConfig):
    """Configure one worker-owned native SD3 rollout pipeline."""

    forward_batch_size: Optional[int] = None
    fp8_enabled: bool = True
    fp8_recipe: str = "current"
    fp8_skip_modules: Tuple[str, ...] = (
        "pos_embed",
        "context_embedder",
        "time_text_embed",
        "norm_out",
        "norm1",
        "ff_context",
    )
    fp8_min_dim: int = 2432
    fp8_dim_multiple: int = 16
    bf16_prefix_steps: int = 0
    bf16_suffix_steps: int = 0
    compile_model: bool = True
    compile_mode: str = "max-autotune-no-cudagraphs"

    def __post_init__(self) -> None:
        require(
            self.forward_batch_size is None or int(self.forward_batch_size) >= 1,
            f"forward_batch_size must be >=1 when set, got {self.forward_batch_size!r}",
        )
        require(self.fp8_recipe in {"current", "delayed"}, "fp8_recipe must be current|delayed")
        require(int(self.fp8_min_dim) >= 0, f"fp8_min_dim must be >=0, got {self.fp8_min_dim}")
        require(
            int(self.fp8_dim_multiple) >= 1,
            f"fp8_dim_multiple must be >=1, got {self.fp8_dim_multiple}",
        )
        require(
            int(self.bf16_prefix_steps) >= 0 and int(self.bf16_suffix_steps) >= 0,
            "bf16_prefix_steps and bf16_suffix_steps must be non-negative",
        )

    def make_engine(self, **deps: Any):
        from .engine import NativeSD3RolloutEngine

        return NativeSD3RolloutEngine(config=self, **deps)


__all__ = ["NativeSD3EngineConfig"]
