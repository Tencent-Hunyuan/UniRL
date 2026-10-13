"""Configuration for the native SD3 rollout worker."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from unirl.config.require import require
from unirl.rollout.engine.base import BaseEngineConfig, RolloutCapabilities

_DEFAULT_FP8_ENABLED = True


@dataclass
class NativeSD3EngineConfig(BaseEngineConfig):
    """Configure one worker-owned native SD3 rollout pipeline."""

    forward_batch_size: Optional[int] = None
    fp8_enabled: bool = _DEFAULT_FP8_ENABLED
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
    bf16_prefix_steps: int = 0
    bf16_suffix_steps: int = 0
    compile_model: bool = True
    compile_mode: str = "max-autotune-no-cudagraphs"

    @classmethod
    def resolve_rollout_capabilities(
        cls,
        config: Mapping[str, object],
        *,
        track_prefix: str = "",
    ) -> RolloutCapabilities:
        """Resolve dynamic capabilities from the uninstantiated Hydra config."""
        del track_prefix
        fp8_enabled = config.get("fp8_enabled", _DEFAULT_FP8_ENABLED)
        if not isinstance(fp8_enabled, bool):
            raise TypeError(f"fp8_enabled must be a bool, got {fp8_enabled!r}")
        precisions = frozenset({"bf16", "fp8"}) if fp8_enabled else frozenset({"bf16"})
        return RolloutCapabilities(
            rollout_precisions=precisions,
            reward_image_resize=True,
            transactional_weight_publication=True,
        )

    def __post_init__(self) -> None:
        for name in ("fp8_enabled", "compile_model"):
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a bool, got {value!r}")
        for name, minimum in (
            ("fp8_min_dim", 0),
            ("bf16_prefix_steps", 0),
            ("bf16_suffix_steps", 0),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer, got {value!r}")
            require(value >= minimum, f"{name} must be >={minimum}, got {value}")
        if self.forward_batch_size is not None and (
            isinstance(self.forward_batch_size, bool) or not isinstance(self.forward_batch_size, int)
        ):
            raise TypeError(f"forward_batch_size must be an integer when set, got {self.forward_batch_size!r}")
        require(
            self.forward_batch_size is None or self.forward_batch_size >= 1,
            f"forward_batch_size must be >=1 when set, got {self.forward_batch_size!r}",
        )
        require(self.fp8_recipe in {"current", "delayed"}, "fp8_recipe must be current|delayed")
        if not isinstance(self.fp8_skip_modules, (list, tuple)) or not all(
            isinstance(pattern, str) and pattern for pattern in self.fp8_skip_modules
        ):
            raise TypeError("fp8_skip_modules must be a list/tuple of non-empty strings")
        self.fp8_skip_modules = tuple(self.fp8_skip_modules)
        if not isinstance(self.compile_mode, str) or not self.compile_mode:
            raise TypeError(f"compile_mode must be a non-empty string, got {self.compile_mode!r}")

    def make_engine(self, **deps: Any):
        from .engine import NativeSD3RolloutEngine

        return NativeSD3RolloutEngine(config=self, **deps)


__all__ = ["NativeSD3EngineConfig"]
