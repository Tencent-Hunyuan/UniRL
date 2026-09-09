"""Configuration for the direct text-only vLLM rollout engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from omegaconf import MISSING

from unirl.config.require import require
from unirl.rollout.engine.base import BaseEngineConfig


@dataclass
class VLLMEngineConfig(BaseEngineConfig):
    """Typed configuration for :class:`VLLMRolloutEngine`."""

    pretrained_model_ckpt_path: str = MISSING
    tp_size: int = 1
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    ignore_eos: bool = False
    request_timeout_s: float = 1800.0
    system_instruction: Optional[str] = None
    chat_template_kwargs: Dict[str, Any] = field(default_factory=dict)
    engine_kwargs: Dict[str, Any] = field(default_factory=dict)

    def make_engine(self, **deps: Any):
        from unirl.rollout.engine.vllm.engine import VLLMRolloutEngine

        return VLLMRolloutEngine(config=self, **deps)

    def __post_init__(self) -> None:
        require(
            bool(self.pretrained_model_ckpt_path),
            "VLLMEngineConfig.pretrained_model_ckpt_path must be set",
        )
        require(self.tp_size >= 1, f"VLLMEngineConfig.tp_size must be >= 1; got {self.tp_size}")
        require(self.max_new_tokens >= 1, "VLLMEngineConfig.max_new_tokens must be >= 1")
        require(self.temperature > 0, "VLLMEngineConfig.temperature must be > 0")
        require(0.0 < self.top_p <= 1.0, "VLLMEngineConfig.top_p must be in (0, 1]")
        require(self.request_timeout_s > 0, "VLLMEngineConfig.request_timeout_s must be > 0")


__all__ = ["VLLMEngineConfig"]
