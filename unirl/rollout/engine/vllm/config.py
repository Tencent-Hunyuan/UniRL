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
    model_revision: Optional[str] = None
    tp_size: int = 1
    trust_remote_code: bool = False
    max_prompt_length: int = 4096
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    ignore_eos: bool = False
    # Compatibility fallback for callers that have not split their timeout
    # budget yet. A command-specific value always takes precedence.
    request_timeout_s: float = 1800.0
    startup_timeout_s: Optional[float] = None
    generate_timeout_s: Optional[float] = None
    weight_update_timeout_s: Optional[float] = None
    health_timeout_s: Optional[float] = None
    sleep_timeout_s: Optional[float] = None
    wake_up_timeout_s: Optional[float] = None
    shutdown_timeout_s: Optional[float] = None
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
        require(
            self.model_revision is None or bool(str(self.model_revision).strip()),
            "VLLMEngineConfig.model_revision must be non-empty when set",
        )
        require(self.tp_size >= 1, f"VLLMEngineConfig.tp_size must be >= 1; got {self.tp_size}")
        require(self.max_prompt_length >= 1, "VLLMEngineConfig.max_prompt_length must be >= 1")
        require(self.max_new_tokens >= 1, "VLLMEngineConfig.max_new_tokens must be >= 1")
        require(self.temperature > 0, "VLLMEngineConfig.temperature must be > 0")
        require(0.0 < self.top_p <= 1.0, "VLLMEngineConfig.top_p must be in (0, 1]")
        require(self.top_k >= 0, "VLLMEngineConfig.top_k must be >= 0")
        require(self.request_timeout_s > 0, "VLLMEngineConfig.request_timeout_s must be > 0")
        for name in (
            "startup_timeout_s",
            "generate_timeout_s",
            "weight_update_timeout_s",
            "health_timeout_s",
            "sleep_timeout_s",
            "wake_up_timeout_s",
            "shutdown_timeout_s",
        ):
            value = getattr(self, name)
            require(value is None or value > 0, f"VLLMEngineConfig.{name} must be > 0 when set")

    def timeout_for(self, command: str) -> float:
        """Return a command timeout, falling back to the legacy shared value."""
        field_name = {
            "startup": "startup_timeout_s",
            "generate": "generate_timeout_s",
            "update_weights": "weight_update_timeout_s",
            "health": "health_timeout_s",
            "sleep": "sleep_timeout_s",
            "wake_up": "wake_up_timeout_s",
            "shutdown": "shutdown_timeout_s",
        }.get(command)
        value = getattr(self, field_name) if field_name is not None else None
        return float(self.request_timeout_s if value is None else value)


__all__ = ["VLLMEngineConfig"]
