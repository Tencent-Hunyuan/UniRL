"""Configuration for the direct text-only vLLM rollout engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import MISSING

from unirl.config.require import require
from unirl.rollout.engine.base import BaseEngineConfig


@dataclass
class VLLMEngineConfig(BaseEngineConfig):
    """Typed configuration for :class:`VLLMRolloutEngine`."""

    pretrained_model_ckpt_path: str = MISSING
    model_revision: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    tp_size: int = 1
    trust_remote_code: bool = False
    dtype: str = "bfloat16"
    moe_backend: str = "triton"
    quantization: Optional[str] = None
    enforce_eager: bool = True
    enable_prefix_caching: bool = False
    enable_chunked_prefill: bool = False
    logprobs_mode: str = "processed_logprobs"
    max_prompt_length: int = 4096
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    ignore_eos: bool = False
    request_timeout_s: float = 1800.0
    startup_timeout_s: Optional[float] = 600.0
    generate_timeout_s: Optional[float] = 1800.0
    weight_update_timeout_s: Optional[float] = 300.0
    health_timeout_s: Optional[float] = 30.0
    sleep_timeout_s: Optional[float] = 120.0
    wake_up_timeout_s: Optional[float] = 120.0
    shutdown_timeout_s: Optional[float] = 30.0
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
        require(
            self.tokenizer_revision is None or bool(str(self.tokenizer_revision).strip()),
            "VLLMEngineConfig.tokenizer_revision must be non-empty when set",
        )
        model_is_local = Path(str(self.pretrained_model_ckpt_path)).expanduser().exists()
        require(
            model_is_local or self.model_revision is not None,
            "Remote vLLM checkpoints require an explicit model_revision",
        )
        require(self.tp_size >= 1, f"VLLMEngineConfig.tp_size must be >= 1; got {self.tp_size}")
        require(self.dtype == "bfloat16", f"vLLM rollout currently supports dtype='bfloat16'; got {self.dtype!r}")
        require(
            self.moe_backend == "triton",
            f"vLLM rollout currently supports moe_backend='triton'; got {self.moe_backend!r}",
        )
        require(self.quantization is None, "vLLM native IPC weight reload does not support quantization")
        require(self.enforce_eager, "vLLM native IPC weight reload requires enforce_eager=true")
        require(not self.enable_prefix_caching, "vLLM native IPC weight reload requires prefix caching disabled")
        require(not self.enable_chunked_prefill, "vLLM rollout currently requires chunked prefill disabled")
        require(
            self.logprobs_mode == "processed_logprobs",
            f"vLLM rollout requires logprobs_mode='processed_logprobs'; got {self.logprobs_mode!r}",
        )
        reserved = {
            "distributed_executor_backend",
            "enable_sleep_mode",
            "worker_extension_cls",
            "weight_transfer_config",
            "trust_remote_code",
            "revision",
            "tokenizer_revision",
            "dtype",
            "quantization",
            "moe_backend",
            "enforce_eager",
            "enable_prefix_caching",
            "enable_chunked_prefill",
            "logprobs_mode",
        }
        conflicts = sorted(reserved.intersection(self.engine_kwargs))
        require(not conflicts, f"VLLMEngineConfig.engine_kwargs contains reserved keys: {conflicts}")
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
            "init_native_weight_transfer": "weight_update_timeout_s",
            "start_native_weight_update": "weight_update_timeout_s",
            "update_native_weights": "weight_update_timeout_s",
            "finish_native_weight_update": "weight_update_timeout_s",
            "release_native_ipc": "weight_update_timeout_s",
            "health": "health_timeout_s",
            "sleep": "sleep_timeout_s",
            "wake_up": "wake_up_timeout_s",
            "shutdown": "shutdown_timeout_s",
        }.get(command)
        value = getattr(self, field_name) if field_name is not None else None
        return float(self.request_timeout_s if value is None else value)


__all__ = ["VLLMEngineConfig"]
