"""``vllm_omni`` engine config + the typed port set it self-reserves."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from omegaconf import MISSING

from unirl.config.require import require
from unirl.rollout.engine.base import BaseEngineConfig
from unirl.rollout.engine.ports import ReservedPorts


@dataclass(frozen=True)
class VLLMOmniPorts(ReservedPorts):
    """The master-port base one ``Omni`` spawn consumes."""

    master_port: int


@dataclass
class VLLMOmniEngineConfig(BaseEngineConfig):
    def make_engine(self, **deps: Any):
        from unirl.rollout.engine.vllm_omni.engine import VLLMOmniRolloutEngine

        return VLLMOmniRolloutEngine(config=self, **deps)

    model_path: str = MISSING
    modality: str = "hi3_t2i"

    enable_sleep_mode: bool = True

    stage_yaml_override: Optional[str] = None

    omni_extra: Dict[str, Any] = field(default_factory=dict)

    max_prompt_length: Optional[int] = None
    image_max_pixels: Optional[int] = None
    video_fps: Optional[float] = None
    video_max_pixels: Optional[int] = None
    use_audio_in_video: Optional[bool] = None
    chat_template_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.modality = str(self.modality or "").strip().lower()
        from unirl.rollout.engine.vllm_omni.adapters import registered_adapters

        valid = registered_adapters()
        require(
            self.modality in valid,
            f"VLLMOmniEngineConfig.modality must be one of {set(valid)}; got {self.modality!r}",
        )

    def server_intent(
        self,
        *,
        model_config: Any,
        ports: Optional[VLLMOmniPorts],
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Spell this config (+ adapter boot extras + reserved ports) as boot intent."""
        del model_config
        extra = dict(extra or {})
        mode = extra.pop("mode", None)

        intent: Dict[str, Any] = {
            "model_path": str(self.model_path),
            "enable_sleep_mode": bool(self.enable_sleep_mode),
            "ports": ports,
        }
        intent.update(extra)

        if self.stage_yaml_override:
            intent["stage_yaml"] = str(self.stage_yaml_override)

        omni_kwargs: Dict[str, Any] = dict(
            stage_init_timeout=1200,
            init_timeout=1800,
        )
        if mode is not None:
            omni_kwargs["mode"] = mode
        omni_kwargs.update(self.omni_extra or {})
        intent["omni_kwargs"] = omni_kwargs
        return intent


__all__ = ["VLLMOmniPorts", "VLLMOmniEngineConfig"]
