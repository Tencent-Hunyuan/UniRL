"""Configuration for the vLLM-Omni rollout engine.

One knob (``modality``) drives YAML selection and request validation:

- ``"t2i"``     — HI3, AR (CoT think) → DiT denoise → image. Two stages.
- ``"it2i"``    — HI3, AR (image+text → CoT/recaption) → DiT → edited image. Two stages.
- ``"i2t"``     — HI3, image+text → AR text. Single stage (AR only).
- ``"t2t"``     — HI3, text → AR text. Single stage (AR only).
- ``"sd3_t2i"`` — SD3, text → DiT denoise → image. Single diffusion stage.
- ``"t2v"``     — HunyuanVideo-1.5, text → DiT denoise → video. Single diffusion stage.

The image modalities install our ``RLHunyuanImage3Pipeline`` subclass via
the static YAMLs in ``stage_configs/`` (per-stage
``engine_args.custom_pipeline_args``). The AR-only modalities use upstream
YAMLs unchanged.

Consumed by ``VLLMOmniRolloutEngine``; the recipe wires ``rollout: {_target_:
...VLLMOmniRolloutEngine, config: {_target_: ...VLLMOmniEngineConfig, ...}}`` and
the rollout actor constructs the engine with ``config=<this>`` + ``device`` /
``strategy`` / ``rank`` / ``model_config``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from omegaconf import MISSING

from unirl.rollout.engine.base import BaseEngineConfig


@dataclass
class VLLMOmniEngineConfig(BaseEngineConfig):
    def make_engine(self, **deps: Any):
        from unirl.rollout.engine.vllm_omni.engine import VLLMOmniRolloutEngine

        return VLLMOmniRolloutEngine(config=self, **deps)

    # Required: HunyuanImage-3 checkpoint path. Set per experiment or via
    # ``cfg.rollout.engine.model_path=...`` on the CLI.
    model_path: str = MISSING
    # Valid values: "t2i" | "it2i" | "i2t" | "t2t" | "sd3_t2i" | "t2v". Kept as ``str``
    # because OmegaConf structured configs reject ``Literal[...]`` annotations;
    # the engine ctor validates the string against the supported modality set.
    modality: str = "t2i"

    # DiT-side defaults (image modalities only). ``default_eta=1.0`` puts
    # SDE on by default; pass ``"eta": 0.0`` per request for the
    # deterministic ODE path.
    default_height: int = 1024
    default_width: int = 1024
    default_num_inference_steps: int = 25
    default_guidance_scale: float = 5.0
    default_eta: float = 1.0

    # AR-side defaults (all modalities).
    default_ar_max_tokens: int = 2048
    default_ar_temperature: float = 0.6
    default_ar_top_p: float = 0.95
    default_ar_top_k: int = 1024

    # Inject ``enable_sleep_mode: True`` into each stage's ``engine_args`` at
    # engine ``__init__`` time so worker.sleep()/wake_up() (level 2) can run.
    # Disable to fall back to the upstream YAML defaults (CuMemAllocator pool
    # off, sleep raises). Required for ``cfg.training.execution.offload_rollout
    # = True`` to do anything for this engine.
    enable_sleep_mode: bool = True

    # Passthrough for advanced ``Omni`` kwargs not surfaced as typed fields.
    omni_extra: Dict[str, Any] = field(default_factory=dict)


__all__ = ["VLLMOmniEngineConfig"]
