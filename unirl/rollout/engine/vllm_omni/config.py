"""``vllm_omni`` engine config + the typed port set it self-reserves.

Ported from v1's ``VLLMOmniEngineConfig`` minus all port/placement math: the
engine reserves its own :class:`VLLMOmniPorts` base at boot (riding the
``Omni(master_port=...)`` ctor kwarg into every stage's ``engine_args``), so
there is no ``_VLLM_OMNI_PORT_BASE + rank * stride`` and no ``RANK``-env
fallback here. ``modality`` is validated against the live adapter registry
rather than the engine raising on an unknown YAML key at boot. v1's
``default_*`` sampling fallbacks are gone too: sampling values ride the
request's typed sampling params (the single source of truth), never the
engine config.

``server_intent`` (the successor of v1's inline YAML-injection + ``Omni``
kwargs assembly) spells this config + the reserved port base + the adapter's
boot extras as the intent dict ``VLLMOmniBackend.boot`` consumes; the seam
translates it into ``Omni`` ctor kwargs — the runtime's own override channel
— rather than rewriting the stage YAML.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from omegaconf import MISSING

from unirl.config.require import require
from unirl.rollout.engine.base import BaseEngineConfig
from unirl.rollout.engine.ports import ReservedPorts


@dataclass(frozen=True)
class VLLMOmniPorts(ReservedPorts):
    """The master-port base one ``Omni`` spawn consumes.

    Rides the ``Omni(master_port=...)`` ctor kwarg, which vllm-omni's loader
    merges into every stage's ``engine_args`` (``load_stage_configs_from_yaml``
    ``base_engine_args`` channel — YAML keys would win, but no stage YAML
    defines ``master_port``). Per-stage separation is the runtime's own job:
    each stage's ``OmniDiffusionConfig.__post_init__`` settles its port from
    this base (at the pinned v0.20.0: ``base + random(0, 100)`` then a
    +37 bind-check scan; from v0.21.0rc2 (#3803): honored verbatim, scan only
    on collision). Reserving the base de-synchronizes colocated engines
    without the v1 ``30200 + rank*200 + idx*50`` math.

    Upgrade landmine (≥ v0.21.0rc2): the env var ``MASTER_PORT`` takes
    precedence over the explicit arg — clear/guard it at the seam when the
    pin is bumped, or the injection is silently ignored.
    """

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
        """Spell this config (+ adapter boot extras + reserved ports) as boot intent.

        ``extra`` is the adapter's ``boot_kwargs()`` — the stage-YAML
        selection, driver-tokenizer need, CVD quirk, and optional
        ``Omni(mode=...)`` kwarg. The ``Omni`` ctor kwargs layer (low → high):
        engine-knowledge defaults (timeouts) < adapter ``mode`` < the
        ``omni_extra`` escape hatch — escape-hatch-highest matches v1, where
        ``omni_extra`` is the documented override for the timeout knobs.
        ``ports`` and ``enable_sleep_mode`` ride dedicated top-level keys that
        the seam applies ON TOP of the escape hatch (they become the
        ``master_port`` / ``enable_sleep_mode`` ctor kwargs vllm-omni merges
        into every stage's ``engine_args`` itself). ``model_config`` is
        accepted for signature symmetry with the other v2 engines but unused —
        vllm-omni's checkpoint path rides ``self.model_path``.
        """
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
