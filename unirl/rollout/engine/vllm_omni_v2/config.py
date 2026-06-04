"""``vllm_omni_v2`` engine config + the typed port set it self-reserves.

Ported from v1's ``VLLMOmniEngineConfig`` minus all port/placement math: the
engine reserves its own :class:`VLLMOmniPorts` at boot (one master port per
stage, riding each stage's own ``engine_args.master_port``), so there is no
``_VLLM_OMNI_PORT_BASE + rank * stride`` and no ``RANK``-env fallback here.
``modality`` is validated against the live adapter registry rather than the
engine raising on an unknown YAML key at boot.

``server_intent`` (the successor of v1's inline YAML-injection + ``Omni``
kwargs assembly) spells this config + the reserved ports + the adapter's boot
extras as the intent dict ``VLLMOmniBackend.boot`` consumes.
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
    """The per-stage master ports one ``Omni`` spawn consumes.

    Each rides the stage's own typed ``engine_args.master_port`` field —
    vllm-omni's ``StageRuntimeData.__post_init__`` otherwise self-settles to
    ``30005 + random(0, 100)``, which collides across colocated actors. Two
    ports cover the maximum stage count (t2i/it2i = AR + DiT); single-stage
    modalities simply leave ``stage1_master_port`` unused (still reserved —
    reservation is cheap and keeps the set one fixed shape).
    """

    stage0_master_port: int
    stage1_master_port: int


@dataclass
class VLLMOmniV2EngineConfig(BaseEngineConfig):
    def make_engine(self, **deps: Any):
        from unirl.rollout.engine.vllm_omni_v2.engine import VLLMOmniV2RolloutEngine

        return VLLMOmniV2RolloutEngine(config=self, **deps)

    # Required: model checkpoint path. Set per experiment or via
    # ``cfg.rollout.engine.model_path=...`` on the CLI.
    model_path: str = MISSING
    # Adapter registry key — one of ``registered_adapters()`` (the 8 v1
    # modalities). Kept as ``str`` because OmegaConf structured configs reject
    # ``Literal[...]``; ``__post_init__`` validates against the live registry.
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

    # Overlay ``enable_sleep_mode: True`` onto each stage's ``engine_args`` at
    # boot so worker.sleep()/wake_up() (level 2) can run. Disable to fall back
    # to the upstream YAML defaults (CuMemAllocator pool off, sleep raises).
    # Required for ``cfg.training.execution.offload_rollout = True``.
    enable_sleep_mode: bool = True

    # Passthrough for advanced ``Omni`` kwargs not surfaced as typed fields.
    omni_extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.modality = str(self.modality or "").strip().lower()
        # Validate against the live adapter registry (importing it registers them).
        from unirl.rollout.engine.vllm_omni_v2.adapters import registered_adapters

        valid = registered_adapters()
        require(
            self.modality in valid,
            f"VLLMOmniV2EngineConfig.modality must be one of {set(valid)}; got {self.modality!r}",
        )

    # ------------------------------------------------------------------
    # Boot intent (consumed by ``VLLMOmniBackend.boot``)
    # ------------------------------------------------------------------

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
        ``ports`` ride a dedicated top-level key (per-stage ``engine_args``
        overlays, untouchable by the escape hatch). ``model_config`` is
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
        # Adapter boot extras: stage_yaml / stage_yaml_source /
        # needs_driver_tokenizer / clear_cuda_visible.
        intent.update(extra)

        omni_kwargs: Dict[str, Any] = dict(
            # HI3 weights are ~150GB; loading from cephfs over the network
            # easily blows past the 300s default. Allow up to 20 min per
            # stage, 30 min for the orchestrator. Override via omni_extra.
            stage_init_timeout=1200,
            init_timeout=1800,
        )
        if mode is not None:
            omni_kwargs["mode"] = mode
        omni_kwargs.update(self.omni_extra or {})
        intent["omni_kwargs"] = omni_kwargs
        return intent


__all__ = ["VLLMOmniPorts", "VLLMOmniV2EngineConfig"]
