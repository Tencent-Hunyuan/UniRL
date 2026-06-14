"""``sglang_diffusion`` engine config — wired by ``_target_``; the rollout actor
constructs the engine via :meth:`SGLangDiffusionEngineConfig.make_engine`.

Ported from the legacy ``SGLangEngineConfig`` minus all port/placement math: the
engine reserves its own :class:`SGLangDiffusionPorts` at boot, so there is no
``with_sglang_ports`` / ``_SGLANG_PORT_*`` here. ``port`` / ``scheduler_port`` survive
only for remote mode (``local_mode=False``). ``model_family`` is validated against
the live adapter registry rather than a hardcoded tuple.

``server_intent`` (the successor of the legacy ``build_server_kwargs``) spells this
config + the model config + the reserved ports as the SGLang ServerArgs intent
dict; the backend filters it against the real ServerArgs fields and spawns.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, Dict, Optional, Tuple

from omegaconf import SI

from unirl.config.require import require
from unirl.rollout.engine.base import BaseEngineConfig
from unirl.rollout.engine.ports import ReservedPorts


@dataclass(frozen=True)
class SGLangDiffusionPorts(ReservedPorts):
    """The ports one local-mode ``DiffGenerator`` spawn consumes.

    - ``server_port`` — HTTP/server bind (``ServerArgs.port``). Unbound in local
      mode (``launch_http_server=False``), but injecting it keeps ServerArgs'
      ``settle_port`` from wandering and mirrors remote mode.
    - ``scheduler_port`` — the scheduler's zmq REP bind; the client connects here.
    - ``master_port`` — ``ServerArgs.master_port``: the spawned workers' dist init
      (``tcp://127.0.0.1:{master_port}``). Left unset, upstream self-settles to a
      random scanned port; injecting a reserved one keeps colocated siblings apart.
    """

    server_port: int
    scheduler_port: int
    master_port: int


@dataclass
class SGLangDiffusionEngineConfig(BaseEngineConfig):
    """Configuration for the ``sglang_diffusion`` rollout engine."""

    def make_engine(self, **deps: Any):
        from unirl.rollout.engine.sglang_diffusion.engine import SGLangDiffusionRolloutEngine

        return SGLangDiffusionRolloutEngine(config=self, **deps)

    # --- Sampling (live interpolation back to top-level cfg.sampling) ---
    sampling: Any = dc_field(default_factory=lambda: SI("${sampling}"))

    # --- Model family: selects the adapter (registry key) ---
    model_family: str = "sd3"

    # --- Conditions packing ---
    populate_conditions: bool = True

    # --- Engine-internal noise fallback (only when caller didn't pre-ship latents) ---
    init_same_noise: bool = False

    # --- Parallelism & GPU ---
    num_gpus: int = 1
    tp_size: Optional[int] = None
    sp_degree: Optional[int] = None

    # --- SGLang engine behaviour ---
    local_mode: bool = True
    disable_autocast: bool = False

    # --- Forward chunking (None = whole batch in one forward) ---
    forward_batch_size: Optional[int] = None

    # --- Weight sync ---
    target_modules: Optional[Tuple[str, ...]] = None

    # --- LoRA ---
    lora_merge_mode: Optional[str] = None

    # --- SGLang network (remote mode only; local mode self-reserves its ports) ---
    host: Optional[str] = None
    port: Optional[int] = None
    scheduler_port: Optional[int] = None

    # --- Escape hatch for rare / advanced ServerArgs overrides ---
    engine_kwargs: Optional[Dict[str, Any]] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.engine_kwargs is None:
            self.engine_kwargs = {}

        self.model_family = str(self.model_family or "").strip().lower()
        # Validate against the live adapter registry (importing it registers them).
        from unirl.rollout.engine.sglang_diffusion.adapters import registered_adapters

        valid_families = registered_adapters()
        require(
            self.model_family in valid_families,
            f"SGLangDiffusionEngineConfig.model_family must be one of {set(valid_families)}; got {self.model_family!r}",
        )

        require(self.num_gpus >= 1, f"num_gpus must be >= 1; got {self.num_gpus!r}")
        require(
            self.tp_size is None or self.tp_size >= 1,
            f"tp_size must be >= 1 when set; got {self.tp_size!r}",
        )
        require(
            self.sp_degree is None or self.sp_degree >= 1,
            f"sp_degree must be >= 1 when set; got {self.sp_degree!r}",
        )
        require(
            self.forward_batch_size is None or self.forward_batch_size >= 1,
            f"forward_batch_size must be >= 1 when set; got {self.forward_batch_size!r}",
        )
        require(
            self.local_mode or (self.host is not None and self.scheduler_port is not None),
            f"remote mode (local_mode=False) requires host and scheduler_port; "
            f"got host={self.host!r}, scheduler_port={self.scheduler_port!r}",
        )
        require(
            not self.local_mode or (self.port is None and self.scheduler_port is None),
            f"local mode self-reserves its ports; remove port / scheduler_port from the "
            f"config — they are silently ignored (the reserved set overrides them). "
            f"got port={self.port!r}, scheduler_port={self.scheduler_port!r}",
        )

    # ------------------------------------------------------------------
    # SGLang ServerArgs intent (successor of the legacy ``build_server_kwargs``)
    # ------------------------------------------------------------------

    def server_intent(
        self,
        *,
        model_config: Any,
        ports: Optional[SGLangDiffusionPorts],
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Spell this config (+ model config + reserved ports) as ServerArgs intent.

        Unfiltered: the backend filters against the real ServerArgs fields and
        spawns. Precedence (low → high): ``engine_kwargs`` escape-hatch < typed
        cfg/model fields < adapter ``extra`` < the reserved ports. In local mode
        the set supplies ``port`` / ``scheduler_port`` / ``master_port`` (all real
        ServerArgs fields — ``master_port`` is the spawned workers' dist init,
        which otherwise self-settles to a random scanned port); in remote mode
        (``ports is None``) they come from ``host`` / ``port`` / ``scheduler_port``
        on this config.
        """
        intent: Dict[str, Any] = {}

        # Layer 1: escape-hatch (lowest priority). Non-ServerArgs keys are dropped
        # by the backend's allowed-keys filter, so passing them through is harmless.
        intent.update(self.engine_kwargs or {})

        # Layer 2: typed cfg + model_config fields.
        if model_config.pretrained_model_ckpt_path:
            intent["model_path"] = model_config.pretrained_model_ckpt_path
        intent["num_gpus"] = int(self.num_gpus)
        if self.tp_size is not None:
            intent["tp_size"] = int(self.tp_size)
        if self.sp_degree is not None:
            intent["sp_degree"] = int(self.sp_degree)
        intent["disable_autocast"] = bool(self.disable_autocast)

        if self.lora_merge_mode is not None:
            intent["lora_merge_mode"] = self.lora_merge_mode
        elif model_config.use_lora:
            intent.setdefault("lora_merge_mode", "online")
        if model_config.use_lora and model_config.lora_target_modules is not None:
            intent["lora_target_modules"] = list(model_config.lora_target_modules)

        if self.host is not None:
            intent["host"] = str(self.host)

        # Layer 3: adapter model-specific extras (override hook).
        if extra:
            intent.update(extra)

        # Layer 4: ports (highest). Local mode → reserved set; remote → cfg fields.
        if ports is not None:
            intent["port"] = ports.server_port
            intent["scheduler_port"] = ports.scheduler_port
            intent["master_port"] = ports.master_port
        else:
            if self.port is not None:
                intent["port"] = int(self.port)
            if self.scheduler_port is not None:
                intent["scheduler_port"] = int(self.scheduler_port)

        return intent


__all__ = ["SGLangDiffusionEngineConfig", "SGLangDiffusionPorts"]
