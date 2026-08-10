"""``fastvideo`` engine config — wired by Hydra ``_target_``; the rollout actor
constructs the engine via :meth:`FastVideoEngineConfig.make_engine`.

Mirrors :class:`SGLangDiffusionEngineConfig` but trimmed to what the in-process
FastVideo ``VideoGenerator`` needs for a colocate diffusion rollout. Port math
is delegated to :class:`FastVideoPorts` (engine-reserved at boot, like SGLang),
so concurrent colocated engines never collide on the worker dist-init port (the
``EADDRINUSE`` failure mode of fixed ``base + rank*stride`` ports).
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
class FastVideoPorts(ReservedPorts):
    """Dist-init port the local-mode FastVideo worker subprocess consumes.

    FastVideo's ``MultiprocExecutor`` brings up a ``TCPStore`` on
    ``master_port``; left unset it self-settles to a scanned port, which races
    when several colocated engines wake concurrently. Reserving one keeps the
    siblings apart (bind-to-zero hint; see ``ReservedPorts``).
    """

    master_port: int


@dataclass
class FastVideoEngineConfig(BaseEngineConfig):
    """Configuration for the ``fastvideo`` rollout engine."""

    def make_engine(self, **deps: Any):
        from unirl.rollout.engine.fastvideo.engine import FastVideoRolloutEngine

        return FastVideoRolloutEngine(config=self, **deps)

    sampling: Any = dc_field(default_factory=lambda: SI("${sampling}"))

    model_family: str = "wan2.1"

    native_logprob: bool = True

    init_same_noise: bool = False

    num_gpus: int = 1
    tp_size: int = 1
    sp_size: int = 1

    local_mode: bool = True
    disable_autocast: bool = False

    forward_batch_size: Optional[int] = None

    target_modules: Optional[Tuple[str, ...]] = None

    fastvideo_path: Optional[str] = None

    engine_kwargs: Optional[Dict[str, Any]] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.engine_kwargs is None:
            self.engine_kwargs = {}
        self.model_family = str(self.model_family or "").strip().lower()
        require(
            self.model_family in {"wan2.1", "wan21"},
            f"FastVideoEngineConfig.model_family currently supports only 'wan2.1'; got {self.model_family!r}",
        )
        require(self.num_gpus >= 1, f"num_gpus must be >= 1; got {self.num_gpus!r}")
        require(
            self.forward_batch_size is None or self.forward_batch_size >= 1,
            f"forward_batch_size must be >= 1 when set; got {self.forward_batch_size!r}",
        )
        require(
            self.local_mode,
            "FastVideoEngineConfig currently supports local_mode=True only (in-process colocate).",
        )


__all__ = ["FastVideoEngineConfig", "FastVideoPorts"]
