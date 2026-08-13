"""Driver-side GPU memory monitoring — verl-parity observability for UniRL."""

from __future__ import annotations

import functools
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from unirl.utils.memory_utils import _cpu_rss_gb, _truthy

logger = logging.getLogger(__name__)

_MEM_PHASE_SPECS: Tuple[Tuple[str, str, str], ...] = (
    ("rollout", "wake_up", "wake_up"),
    ("rollout", "generate", "generate"),
    ("rollout", "sleep", "sleep"),
    ("weight_sync", "sync", "weight_sync"),
    ("weight_sync", "extract", "ws_extract"),
    ("weight_sync", "push", "ws_push"),
    ("backend", "offload", "offload"),
    ("backend", "onload", "onload"),
    ("reward", "score_and_attach", "reward"),
    ("stack", "train_track", "train"),
    ("diffusion.stack", "train_track", "diffusion_train"),
    ("ar.stack", "train_track", "ar_train"),
)

_FOLD_KEYS = {
    "max_allocated_gb": "max_memory_allocated_gb",
    "max_reserved_gb": "max_memory_reserved_gb",
    "device_used_gb": "device_memory_used_gb",
    "cpu_rss_gb": "cpu_memory_used_gb",
}


def _resolve_attr(root: Any, path: str) -> Any:
    obj = root
    for part in path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _parse_step_range(spec: Optional[str]) -> Optional[Tuple[int, int]]:
    """``"2:4"`` → (2, 4) inclusive; ``"3"`` → (3, 3); None/invalid → None."""
    if not spec or not spec.strip():
        return None
    try:
        if ":" in spec:
            lo, hi = spec.split(":", 1)
            return int(lo), int(hi)
        step = int(spec)
        return step, step
    except ValueError:
        logger.warning("memory: UNIRL_MEMSNAP_STEPS=%r unparseable; snapshot dumps disabled", spec)
        return None


class MemoryMonitor:
    """Orchestrates worker memory probes; aggregates per step for wandb."""

    def __init__(
        self,
        *,
        log_boundaries: bool = False,
        empty_cache_at: Sequence[str] = (),
    ) -> None:
        self.log_boundaries = bool(log_boundaries)
        self.empty_cache_at = tuple(empty_cache_at or ())
        self._step_max: Dict[str, float] = {}
        self._fallback = None
        self._installed = False
        self._memsnap_steps = _parse_step_range(os.environ.get("UNIRL_MEMSNAP_STEPS"))

    def _probe(self, handle: Any, **kwargs: Any) -> List[Dict[str, float]]:
        """One BROADCAST probe; returns per-rank dicts ({} for CUDA-less ranks)."""
        try:
            results = handle.get_memory_stats(**kwargs)
        except Exception:  # diagnostics must never break training
            logger.warning("memory: probe failed", exc_info=True)
            return []
        if isinstance(results, dict):
            results = [results]
        return [r for r in (results or []) if r]

    def _fold(self, readings: List[Dict[str, float]]) -> None:
        for r in readings:
            for src, dst in _FOLD_KEYS.items():
                if src in r:
                    self._step_max[dst] = max(self._step_max.get(dst, 0.0), float(r[src]))

    def _log_line(self, stage: str, readings: List[Dict[str, float]]) -> None:
        if not readings:
            return
        alloc = [(r.get("max_allocated_gb", 0.0), int(r.get("rank", -1))) for r in readings]
        hi, hi_rank = max(alloc)
        lo = min(a for a, _ in alloc)
        any_r = max(readings, key=lambda r: r.get("max_allocated_gb", 0.0))
        logger.info(
            "[mem] stage=%s peak_alloc=%.2f (rank%d, min %.2f) reserved=%.2f device_used=%.1f (GB)",
            stage,
            hi,
            hi_rank,
            lo,
            any_r.get("reserved_gb", 0.0),
            any_r.get("device_used_gb", 0.0),
        )

    def _wrap(self, handle: Any, fn: Callable, phase: str) -> Callable:
        @functools.wraps(fn)
        def _probed(*args: Any, **kwargs: Any):
            begin = self._probe(
                handle,
                reset_peak=True,
                log_stage=f"{phase}:begin" if self.log_boundaries else None,
            )
            self._fold(begin)
            try:
                return fn(*args, **kwargs)
            finally:
                end = self._probe(
                    handle,
                    log_stage=f"{phase}:end" if self.log_boundaries else None,
                    empty_cache=phase in self.empty_cache_at,
                )
                self._fold(end)
                if self.log_boundaries:
                    self._log_line(phase, end)

        return _probed

    def _wrap_collaborators(self, trainer: Any) -> None:
        for attr_path, method, phase in _MEM_PHASE_SPECS:
            handle = _resolve_attr(trainer, attr_path)
            if handle is None:
                continue
            fn = getattr(handle, method, None)
            if not callable(fn):
                continue
            if not callable(getattr(handle, "get_memory_stats", None)):
                continue
            setattr(handle, method, self._wrap(handle, fn, phase))

    def install(self, trainer: Any) -> None:
        """Register with the live logger now; defer collaborator wrapping to step 1."""
        if self._installed:
            return
        for attr in ("stack", "backend"):
            handle = getattr(trainer, attr, None)
            if handle is not None and callable(getattr(handle, "get_memory_stats", None)):
                self._fallback = handle
                break
        trainer.wandb_logger.memory_monitor = self
        self._installed = True

        inner = getattr(trainer, "train_step", None)
        if not callable(inner):
            self._wrap_collaborators(trainer)
            return

        @functools.wraps(inner)
        def _wrap_after_first_step(*args: Any, **kwargs: Any):
            try:
                return inner(*args, **kwargs)
            finally:
                self._wrap_collaborators(trainer)
                if trainer.train_step is _wrap_after_first_step:
                    trainer.train_step = inner

        trainer.train_step = _wrap_after_first_step

    def step_summary(self, step: Optional[int] = None) -> Dict[str, float]:
        """Fold the step's probes into verl-parity wandb keys; re-arm for the next step."""
        if self._fallback is not None:
            dump_tag = None
            if (
                step is not None
                and self._memsnap_steps is not None
                and self._memsnap_steps[0] <= step <= self._memsnap_steps[1]
            ):
                dump_tag = f"step{step}"
            closing = self._probe(self._fallback, reset_peak=True, dump_snapshot_tag=dump_tag)
            self._fold(closing)
            for r in closing:
                report = r.get("snapshot_report")
                if report:
                    logger.info("memory: snapshot %s (rank %d)\n%s", dump_tag, int(r.get("rank", 0)), report)
        summary = dict(self._step_max)
        driver_rss = _cpu_rss_gb()
        if driver_rss is not None:
            summary["cpu_memory_used_gb"] = max(summary.get("cpu_memory_used_gb", 0.0), driver_rss)
        self._step_max.clear()
        return summary

    def boundary(self, stage: str, handle: Any) -> None:
        if handle is None or not callable(getattr(handle, "get_memory_stats", None)):
            return
        readings = self._probe(handle, log_stage=stage if self.log_boundaries else None)
        self._fold(readings)
        if self.log_boundaries:
            self._log_line(stage, readings)


def install_memory_monitoring(trainer: Any) -> Optional[MemoryMonitor]:
    """Build a monitor from the trainer's ``logging.memory`` block (or env override)."""
    logging_cfg = getattr(trainer, "logging_cfg", None) or {}
    mem_cfg = logging_cfg.get("memory") if hasattr(logging_cfg, "get") else None
    mem_cfg = mem_cfg or {}
    enabled = bool(mem_cfg.get("enabled", False))
    env_override = os.environ.get("UNIRL_MEM_MONITOR")
    if env_override is not None:
        enabled = _truthy(env_override, default=enabled)
    if _truthy(os.environ.get("UNIRL_MEMSNAP")):
        if env_override is not None and not enabled:
            logger.warning(
                "memory: UNIRL_MEMSNAP=1 but UNIRL_MEM_MONITOR is off — snapshots will "
                "record (overhead) but never dump; set UNIRL_MEM_MONITOR=1 to dump them."
            )
        else:
            enabled = True
    if not enabled:
        return None
    return MemoryMonitor(
        log_boundaries=bool(mem_cfg.get("log_boundaries", False)),
        empty_cache_at=tuple(mem_cfg.get("empty_cache_at", ()) or ()),
    )
