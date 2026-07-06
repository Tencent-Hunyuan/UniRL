"""Driver-side GPU memory monitoring — verl-parity observability for UniRL.

Reuses the ``install_phase_timing`` pattern (wandb_logger.py): monkey-patch the
step collaborators' methods once at startup so every trainer gets memory
probes at the train/rollout hand-off boundaries with zero per-trainer edits.
Where the timing wrapper adds a stopwatch, this one brackets each phase with a
``Remote.get_memory_stats`` BROADCAST probe (remote.py) — UniRL's trainers run
on a CUDA-less Ray driver, so readings must come from the workers.

Outputs (granularity mirrors verl):

* wandb, once per step via :meth:`MemoryMonitor.step_summary` (consumed by
  ``log_rollout_step``): ``perf/max_memory_allocated_gb`` /
  ``perf/max_memory_reserved_gb`` / ``perf/cpu_memory_used_gb`` (verl's trio)
  plus ``perf/device_memory_used_gb`` — the device-level view that still sees
  a colocated SGLang server process the workers' allocators cannot.
* logs, per phase begin/end, only when ``logging.memory.log_boundaries`` is
  on: one aggregated ``[mem]`` driver line + per-rank worker lines (the verl
  ``log_gpu_memory_usage("After switch ...")`` equivalent).

Peak-counter protocol: torch keeps ONE high-water mark per process. Phase
wrappers own the reset chain — reset on phase entry, read on exit — so each
phase's ``max_allocated`` is its own peak. Phases are serial (handle dispatch
is a blocking barrier), so resets never interleave. Step-level peaks are the
driver-side max over all probe readings; the closing probe in
:meth:`step_summary` re-arms the counter for the next step. On paths with no
wrapped phases (async_ar) the closing probe alone still spans the whole step.

Behaviour-neutral by design: ``logging.memory.enabled=false`` (or
``UNIRL_MEM_MONITOR=0``) installs nothing; ``empty_cache_at`` is empty by
default so no cleanup is ever triggered by monitoring.
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from unirl.utils.memory_utils import _cpu_rss_gb, _truthy

logger = logging.getLogger(__name__)

#: Memory-probe phase specs: ``(trainer attr path, method, phase name)``.
#: Deliberately separate from wandb_logger's ``_STEP_PHASE_SPECS`` (timing) so
#: the two systems evolve independently; dotted paths reach nested handles
#: (PE's per-track stacks). Missing/uncallable attrs are skipped, so one table
#: covers all five trainers.
_MEM_PHASE_SPECS: Tuple[Tuple[str, str, str], ...] = (
    ("rollout", "wake_up", "wake_up"),
    ("rollout", "generate", "generate"),
    ("rollout", "sleep", "sleep"),
    ("weight_sync", "sync", "weight_sync"),
    ("weight_sync", "extract", "ws_extract"),  # unified_model
    ("weight_sync", "push", "ws_push"),  # unified_model
    ("backend", "offload", "offload"),  # diffusion / unified_model
    ("backend", "onload", "onload"),
    ("reward", "score_and_attach", "reward"),
    ("stack", "train_track", "train"),
    ("diffusion.stack", "train_track", "diffusion_train"),  # pe
    ("ar.stack", "train_track", "ar_train"),  # pe
)

#: worker-probe key → wandb key (fold = running max across probes and ranks)
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
        self._fallback = None  # closing-probe / snapshot-dump target handle
        self._installed = False
        self._memsnap_steps = _parse_step_range(os.environ.get("UNIRL_MEMSNAP_STEPS"))

    # ── probing ──────────────────────────────────────────────────────────

    def _probe(self, handle: Any, **kwargs: Any) -> List[Dict[str, float]]:
        """One BROADCAST probe; returns per-rank dicts ({} for CUDA-less ranks)."""
        try:
            results = handle.get_memory_stats(**kwargs)
        except Exception:  # diagnostics must never break training
            logger.warning("memory: probe failed", exc_info=True)
            return []
        if isinstance(results, dict):  # single-worker handles may not wrap in a list
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

    # ── phase wrapping (install_phase_timing pattern) ────────────────────

    def _wrap(self, handle: Any, fn: Callable, phase: str) -> Callable:
        @functools.wraps(fn)
        def _probed(*args: Any, **kwargs: Any):
            begin = self._probe(
                handle,
                reset_peak=True,
                log_stage=f"{phase}:begin" if self.log_boundaries else None,
            )
            self._fold(begin)  # pre-reset reading also covers the gap since the last probe
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
                continue  # local (non-Remote) collaborator — nothing to probe
            setattr(handle, method, self._wrap(handle, fn, phase))

    def install(self, trainer: Any) -> None:
        """Register with the live logger now; defer collaborator wrapping to step 1.

        Called from ``BaseTrainer._init_wandb``. Wrapping is deferred until after
        the first ``train_step`` so it lands OUTSIDE ``install_phase_timing``'s
        wrappers (which install lazily on step 1) — the memory probes then stay
        out of ``perf/<phase>_time_s`` (step 1 itself is unmonitored).
        """
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

    # ── per-step summary (consumed by log_rollout_step) ──────────────────

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
            for r in closing:  # driver-side log so the report reaches the training console
                report = r.get("snapshot_report")
                if report:
                    logger.info("memory: snapshot %s (rank %d)\n%s", dump_tag, int(r.get("rank", 0)), report)
        summary = dict(self._step_max)
        driver_rss = _cpu_rss_gb()
        if driver_rss is not None:
            summary["cpu_memory_used_gb"] = max(summary.get("cpu_memory_used_gb", 0.0), driver_rss)
        self._step_max.clear()
        return summary

    # ── one-off boundaries (checkpoint save, ad-hoc) ──────────────────────

    def boundary(self, stage: str, handle: Any) -> None:
        if handle is None or not callable(getattr(handle, "get_memory_stats", None)):
            return
        readings = self._probe(handle, log_stage=stage if self.log_boundaries else None)
        self._fold(readings)
        if self.log_boundaries:
            self._log_line(stage, readings)


def install_memory_monitoring(trainer: Any) -> Optional[MemoryMonitor]:
    """Build a monitor from the trainer's ``logging.memory`` block (or env override).

    Returns None when disabled — nothing is wrapped and the tree is unpatched.
    Disabled by default (opt-in): even the folding path spends two blocking
    BROADCAST probes per wrapped phase. ``UNIRL_MEM_MONITOR=0/1`` overrides
    ``logging.memory.enabled``. ``UNIRL_MEMSNAP=1`` force-enables the monitor
    (snapshots dump only through its closing probe), unless ``UNIRL_MEM_MONITOR=0``
    explicitly wins.
    """
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
