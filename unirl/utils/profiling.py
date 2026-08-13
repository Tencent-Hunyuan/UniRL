"""Opt-in torch.profiler harness for the worker-side training step."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Iterator, Optional

import torch

logger = logging.getLogger(__name__)


def _truthy(value: Optional[str], *, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("profiling: %s=%r is not an int; using default %d", name, raw, default)
        return default


def _rank_enabled(rank: int) -> bool:
    spec = os.environ.get("UNIRL_PROFILE_RANKS", "0").strip().lower()
    if spec in ("all", "*"):
        return True
    try:
        return rank in {int(p) for p in spec.split(",") if p.strip()}
    except ValueError:
        logger.warning("profiling: UNIRL_PROFILE_RANKS=%r unparseable; defaulting to rank 0 only", spec)
        return rank == 0


_MODE_WARNED: set = set()


def profile_mode() -> str:
    """Resolve the single switch ``UNIRL_PROFILE``. The value names the region recorded:"""
    v = os.environ.get("UNIRL_PROFILE", "").strip().lower()
    if v in ("", "0", "false", "no", "off"):
        return "off"
    if v in ("one-update", "train"):
        return v
    if v not in _MODE_WARNED:
        _MODE_WARNED.add(v)
        logger.warning("UNIRL_PROFILE=%r not recognized; use 'one-update' or 'train'. Profiling disabled.", v)
    return "off"


def profile_enabled() -> bool:
    return profile_mode() != "off"


def profile_scope() -> str:
    """The region being profiled: ``one-update`` or ``train`` (or ``off``)."""
    return profile_mode()


def _out_dir() -> str:
    """Trace output dir."""
    return os.environ.get("UNIRL_PROFILE_DIR", "").strip() or "outputs/profiler"


class TrainStepProfiler:
    """Thin wrapper: a torch profiler stepped once per rollout train call."""

    def __init__(self, prof: "torch.profiler.profile", total_steps: int, out_dir: str) -> None:
        self._prof = prof
        self._total = total_steps
        self._out_dir = out_dir
        self._n = 0
        self._stopped = False
        prof.start()

    def step(self) -> None:
        """Advance the schedule by one rollout; auto-stop + export after the window."""
        if self._stopped:
            return
        try:
            self._prof.step()
            self._n += 1
            if self._n >= self._total:
                self._prof.stop()
                self._stopped = True
                logger.info("TrainStepProfiler: %d steps profiled; trace written to %s", self._n, self._out_dir)
        except Exception:
            self._stopped = True
            try:
                self._prof.stop()
            except Exception:
                pass
            logger.warning("TrainStepProfiler: profiling/export failed; training continues", exc_info=True)

    @contextmanager
    def record(self, name: str) -> Iterator[None]:
        with torch.profiler.record_function(name):
            yield


def maybe_build_train_profiler(rank: int) -> Optional[TrainStepProfiler]:
    """Build a :class:`TrainStepProfiler` from env, or ``None`` if disabled."""
    if not profile_enabled():
        return None
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    if not _rank_enabled(int(rank)):
        return None

    wait = _int_env("UNIRL_PROFILE_WAIT", 1)
    warmup = _int_env("UNIRL_PROFILE_WARMUP", 1)
    active = _int_env("UNIRL_PROFILE_ACTIVE", 1)
    repeat = _int_env("UNIRL_PROFILE_REPEAT", 1)
    out_dir = _out_dir()
    os.makedirs(out_dir, exist_ok=True)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if _truthy(os.environ.get("UNIRL_PROFILE_CUDA"), default=True) and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    sched = torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)
    prof = torch.profiler.profile(
        activities=activities,
        schedule=sched,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(out_dir, worker_name=f"rank{int(rank)}", use_gzip=True),
        record_shapes=False,
        profile_memory=_truthy(os.environ.get("UNIRL_PROFILE_MEMORY"), default=False),
        with_stack=False,
    )
    total = max(1, (wait + warmup + active) * max(1, repeat))
    logger.info(
        "TrainStepProfiler[rank%d]: enabled (wait=%d warmup=%d active=%d repeat=%d) -> %s",
        int(rank),
        wait,
        warmup,
        active,
        repeat,
        out_dir,
    )
    try:
        return TrainStepProfiler(prof, total_steps=total, out_dir=out_dir)
    except Exception:
        logger.warning("TrainStepProfiler: profiler start failed (CUPTI init?); not profiling this run", exc_info=True)
        return None


@contextmanager
def maybe_profile_update(owner, rank: int) -> Iterator[None]:
    """One-shot profiler around a SINGLE ``_run_update`` (``UNIRL_PROFILE=one-update``)."""
    enabled = profile_enabled()
    if enabled:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        enabled = _rank_enabled(int(rank))

    n = getattr(owner, "_prof_update_seen", 0)
    owner._prof_update_seen = n + 1
    skip = _int_env("UNIRL_PROFILE_WARMUP", 2)
    if not enabled or getattr(owner, "_prof_update_done", False) or n != skip:
        yield
        return

    out_dir = _out_dir()
    os.makedirs(out_dir, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if _truthy(os.environ.get("UNIRL_PROFILE_CUDA"), default=True) and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    prof = torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        profile_memory=_truthy(os.environ.get("UNIRL_PROFILE_MEMORY"), default=False),
        with_stack=False,
    )
    try:
        prof.start()
    except Exception:
        owner._prof_update_done = True
        logger.warning(
            "maybe_profile_update[rank%d]: profiler start failed (CUPTI init?); update unprofiled",
            int(rank),
            exc_info=True,
        )
        yield
        return
    logger.info("maybe_profile_update[rank%d]: profiling one optimizer update -> %s", int(rank), out_dir)
    try:
        yield
    finally:
        # Best-effort export: mark done, then log-and-swallow failures (never kill training).
        owner._prof_update_done = True
        try:
            prof.stop()
            raw = os.path.join(out_dir, f"update_rank{int(rank)}.pt.trace.json")
            prof.export_chrome_trace(raw)
            import gzip
            import shutil

            out = raw + ".gz"
            with open(raw, "rb") as fin, gzip.open(out, "wb") as fout:
                shutil.copyfileobj(fin, fout)
            os.remove(raw)
            logger.info("maybe_profile_update[rank%d]: trace written to %s", int(rank), out)
        except Exception:
            logger.warning(
                "maybe_profile_update[rank%d]: trace export failed; training continues", int(rank), exc_info=True
            )


__all__ = [
    "TrainStepProfiler",
    "maybe_build_train_profiler",
    "maybe_profile_update",
    "profile_mode",
    "profile_enabled",
    "profile_scope",
]
