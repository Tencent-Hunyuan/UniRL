"""Opt-in torch.profiler harness for the worker-side training step.

UniRL is an RL framework (rollout → reward → advantage → optimizer step). To
profile *only* the training compute (forward / loss / backward / optimizer) of a
backend — without the SGLang/vLLM rollout — wrap the per-rollout train call on
the worker with :class:`TrainStepProfiler`. The rollout runs in a separate engine
phase, so the profiled region here is the pure train step.

Entirely env-gated; a no-op unless ``UNIRL_PROFILE`` is set, so it can ship in the
hot path. ONE switch, whose value names the region recorded:

* ``UNIRL_PROFILE=one-update`` — profile ONE optimizer update (forward + backward +
  cross-GPU comm + optimizer), excluding the big anchor/SDE-replay forward. Small trace; for
  compute/comm OVERLAP. Exports ``update_rank0.pt.trace.json.gz`` (opens directly in Perfetto).
* ``UNIRL_PROFILE=train`` — profile the WHOLE train step (anchor forward + all N
  updates). Big trace; the complete picture of one training step.

Optional knobs (sensible defaults; override any one):

* ``UNIRL_PROFILE_DIR``     — trace output dir (default ``outputs/profiler``, auto-created).
* ``UNIRL_PROFILE_RANKS``   — ``0`` (default), ``all``, or a comma list ``0,8``.
* ``UNIRL_PROFILE_CUDA``    — record CUDA kernels (default ``1``; ``0`` = CPU-only trace).
* ``UNIRL_PROFILE_WARMUP``  — one-update: skip N updates before capturing (default ``2``);
                              train: schedule warmup steps (default ``1``).
* ``UNIRL_PROFILE_WAIT`` / ``_ACTIVE`` / ``_REPEAT`` — train schedule (default ``1``/``1``/``1``
                              = capture exactly ONE full step).
* ``UNIRL_PROFILE_MEMORY``  — also record CUDA memory alloc/free (default off; bigger trace).

Both modes export a gzipped Chrome/Perfetto trace, one file per profiled rank.
"""

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
    """Resolve the single switch ``UNIRL_PROFILE``. The value names the region recorded:

    * ``one-update`` — profile ONE optimizer update (forward + backward + cross-GPU comm +
      optimizer), excluding the big anchor/SDE-replay forward. Small trace; for OVERLAP.
    * ``train``  — profile the WHOLE train step (anchor forward + all N updates).
      Big trace; the complete picture of one training step.
    * unset / ``0`` / ``false`` / ``off`` — disabled (a no-op).

    Any other value is unrecognized -> disabled (warned once). Individual ``UNIRL_PROFILE_*``
    knobs (DIR, RANKS, CUDA, WARMUP, ...) still tune the run.
    """
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
    """The region being profiled: ``one-update`` or ``train`` (or ``off``).

    Identical to the ``UNIRL_PROFILE`` value — the switch *is* the scope, no separate knob.
    """
    return profile_mode()


def _out_dir() -> str:
    """Trace output dir. Defaults to ``outputs/profiler`` (relative to cwd, created if
    missing); set ``UNIRL_PROFILE_DIR`` to write elsewhere. NOTE: writing a very large
    (multi-GB) trace to a network FS can stall the export — point ``UNIRL_PROFILE_DIR``
    at a node-local path for whole-step captures."""
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
        """Advance the schedule by one rollout; auto-stop + export after the window.
        Export is best-effort — a failure is logged, never raised into training."""
        if self._stopped:
            return
        try:
            self._prof.step()  # may trigger on_trace_ready (export) at the active->done edge
            self._n += 1
            if self._n >= self._total:
                self._prof.stop()
                self._stopped = True
                logger.info("TrainStepProfiler: %d steps profiled; trace written to %s", self._n, self._out_dir)
        except Exception:
            self._stopped = True
            try:
                self._prof.stop()  # release CUPTI hooks so later steps carry no overhead
            except Exception:
                pass
            logger.warning("TrainStepProfiler: profiling/export failed; training continues", exc_info=True)

    @contextmanager
    def record(self, name: str) -> Iterator[None]:
        with torch.profiler.record_function(name):
            yield


def maybe_build_train_profiler(rank: int) -> Optional[TrainStepProfiler]:
    """Build a :class:`TrainStepProfiler` from env, or ``None`` if disabled.

    Called lazily on the worker the first time a train step runs (so the device
    is bound and the profiler attaches to the right CUDA context).
    """
    if not profile_enabled():
        return None
    # The caller passes a backend-specific rank that is 0 on every worker for some
    # backends (e.g. FSDP colocate lacks `_rank`). Prefer the true global rank from
    # the process group so UNIRL_PROFILE_RANKS actually restricts to one worker —
    # profiling every rank makes 8 CUPTI trace-flushes contend and stall the export.
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    if not _rank_enabled(int(rank)):
        return None

    # Default = capture ONE full step (wait1 + warmup1 + active1) so a bare
    # UNIRL_PROFILE=train already produces a small, Perfetto-loadable trace.
    wait = _int_env("UNIRL_PROFILE_WAIT", 1)
    warmup = _int_env("UNIRL_PROFILE_WARMUP", 1)
    active = _int_env("UNIRL_PROFILE_ACTIVE", 1)
    repeat = _int_env("UNIRL_PROFILE_REPEAT", 1)
    out_dir = _out_dir()
    os.makedirs(out_dir, exist_ok=True)

    activities = [torch.profiler.ProfilerActivity.CPU]
    # CUDA (CUPTI) activity can be disabled with UNIRL_PROFILE_CUDA=0. On some
    # torch/driver/CUPTI combos the CUDA kineto trace-finalize (stop_trace) hangs
    # the export; a CPU-only trace still opens in Perfetto and shows the step
    # structure + cudaLaunchKernel timeline.
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
        return TrainStepProfiler(prof, total_steps=total, out_dir=out_dir)  # __init__ calls prof.start()
    except Exception:
        logger.warning("TrainStepProfiler: profiler start failed (CUPTI init?); not profiling this run", exc_info=True)
        return None


@contextmanager
def maybe_profile_update(owner, rank: int) -> Iterator[None]:
    """One-shot profiler around a SINGLE ``_run_update`` (``UNIRL_PROFILE=one-update``).

    torch.profiler records continuously while active, so the schedule-based
    :class:`TrainStepProfiler` (which spans a whole rollout) always sweeps in the big
    SDE-replay ``prepare_segment`` too. For compute/comm OVERLAP analysis we want just
    one optimizer update — forward + backward + cross-GPU comm + optimizer_step.
    This wraps exactly that region in its own profiler and exports immediately, so the
    trace is small and contains only the overlap-relevant window.

    Fires once, on rank0 (true global rank), after skipping ``UNIRL_PROFILE_WARMUP``
    updates (default 2) so the profiled step is past first-iter compile/allocation.
    A no-op context otherwise.
    """
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
            # gzip in place -> opens directly in Perfetto and is small enough to download
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
