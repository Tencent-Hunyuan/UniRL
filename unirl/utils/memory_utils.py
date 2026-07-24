"""GPU memory observation utilities (verl-parity toolkit).

Four tools, semantics aligned with verl's ``verl/utils/memory_utils.py`` so
their monitoring playbook (wandb ``perf/max_memory_*`` curves, stage-tagged
``[mem]`` log lines, allocator snapshots) applies to UniRL unchanged:

* :func:`get_memory_info`      — one dict of allocator + device-level readings.
* :func:`log_memory_usage`     — the readings as a single stage-tagged log line.
* :func:`aggressive_empty_cache` — looped ``gc.collect + empty_cache`` until a
  round frees less than ``min_freed_gb``.
* :class:`MemorySnapshotSampler` — ``torch.cuda.memory._record_memory_history``
  recorder + ``_dump_snapshot`` dumper (open dumps with
  https://pytorch.org/memory_viz).

Everything here reads the CURRENT process only. UniRL trainers run on a
CUDA-less Ray driver, so these functions are useful on workers — the driver
reaches them through ``Remote.get_memory_stats`` (a BROADCAST RPC probe).
Orchestration (when to probe, per-step aggregation, wandb keys) lives in
``unirl/utils/memory_monitor.py``.

``misc.clear_memory()`` remains the one-shot cleanup helper; use
:func:`aggressive_empty_cache` when you want the verl-style "loop until dry"
behaviour with freed-bytes accounting.
"""

from __future__ import annotations

import gc
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)

_GB = float(2**30)  # matches train/stack/base.py's cuda_alloc_gb convention


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
        logger.warning("memory: %s=%r is not an int; using default %d", name, raw, default)
        return default


def _rank_enabled(rank: int, spec_env: str = "UNIRL_MEMSNAP_RANKS") -> bool:
    spec = os.environ.get(spec_env, "0").strip().lower()
    if spec in ("all", "*"):
        return True
    try:
        return rank in {int(p) for p in spec.split(",") if p.strip()}
    except ValueError:
        logger.warning("memory: %s=%r unparseable; defaulting to rank 0 only", spec_env, spec)
        return rank == 0


def _cpu_rss_gb() -> Optional[float]:
    try:
        import psutil

        return psutil.Process().memory_info().rss / _GB
    except Exception:  # psutil missing or /proc unreadable — omit rather than fail
        return None


def get_memory_info(device: Optional[int] = None) -> Dict[str, float]:
    """Current-process memory readings in GB. Empty dict without CUDA.

    Allocator-level (this process only): ``allocated_gb`` / ``reserved_gb`` /
    ``cached_gb`` (= reserved - allocated) / ``max_allocated_gb`` /
    ``max_reserved_gb`` (peaks since the last ``reset_peak_memory_stats``).

    Device-level (every process on the GPU, e.g. a colocated SGLang server this
    process's allocator cannot see): ``total_gb`` / ``device_used_gb`` from
    ``torch.cuda.mem_get_info``.

    Plus ``cpu_rss_gb`` for this process when psutil is available.
    """
    if not torch.cuda.is_available():
        return {}
    dev = torch.cuda.current_device() if device is None else device
    allocated = torch.cuda.memory_allocated(dev) / _GB
    reserved = torch.cuda.memory_reserved(dev) / _GB
    info: Dict[str, float] = {
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "cached_gb": reserved - allocated,
        "max_allocated_gb": torch.cuda.max_memory_allocated(dev) / _GB,
        "max_reserved_gb": torch.cuda.max_memory_reserved(dev) / _GB,
    }
    try:
        free_b, total_b = torch.cuda.mem_get_info(dev)
        info["total_gb"] = total_b / _GB
        info["device_used_gb"] = (total_b - free_b) / _GB
    except Exception:  # some backends lack mem_get_info — allocator stats still stand
        pass
    rss = _cpu_rss_gb()
    if rss is not None:
        info["cpu_rss_gb"] = rss
    return info


def log_memory_usage(
    stage: str,
    logger_: Optional[logging.Logger] = None,
    level: int = logging.INFO,
) -> Dict[str, float]:
    """Log :func:`get_memory_info` as one ``[mem] stage=...`` line; return the info."""
    info = get_memory_info()
    log = logger_ or logger
    if not info:
        log.log(level, "[mem] stage=%s (no CUDA in this process)", stage)
        return info
    log.log(
        level,
        "[mem] stage=%s alloc=%.2f reserved=%.2f peak_alloc=%.2f peak_reserved=%.2f "
        "device_used=%.1f/%.1f cpu_rss=%.1f (GB)",
        stage,
        info.get("allocated_gb", 0.0),
        info.get("reserved_gb", 0.0),
        info.get("max_allocated_gb", 0.0),
        info.get("max_reserved_gb", 0.0),
        info.get("device_used_gb", 0.0),
        info.get("total_gb", 0.0),
        info.get("cpu_rss_gb", 0.0),
    )
    return info


def aggressive_empty_cache(
    force_sync: bool = False,
    max_rounds: int = 10,
    min_freed_gb: float = 1.0,
) -> Dict[str, float]:
    """Loop ``gc.collect + torch.cuda.empty_cache`` until a round frees < ``min_freed_gb``.

    verl semantics: the first round returns the allocator's big cached blocks,
    later rounds catch tensors that only die once reference cycles are
    collected. Stops early when a round stops paying. Never called by the
    monitoring path by default — behaviour-neutral unless explicitly invoked
    (e.g. via ``logging.memory.empty_cache_at``).

    Returns ``{"freed_reserved_gb", "freed_allocated_gb", "rounds"}``.
    """
    if not torch.cuda.is_available():
        return {"freed_reserved_gb": 0.0, "freed_allocated_gb": 0.0, "rounds": 0.0}
    start_reserved = torch.cuda.memory_reserved() / _GB
    start_allocated = torch.cuda.memory_allocated() / _GB
    rounds = 0
    for attempt in range(max_rounds):
        before_reserved = torch.cuda.memory_reserved() / _GB
        gc.collect()
        torch.cuda.empty_cache()
        if force_sync:
            torch.cuda.synchronize()
        rounds = attempt + 1
        freed = before_reserved - torch.cuda.memory_reserved() / _GB
        logger.debug("memory: cleanup round %d freed %.2f GB reserved", rounds, freed)
        if freed < min_freed_gb:
            break
    return {
        "freed_reserved_gb": start_reserved - torch.cuda.memory_reserved() / _GB,
        "freed_allocated_gb": start_allocated - torch.cuda.memory_allocated() / _GB,
        "rounds": float(rounds),
    }


def _top_frame(frames: Optional[list]) -> str:
    """The innermost non-torch-internal call frame as ``file:line:func``.

    Attributes an allocation to the user code that requested it, skipping
    torch's own allocator frames (which are the same for every allocation).
    """
    if not frames:
        return "<no stack captured>"
    for fr in frames:
        fn = (fr.get("filename") or "").replace("\\", "/")
        if fn and "/torch/" not in fn:
            return f"{fn}:{fr.get('line', '?')}:{fr.get('name', '?')}"
    fr = frames[0]
    return f"{fr.get('filename', '?')}:{fr.get('line', '?')}:{fr.get('name', '?')}"


def summarize_snapshot(snapshot, top: int = 15) -> str:
    """Rank the call sites holding live GPU memory — a text report, no GUI.

    ``snapshot`` is either a loaded torch snapshot dict
    (``torch.cuda.memory._snapshot()``) or a path to a ``.pickle`` dump. The
    pickle is plain dicts/lists, so this reads it **without torch or CUDA** —
    an agent can analyse a dump on any machine.

    Groups every still-live block (``state == "active_allocated"``) by its
    allocating ``file:line`` and sorts by total bytes. For a leak, diff two
    dumps (e.g. ``memsnap_step2`` vs ``memsnap_step8``): the site whose GB grew
    is the leak. A single dump already shows the biggest holders.
    """
    if isinstance(snapshot, (str, Path)):
        import pickle

        with open(snapshot, "rb") as f:
            snapshot = pickle.load(f)

    from collections import defaultdict

    by_site: Dict[str, list] = defaultdict(lambda: [0, 0])  # site -> [bytes, count]
    total_live = 0
    total_blocks = 0
    for seg in snapshot.get("segments", []):
        for blk in seg.get("blocks", []):
            if blk.get("state") != "active_allocated":
                continue
            size = blk.get("requested_size") or blk.get("size", 0)
            site = _top_frame(blk.get("frames"))
            by_site[site][0] += size
            by_site[site][1] += 1
            total_live += size
            total_blocks += 1

    ranked = sorted(by_site.items(), key=lambda kv: kv[1][0], reverse=True)[:top]
    lines = [
        f"live GPU allocations: {total_live / _GB:.2f} GB across {total_blocks} blocks",
        f"top {len(ranked)} call sites by live bytes:",
    ]
    for site, (nbytes, count) in ranked:
        lines.append(f"  {nbytes / _GB:6.2f} GB  x{count:<5d}  {site}")
    if not ranked:
        lines.append("  (no live allocations with captured stacks)")
    return "\n".join(lines)


class MemorySnapshotSampler:
    """Record per-allocation history and dump it as memory_viz-openable pickles.

    Must live in the process whose allocations you want to see (a Ray worker,
    not the driver). Recording hooks every allocation — measurable overhead and
    dumps of up to hundreds of MB — so it is env-gated off by default and
    normally limited to rank 0.
    """

    def __init__(self, out_dir: str, max_entries: int = 100_000, rank: int = 0) -> None:
        self.out_dir = Path(out_dir)
        self.max_entries = max_entries
        self.rank = rank
        self._recording = False

    def start(self) -> None:
        if self._recording or not torch.cuda.is_available():
            return
        try:
            # Capture PYTHON allocation stacks so summarize_snapshot can attribute
            # live memory to a file:line. stacks="all" gives C++ unwind frames
            # (useless here — everything collapses to torch::unwind); "python" is
            # what yields real .py:line sites. Fall back to the minimal call on
            # older torch that lacks the context/stacks keywords.
            try:
                torch.cuda.memory._record_memory_history(max_entries=self.max_entries, context="all", stacks="python")
            except TypeError:
                torch.cuda.memory._record_memory_history(max_entries=self.max_entries)
            self._recording = True
            logger.info("memory: snapshot recording started (max_entries=%d)", self.max_entries)
        except Exception:
            logger.warning("memory: failed to start snapshot recording", exc_info=True)

    def dump(self, tag: str) -> Optional[str]:
        """Dump history to ``<out_dir>/memsnap_<tag>_rank<r>.pickle``; return the
        ranked :func:`summarize_snapshot` report string (None on failure).

        Runs on the worker; the report is RETURNED (not logged here) so the driver
        can surface it inline — a worker-side ``logging`` call would only reach the
        Ray worker log files, not the training console.
        """
        if not self._recording:
            return None
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            path = self.out_dir / f"memsnap_{tag}_rank{self.rank}.pickle"
            torch.cuda.memory._dump_snapshot(str(path))
            logger.info("memory: snapshot dumped to %s", path)
            try:
                return summarize_snapshot(torch.cuda.memory._snapshot())
            except Exception:  # analysis is a bonus; a failure must not lose the pickle
                logger.warning("memory: snapshot analysis failed for tag=%s", tag, exc_info=True)
                return None
        except Exception:  # never let diagnostics kill training
            logger.warning("memory: snapshot dump failed for tag=%s", tag, exc_info=True)
            return None

    def stop(self) -> None:
        if not self._recording:
            return
        try:
            torch.cuda.memory._record_memory_history(enabled=None)
        except Exception:
            pass
        self._recording = False

    @classmethod
    def maybe_from_env(cls, rank: int = 0) -> Optional["MemorySnapshotSampler"]:
        """Build + start a sampler when ``UNIRL_MEMSNAP`` is truthy for this rank.

        Env knobs (same family as ``UNIRL_PROFILE_*``): ``UNIRL_MEMSNAP``,
        ``UNIRL_MEMSNAP_DIR`` (default ``outputs/memsnap``),
        ``UNIRL_MEMSNAP_MAX_ENTRIES``, ``UNIRL_MEMSNAP_RANKS`` (default ``0``).
        """
        if not _truthy(os.environ.get("UNIRL_MEMSNAP")):
            return None
        if not _rank_enabled(rank):
            return None
        sampler = cls(
            out_dir=os.environ.get("UNIRL_MEMSNAP_DIR", "outputs/memsnap"),
            max_entries=_int_env("UNIRL_MEMSNAP_MAX_ENTRIES", 100_000),
            rank=rank,
        )
        sampler.start()
        return sampler


# ── Process-level sampler registry ─────────────────────────────────────────
#
# The sampler must record inside the worker process, but the dump trigger
# arrives via ``Remote.get_memory_stats`` (defined on a base class with no
# Worker back-reference). A module-level slot is the simplest bridge: Worker
# installs at startup, the probe fetches at dump time.

_PROCESS_SAMPLER: Optional[MemorySnapshotSampler] = None


def init_process_snapshot_sampler(rank: int = 0) -> Optional[MemorySnapshotSampler]:
    """Install this process's sampler from env (idempotent). Called by Worker.__init__."""
    global _PROCESS_SAMPLER
    if _PROCESS_SAMPLER is None:
        _PROCESS_SAMPLER = MemorySnapshotSampler.maybe_from_env(rank=rank)
    return _PROCESS_SAMPLER


def get_process_snapshot_sampler() -> Optional[MemorySnapshotSampler]:
    return _PROCESS_SAMPLER
