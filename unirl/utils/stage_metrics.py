"""Process-local stage timing and CUDA peaks; overlap and reset contracts live in utils/README.md."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from threading import Lock
from time import perf_counter

import torch

STAGES = ("denoise", "decode", "score", "policy_forward", "kl_ref_forward", "backward", "optimizer_step")


@dataclass(eq=False)
class Reading:
    seconds: float = 0.0
    peak_alloc_gb: float = 0.0
    peak_reserved_gb: float = 0.0


_lock = Lock()
_active: list[Reading] = []
_totals: dict[str, Reading] = {}
_saved_peaks: dict[int, tuple[float, float]] = {}


def _capture(device: int) -> tuple[float, float]:
    allocated = torch.cuda.max_memory_allocated(device) / 2**30
    reserved = torch.cuda.max_memory_reserved(device) / 2**30
    for reading in _active:
        reading.peak_alloc_gb = max(reading.peak_alloc_gb, allocated)
        reading.peak_reserved_gb = max(reading.peak_reserved_gb, reserved)
    previous = _saved_peaks.get(device, (0.0, 0.0))
    peaks = max(previous[0], allocated), max(previous[1], reserved)
    _saved_peaks[device] = peaks
    return peaks


def memory_peaks(device: int, *, reset: bool = False) -> tuple[float, float]:
    """Preserve outer monitor peaks across stage-local allocator resets; values are GiB."""
    with _lock:
        peaks = _capture(device)
        if reset:
            _saved_peaks.pop(device)
            torch.cuda.reset_peak_memory_stats(device)
        return peaks


@contextmanager
def record(stage: str):
    """Accumulate synchronized boundary time and device-wide peaks, including concurrent work."""
    cuda = torch.cuda.is_available()
    reading = Reading()
    if cuda:
        torch.cuda.synchronize()
    with _lock:
        if cuda:
            device = torch.cuda.current_device()
            _capture(device)
            torch.cuda.reset_peak_memory_stats(device)
        _active.append(reading)
    started = perf_counter()
    try:
        yield
    finally:
        if cuda:
            torch.cuda.synchronize()
        reading.seconds = perf_counter() - started
        with _lock:
            if cuda:
                _capture(device)
            _active.remove(reading)
            total = _totals.setdefault(stage, Reading())
            total.seconds += reading.seconds
            total.peak_alloc_gb = max(total.peak_alloc_gb, reading.peak_alloc_gb)
            total.peak_reserved_gb = max(total.peak_reserved_gb, reading.peak_reserved_gb)


def take() -> dict[str, dict[str, float]]:
    """Drain one worker's counters at a completed training or evaluation boundary."""
    with _lock:
        readings = {stage: asdict(_totals.get(stage, Reading())) for stage in STAGES}
        _totals.clear()
    return readings


def aggregate(readings: list[dict[str, dict[str, float]]]) -> dict[str, float]:
    """Reduce worker-local sums by mean/max and worker-local peaks by max."""
    metrics = {}
    for stage in STAGES:
        rows = [rank[stage] for rank in readings]
        metrics[f"stage/{stage}_s"] = sum(row["seconds"] for row in rows) / len(rows)
        metrics[f"stage/{stage}_s_max"] = max(row["seconds"] for row in rows)
        for name in ("peak_alloc_gb", "peak_reserved_gb"):
            metrics[f"mem/{stage}_{name}"] = max(row[name] for row in rows)
    return metrics
