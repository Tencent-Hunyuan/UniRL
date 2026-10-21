"""Versioned, stdlib-only benchmark result schema and JSON writer."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCHEMA_VERSION = 1


def _validate_number(value: int | float, name: str, *, integer: bool = False, strictly_positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be an int or float")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if integer and not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0 or (strictly_positive and value == 0):
        requirement = "strictly positive" if strictly_positive else "non-negative"
        raise ValueError(f"{name} must be {requirement}")


def _validate_string(value: object, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")


@dataclass(frozen=True)
class Environment:
    """Describe the runtime; ``device`` is ``cpu`` or the accelerator model (e.g. ``NVIDIA H20``)."""

    python: str
    torch: str | None
    device: str
    world_size: int

    def __post_init__(self) -> None:
        _validate_string(self.python, "python")
        if self.torch is not None:
            _validate_string(self.torch, "torch")
        _validate_string(self.device, "device")
        _validate_number(self.world_size, "world_size", integer=True, strictly_positive=True)


@dataclass(frozen=True)
class Workload:
    """Describe total measured samples and warmup/measured iteration counts."""

    samples: int
    warmup_iterations: int
    measured_iterations: int

    def __post_init__(self) -> None:
        _validate_number(self.samples, "samples", integer=True)
        _validate_number(self.warmup_iterations, "warmup_iterations", integer=True)
        _validate_number(self.measured_iterations, "measured_iterations", integer=True, strictly_positive=True)


@dataclass(frozen=True)
class Metrics:
    """Store measured duration and optional resource or phase metrics."""

    wall_clock_s: float
    peak_memory_bytes: int | None = None
    phases_s: dict[str, float] | None = None

    def __post_init__(self) -> None:
        _validate_number(self.wall_clock_s, "wall_clock_s", strictly_positive=True)
        if self.peak_memory_bytes is not None:
            _validate_number(self.peak_memory_bytes, "peak_memory_bytes", integer=True)
        if self.phases_s is not None and not isinstance(self.phases_s, dict):
            raise TypeError("phases_s must be a dict or None")
        for name, value in (self.phases_s or {}).items():
            if not isinstance(name, str):
                raise TypeError(f"phases_s keys must be str, got {name!r}")
            _validate_number(value, f"phases_s[{name!r}]")


@dataclass(frozen=True)
class BenchmarkResult:
    """Represent one versioned benchmark result."""

    benchmark: str
    run_id: str
    started_at: datetime
    environment: Environment
    workload: Workload
    metrics: Metrics

    def __post_init__(self) -> None:
        _validate_string(self.benchmark, "benchmark")
        _validate_string(self.run_id, "run_id")
        if not isinstance(self.started_at, datetime) or self.started_at.utcoffset() is None:
            raise ValueError("started_at must be a timezone-aware datetime")
        if not isinstance(self.environment, Environment):
            raise TypeError("environment must be an Environment")
        if not isinstance(self.workload, Workload):
            raise TypeError("workload must be a Workload")
        if not isinstance(self.metrics, Metrics):
            raise TypeError("metrics must be Metrics")
        try:
            samples_per_second = self.workload.samples / self.metrics.wall_clock_s
        except OverflowError as exc:
            raise ValueError("samples_per_second must be finite") from exc
        _validate_number(samples_per_second, "samples_per_second")

    def to_dict(self) -> dict[str, Any]:
        metrics: dict[str, Any] = {
            "wall_clock_s": self.metrics.wall_clock_s,
            "samples_per_second": self.workload.samples / self.metrics.wall_clock_s,
        }
        if self.metrics.peak_memory_bytes is not None:
            metrics["peak_memory_bytes"] = self.metrics.peak_memory_bytes
        if self.metrics.phases_s is not None:
            metrics["phases_s"] = dict(sorted(self.metrics.phases_s.items()))
        return {
            "schema_version": _SCHEMA_VERSION,
            "benchmark": self.benchmark,
            "run_id": self.run_id,
            "started_at": self.started_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "environment": {name: value for name, value in asdict(self.environment).items() if value is not None},
            "workload": asdict(self.workload),
            "metrics": metrics,
        }


def write_result(result: BenchmarkResult, path: Path) -> None:
    """Write one benchmark result as deterministic JSON."""

    payload = result.to_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
