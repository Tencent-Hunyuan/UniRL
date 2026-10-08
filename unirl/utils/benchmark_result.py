"""Versioned, stdlib-only benchmark result schema and JSON writer."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SCHEMA_VERSION = 1


def _validate_number(value: int | float, name: str, *, integer: bool = False, strictly_positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be an int or float")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if integer and not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0 or (strictly_positive and value == 0):
        requirement = "strictly positive" if strictly_positive else "non-negative"
        raise ValueError(f"{name} must be {requirement}")


@dataclass(frozen=True)
class Environment:
    """Describe the runtime that produced a benchmark result."""

    python: str
    torch: str
    device: str
    world_size: int

    def to_dict(self) -> dict[str, Any]:
        _validate_number(self.world_size, "world_size", integer=True)
        return {
            "python": self.python,
            "torch": self.torch,
            "device": self.device,
            "world_size": self.world_size,
        }


@dataclass(frozen=True)
class Workload:
    """Describe sample and iteration counts for a benchmark run."""

    samples: int
    warmup_iterations: int
    measured_iterations: int

    def to_dict(self) -> dict[str, int]:
        for name, value in (
            ("samples", self.samples),
            ("warmup_iterations", self.warmup_iterations),
            ("measured_iterations", self.measured_iterations),
        ):
            _validate_number(value, name, integer=True)
        return {
            "samples": self.samples,
            "warmup_iterations": self.warmup_iterations,
            "measured_iterations": self.measured_iterations,
        }


@dataclass(frozen=True)
class Metrics:
    """Store measured duration and optional resource or phase metrics."""

    wall_clock_s: float
    peak_memory_bytes: int | None = None
    phases_s: dict[str, float] | None = None

    def to_dict(self, samples: int) -> dict[str, Any]:
        _validate_number(self.wall_clock_s, "wall_clock_s", strictly_positive=True)
        if self.peak_memory_bytes is not None:
            _validate_number(self.peak_memory_bytes, "peak_memory_bytes")
        if self.phases_s is not None:
            for name, value in self.phases_s.items():
                _validate_number(value, f"phases_s[{name!r}]")
        result: dict[str, Any] = {
            "wall_clock_s": self.wall_clock_s,
            "samples_per_second": samples / self.wall_clock_s,
        }
        if self.peak_memory_bytes is not None:
            result["peak_memory_bytes"] = self.peak_memory_bytes
        if self.phases_s is not None:
            result["phases_s"] = {name: self.phases_s[name] for name in sorted(self.phases_s)}
        return result


@dataclass(frozen=True)
class BenchmarkResult:
    """Represent one versioned benchmark result."""

    benchmark: str
    run_id: str
    started_at: str
    environment: Environment
    workload: Workload
    metrics: Metrics
    schema_version: int = _SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {_SCHEMA_VERSION}")
        workload = self.workload.to_dict()
        return {
            "schema_version": self.schema_version,
            "benchmark": self.benchmark,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "environment": self.environment.to_dict(),
            "workload": workload,
            "metrics": self.metrics.to_dict(workload["samples"]),
        }


def write_result(result: BenchmarkResult, path: Path) -> None:
    """Validate and write one benchmark result as deterministic JSON."""

    payload = result.to_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, allow_nan=False) + "\n")
