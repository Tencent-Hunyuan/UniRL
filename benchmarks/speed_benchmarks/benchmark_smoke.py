"""Run a CPU-only synthetic benchmark and write one schema-v1 result."""

from __future__ import annotations

import argparse
import os
import platform
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from uuid import uuid4

from unirl.utils.benchmark_result import BenchmarkResult, Environment, Metrics, Workload, write_result
from unirl.utils.run_id import resolve_run_id

_HELP = """Run a CPU-only synthetic benchmark and write one schema-v1 result.

This smoke benchmark is a contract check, not a UniRL performance measurement.

    python -m benchmarks.speed_benchmarks.benchmark_smoke
"""


def _run_iteration(samples: int) -> None:
    for _ in range(samples):
        pass


def _torch_version() -> str | None:
    try:
        return metadata.version("torch")
    except metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=_HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=int, default=128, help="dummy samples per iteration (default: %(default)s)")
    parser.add_argument("--warmup-iterations", type=int, default=2, help="untimed iterations (default: %(default)s)")
    parser.add_argument("--measured-iterations", type=int, default=10, help="timed iterations (default: %(default)s)")
    parser.add_argument("--out", type=Path, default=Path("outputs/benchmark"), help="output dir (default: %(default)s)")
    args = parser.parse_args()
    if args.samples < 1 or args.warmup_iterations < 0 or args.measured_iterations < 1:
        parser.error("--samples must be positive; iterations must be warmup >= 0 and measured >= 1")

    for _ in range(args.warmup_iterations):
        _run_iteration(args.samples)
    started_at = datetime.now(timezone.utc)
    start = time.perf_counter()
    for _ in range(args.measured_iterations):
        _run_iteration(args.samples)
    wall_clock_s = time.perf_counter() - start
    run_id = resolve_run_id(os.environ.get("UNIRL_RUN_ID") or f"smoke-{int(time.time())}-{uuid4().hex[:6]}")
    result_path = args.out / f"{run_id}.json"
    write_result(
        BenchmarkResult(
            benchmark="smoke-synthetic",
            run_id=run_id,
            started_at=started_at,
            environment=Environment(
                python=platform.python_version(), torch=_torch_version(), device="cpu", world_size=1
            ),
            workload=Workload(
                samples=args.samples * args.measured_iterations,
                warmup_iterations=args.warmup_iterations,
                measured_iterations=args.measured_iterations,
            ),
            metrics=Metrics(wall_clock_s=wall_clock_s),
        ),
        result_path,
    )
    print("not a UniRL performance measurement")
    print(result_path)


if __name__ == "__main__":
    main()
