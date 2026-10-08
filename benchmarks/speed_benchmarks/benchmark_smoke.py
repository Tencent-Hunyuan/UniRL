"""Run a CPU-only synthetic benchmark and write one schema-v1 result."""

from __future__ import annotations

import argparse
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

# Make the documented file-path invocation import the repository package.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from unirl.utils.benchmark_result import BenchmarkResult, Environment, Metrics, Workload, write_result

_HELP = """Run a CPU-only synthetic benchmark and write one schema-v1 result.

This smoke benchmark is a contract check, not a UniRL performance measurement.
"""
_DEFAULT_SAMPLES = 128
_DEFAULT_WARMUP_ITERATIONS = 2
_DEFAULT_MEASURED_ITERATIONS = 10
_DEFAULT_OUT = Path("outputs/benchmark")


def _run_iteration(samples: int) -> None:
    for _ in range(samples):
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=_HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=int, default=_DEFAULT_SAMPLES, help="dummy samples per iteration")
    parser.add_argument("--warmup-iterations", type=int, default=_DEFAULT_WARMUP_ITERATIONS)
    parser.add_argument("--measured-iterations", type=int, default=_DEFAULT_MEASURED_ITERATIONS)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    args = parser.parse_args()
    if args.samples < 1 or args.warmup_iterations < 0 or args.measured_iterations < 1:
        parser.error("--samples must be positive; iterations must be warmup >= 0 and measured >= 1")

    for _ in range(args.warmup_iterations):
        _run_iteration(args.samples)
    started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    start = time.perf_counter()
    for _ in range(args.measured_iterations):
        _run_iteration(args.samples)
    wall_clock_s = time.perf_counter() - start
    run_id = f"smoke-{int(time.time())}-{uuid4().hex[:6]}"
    result_path = args.out / f"{run_id}.json"
    write_result(
        BenchmarkResult(
            benchmark="smoke-synthetic",
            run_id=run_id,
            started_at=started_at,
            environment=Environment(python=platform.python_version(), torch="unavailable", device="cpu", world_size=1),
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
