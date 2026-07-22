"""Extract per-step wall-clock from UniRL training logs into a throughput table.

UniRL logs one line per training step (``rollout N/M  reward=...``); deltas
between consecutive line timestamps give the end-to-end step time (generate +
reward + train + weight sync). For the per-phase breakdown use the wandb
``perf/*`` keys instead (``perf/step_time_s`` / ``perf/<phase>_time_s``;
``perf/rollout_time_s`` on older runs).

    python benchmarks/speed_benchmarks/parse_perf.py a.log b.log \\
        --samples-per-step 48 --gpus 8
"""

from __future__ import annotations

import argparse
import re
import statistics
from datetime import datetime
from pathlib import Path

# Hydra job_logging default: "[%(asctime)s][%(name)s][%(levelname)s] - rollout N/M  reward=..."
# unirl.utils.configure_logger: "%(asctime)s - %(name)s - %(levelname)s - rollout N/M  reward=..."
_STEP_LINE = re.compile(r"^\[?(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+\]?.* - rollout (\d+)/\d+\s+reward=")


def step_deltas(log_path: Path, skip: int) -> list[float]:
    stamps = []
    with open(log_path, errors="replace") as f:
        for line in f:
            match = _STEP_LINE.match(line)
            if match:
                stamps.append(datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S"))
    deltas = [(b - a).total_seconds() for a, b in zip(stamps, stamps[1:])]
    return [d for d in deltas[skip:] if d > 0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--skip", type=int, default=2, help="warmup steps to drop (default: %(default)s)")
    parser.add_argument("--samples-per-step", type=int, help="rollout samples per step (batch_size × group size)")
    parser.add_argument("--gpus", type=int, help="GPUs used, for samples/GPU-hour")
    args = parser.parse_args()

    header = ["log", "steps", "median s/step", "mean s/step", "p90 s/step"]
    if args.samples_per_step:
        header += ["samples/s"] + (["samples/GPU-h"] if args.gpus else [])
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for log_path in args.logs:
        deltas = step_deltas(log_path, args.skip)
        if not deltas:
            print(f"| {log_path.name} | no step lines found |" + " |" * (len(header) - 2))
            continue
        deltas.sort()
        median = statistics.median(deltas)
        row = [
            log_path.name,
            str(len(deltas)),
            f"{median:.1f}",
            f"{statistics.mean(deltas):.1f}",
            f"{deltas[int(0.9 * (len(deltas) - 1))]:.1f}",
        ]
        if args.samples_per_step:
            row.append(f"{args.samples_per_step / median:.2f}")
            if args.gpus:
                row.append(f"{3600 * args.samples_per_step / median / args.gpus:.1f}")
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
