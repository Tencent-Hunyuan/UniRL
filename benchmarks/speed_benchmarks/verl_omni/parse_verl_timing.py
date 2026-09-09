"""Extract per-step wall-clock from a verl-omni console log."""

import argparse
import re
import statistics
from pathlib import Path

STEP_TIMING = re.compile(r"timing_s/step['\"]?[:=]\s*([0-9.]+)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("--skip", type=int, default=2)
    ap.add_argument("--samples-per-step", type=int)
    ap.add_argument("--gpus", type=int)
    args = ap.parse_args()

    vals = [float(m.group(1)) for m in STEP_TIMING.finditer(args.log.read_text(errors="replace"))]
    vals = vals[args.skip :]
    if not vals:
        raise SystemExit("no timing_s/step values found")
    vals.sort()
    med = statistics.median(vals)
    row = {
        "steps": len(vals),
        "median s/step": round(med, 1),
        "mean s/step": round(statistics.mean(vals), 1),
        "p90 s/step": round(vals[int(0.9 * (len(vals) - 1))], 1),
    }
    if args.samples_per_step:
        row["samples/s"] = round(args.samples_per_step / med, 2)
        if args.gpus:
            row["samples/GPU-h"] = round(3600 * args.samples_per_step / med / args.gpus, 1)
    print("| " + " | ".join(row) + " |")
    print("|" + "---|" * len(row))
    print("| " + " | ".join(str(v) for v in row.values()) + " |")


if __name__ == "__main__":
    main()
