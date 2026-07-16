#!/usr/bin/env python3
"""GPU smoke: prove the deep-research AGENTIC TRAINING step runs end-to-end.

Runs the M1 (calculator + math-verify, NO external services) recipe for a couple
of rollouts and asserts a real optimizer step actually happened: finite loss, an
on-policy importance ratio ~ 1, and reward in [0, 1]. This is the committed
regression guard the training path lacked — the unit tests cover only pure
helpers, and ``agentic_engine_smoke.py`` is rollout-only (it never runs
reward -> advantage -> ``stack.train_track``).

    QWEN3_INSTRUCT_PATH=/path/to/Qwen3-Instruct DATA_PATH=/path/to/calc.jsonl \
        NUM_DEVICES=8 .venv-sglang/bin/python scripts/train_deep_research_smoke.py

Exit 0 = PASS, non-zero = FAIL. GPU-only (needs a local model + >=2 GPUs).
``NUM_DEVICES`` must be a multiple of the node's GPU count (``devices_per_node``,
default 8) and divide ``batch_size * samples_per_prompt`` (8 in the M1 recipe).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

# One trainer log line per rollout, e.g.
#   rollout 9/30  reward=0.8750  loss=2.0012 gn=1.1480 lr=1.00e-06 ratio=1.0003±0.0078 ...
_ROLLOUT_RE = re.compile(r"rollout\s+\d+/\d+\s+reward=([\d.]+)\s+loss=([\d.eE+-]+).*?ratio=([\d.]+)")


def _log(msg: str) -> None:
    print(f"[dr-train-smoke] {msg}", flush=True)


def main() -> int:
    model = os.environ.get("QWEN3_INSTRUCT_PATH")
    data = os.environ.get("DATA_PATH")
    if not model or not data:
        _log("ERROR: set QWEN3_INSTRUCT_PATH and DATA_PATH")
        return 2
    num_devices = os.environ.get("NUM_DEVICES", "8")
    env = dict(
        os.environ,
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
    )
    cmd = [
        sys.executable,
        "-m",
        "unirl.train_deep_research",
        "--config-name=deep_research/deep_research_calc_mathverify",
        f"num_devices={num_devices}",
        "num_rollouts=2",
    ]
    _log("launching: " + " ".join(cmd))
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        _log("TIMEOUT after 1800s — FAIL")
        return 1

    log = proc.stdout + "\n" + proc.stderr
    rows = _ROLLOUT_RE.findall(log)
    if not rows:
        _log("no training rollout lines found — FAIL. Tail:")
        _log(log[-2500:])
        return 1

    ok = True
    for reward, loss, ratio in rows:
        rf, lf, ra = float(reward), float(loss), float(ratio)
        finite = (rf == rf) and (lf == lf)  # NaN != NaN
        in_range = 0.0 <= rf <= 1.0
        near_one = 0.8 <= ra <= 1.2  # strict on-policy MVP: ratio ~ 1
        _log(f"reward={rf} loss={lf} ratio={ra}  finite={finite} in[0,1]={in_range} ratio~1={near_one}")
        ok = ok and finite and in_range and near_one

    if proc.returncode != 0:
        _log(f"process exited {proc.returncode} — FAIL")
        return 1
    if not ok:
        _log("assertions FAILED")
        return 1
    _log(f"PASSED ✅  ({len(rows)} real training steps: finite loss, ratio~1, reward in [0,1])")
    return 0


if __name__ == "__main__":
    sys.exit(main())
