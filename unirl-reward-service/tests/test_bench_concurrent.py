from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

_SCRIPT = Path(__file__).parents[1] / "scripts" / "bench_concurrent.py"
_SPEC = importlib.util.spec_from_file_location("bench_concurrent", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
bench = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = bench
_SPEC.loader.exec_module(bench)


def test_invalid_server_output_is_counted_as_failure() -> None:
    class Client:
        def score(self, requests):
            return [{"clip": {"clip": float("nan")}}]

    bench._thread_local.client = Client()
    try:
        outcome = bench._fire_once(
            "unused", None, False, "test", Image.new("RGB", (2, 2)), ["clip"], 1
        )
    finally:
        del bench._thread_local.client
    assert not outcome.ok
    assert outcome.per_reward_errs == Counter({"clip": 1})


def test_run_record_keeps_scores_and_throughput() -> None:
    stats = bench._RunStats(
        concurrency=2,
        total=2,
        wall_s=0.5,
        outcomes=[
            bench._Outcome(ok=True, latency_s=0.1, score_values={"clip.clip": [0.25, 0.75]}),
            bench._Outcome(ok=False, latency_s=0.2, err="TimeoutError: late"),
        ],
    )
    record = bench._run_record(stats, ["clip"], batch_size=2, repetition=1)
    assert record["successful_requests"] == 1
    assert record["failed_requests"] == 1
    assert record["items_per_second"] == 4.0
    assert record["scores"]["clip.clip"]["mean"] == 0.5
