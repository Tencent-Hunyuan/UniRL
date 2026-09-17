from __future__ import annotations

import importlib.util
import sys
import time
from collections import Counter
from pathlib import Path

from PIL import Image

_SCRIPT = Path(__file__).parents[1] / "scripts" / "bench_concurrent.py"
_SPEC = importlib.util.spec_from_file_location("bench_concurrent", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
bench = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = bench
_SPEC.loader.exec_module(bench)


def test_stats_json_contains_failures_latency_and_scores() -> None:
    stats = bench._RunStats(
        concurrency=2,
        total=2,
        wall_s=0.5,
        outcomes=[
            bench._Outcome(
                ok=True,
                latency_s=0.1,
                score_values={"clip.clip": [0.25, 0.75]},
            ),
            bench._Outcome(
                ok=False,
                latency_s=0.2,
                err="TimeoutError: late",
                per_reward_errs=Counter({"clip": 1}),
            ),
        ],
    )
    result = bench._stats_to_dict(
        stats, rewards=["clip"], batch_size=2, label=None
    )
    assert result["successful_requests"] == 1
    assert result["failed_requests"] == 1
    assert result["latency_ms"]["p99"] == 100.0
    assert result["transport_errors"] == {"TimeoutError": 1}
    assert result["scores"]["clip.clip"]["mean"] == 0.5


def test_gpu_monitor_summary_uses_peak_and_mean() -> None:
    monitor = bench._GpuMonitor(1.0)
    monitor.samples = [
        {
            "timestamp": str(index),
            "gpus": [{
                "index": 0,
                "uuid": "GPU-0",
                "name": "H20",
                "memory_total_mib": 1000.0,
                "memory_used_mib": memory,
                "gpu_utilization_percent": utilization,
                "memory_utilization_percent": utilization / 2,
            }],
        }
        for index, (memory, utilization) in enumerate(((100.0, 20.0), (300.0, 80.0)))
    ]
    device = monitor.summary()["devices"][0]
    assert device["peak_memory_used_mib"] == 300.0
    assert device["mean_gpu_utilization_percent"] == 50.0


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


def test_gpu_monitor_stop_waits_for_sampler(monkeypatch) -> None:
    def slow_query():
        time.sleep(0.03)
        return []

    monkeypatch.setattr(bench, "_query_gpus", slow_query)
    monkeypatch.setattr(bench.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    monitor = bench._GpuMonitor(0.01)
    monitor.start()
    monitor.stop()
    assert monitor._thread is not None and not monitor._thread.is_alive()
