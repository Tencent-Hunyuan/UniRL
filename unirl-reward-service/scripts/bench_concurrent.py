"""Concurrent load test for a running reward service.

Fires ``--total`` requests at ``--concurrency`` parallelism and reports
per-request latency (min / max / mean + percentiles), throughput, and
error rate. Supports a sweep mode (``--sweep 100 500 1000 2000``) to see
how the service degrades as concurrency climbs.

Each worker thread owns its own ``RewardClient`` (so the per-session
HTTP connection pool doesn't serialize calls). Threads are fine here
since ``requests`` is blocking and workers spend most of their time
waiting on I/O.

Usage:
    # 1000 requests at 200-way concurrency, single reward
    python3 scripts/bench_concurrent.py \\
        --url http://10.1.2.3:8080 --concurrency 200 --total 1000 --rewards clip

    # Sweep: 1000 requests at each concurrency level
    python3 scripts/bench_concurrent.py \\
        --url http://10.1.2.3:8080 --sweep 100 500 1000 2000 --total 1000

    # Per-reward isolated: one round per reward, then a comparison table.
    # A single /score request asking for N rewards is latency-bound by the
    # slowest, because HTTP returns all scores in one response. Use this
    # mode to see each reward's standalone latency.
    python3 scripts/bench_concurrent.py \\
        --url http://10.1.2.3:8080 --concurrency 200 --total 500 \\
        --rewards clip,hpsv2,hpsv3 --per-reward-isolated

Notes:
- ``--concurrency`` is the in-flight cap, not a synchronized-launch
  count. ThreadPoolExecutor starts workers as tasks are submitted;
  within a few ms they're all in flight, which is close enough for the
  1000-request volumes this script targets.
- Large sweeps print one progress line per 10% of completion so long
  runs aren't silent.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import socket
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from reward_service.client import RewardClient, RewardRequest

_BUNDLED_SAMPLE = Path(__file__).resolve().parent.parent / "tests" / "assets" / "sample.jpg"
_DEPLOYMENT_MODES = ("dedicated", "colocated-no-mps", "colocated-mps", "consolidated")
# Thread-local so each worker reuses one RewardClient (one connection pool).
# Creating a new Session per request would add handshake/pool setup cost
# that swamps the actual scoring latency at high concurrency.
_thread_local = threading.local()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


@dataclass
class _Outcome:
    """One request's outcome — what the aggregator needs to know."""

    ok: bool
    latency_s: float
    err: str | None = None
    # Per-reward failures reported in body["errors"] — counted separately
    # from transport failures (HTTP errors, timeouts) which go in ``err``.
    per_reward_errs: Counter = field(default_factory=Counter)
    # Flattened ``reward.submetric -> values`` for numerical parity checks.
    score_values: dict[str, list[float]] = field(default_factory=dict)


def _split_rewards(tokens: list[str]) -> list[str]:
    """Accept both ``--rewards clip hpsv2`` and ``--rewards clip,hpsv2``."""
    return [r for t in tokens for r in t.split(",") if r]


def _percentile(sorted_data: list[float], p: float) -> float:
    """Linear-interpolation percentile. ``sorted_data`` must be pre-sorted."""
    if not sorted_data:
        return float("nan")
    k = (len(sorted_data) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_data) - 1)
    if lo == hi:
        return sorted_data[lo]
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (k - lo)


def _get_client(url: str, timeout: float | None, trust_env: bool) -> RewardClient:
    client = getattr(_thread_local, "client", None)
    if client is None:
        client = RewardClient(url, timeout=timeout, trust_env=trust_env)
        _thread_local.client = client
    return client


def _fire_once(
    url: str,
    timeout: float | None,
    trust_env: bool,
    prompt: str,
    image: Image.Image,
    rewards: list[str],
    batch_size: int,
) -> _Outcome:
    client = _get_client(url, timeout, trust_env)
    reqs = [
        RewardRequest(history=[(prompt, image)], required_rewards=rewards)
        for _ in range(batch_size)
    ]
    t0 = time.perf_counter()
    try:
        results = client.score(reqs)
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return _Outcome(
            ok=False,
            latency_s=elapsed,
            err=f"{type(exc).__name__}: {exc}",
        )
    else:
        elapsed = time.perf_counter() - t0

    if len(results) != batch_size:
        return _Outcome(
            ok=False,
            latency_s=elapsed,
            err=f"ProtocolError: expected {batch_size} results, got {len(results)}",
        )

    # Server-side per-reward failures come back as missing keys in the
    # result dict. We don't have direct access to body["errors"] here
    # (the client strips it), so missing = failed.
    reward_errs: Counter = Counter()
    score_values: dict[str, list[float]] = {}
    for result in results:
        if not isinstance(result, dict):
            return _Outcome(
                ok=False,
                latency_s=elapsed,
                err=f"ProtocolError: result is {type(result).__name__}, expected dict",
            )
        for name in rewards:
            if name not in result:
                reward_errs[name] += 1
                continue
            reward_result = result[name]
            if not isinstance(reward_result, dict) or not reward_result:
                reward_errs[name] += 1
                continue
            invalid = any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                for value in reward_result.values()
            )
            if invalid:
                reward_errs[name] += 1
                continue
            for metric, value in reward_result.items():
                score_values.setdefault(f"{name}.{metric}", []).append(float(value))

    return _Outcome(
        ok=not reward_errs,
        latency_s=elapsed,
        err=None,
        per_reward_errs=reward_errs,
        score_values=score_values,
    )


@dataclass
class _RunStats:
    concurrency: int
    total: int
    wall_s: float
    outcomes: list[_Outcome]

    @property
    def ok(self) -> int:
        return sum(1 for o in self.outcomes if o.ok)

    @property
    def fail(self) -> int:
        return len(self.outcomes) - self.ok

    @property
    def latencies_ms(self) -> list[float]:
        # Only successful requests' latencies — failures get their own bucket.
        return [o.latency_s * 1000 for o in self.outcomes if o.ok]

    @property
    def qps(self) -> float:
        return self.ok / self.wall_s if self.wall_s > 0 else 0.0

    @property
    def err_kinds(self) -> Counter:
        return Counter(
            o.err.split(":", 1)[0]
            for o in self.outcomes
            if o.err is not None
        )

    @property
    def reward_failed_requests(self) -> int:
        return sum(bool(outcome.per_reward_errs) for outcome in self.outcomes)

    @property
    def per_reward_errs(self) -> Counter:
        agg: Counter = Counter()
        for o in self.outcomes:
            agg.update(o.per_reward_errs)
        return agg

    @property
    def score_values(self) -> dict[str, list[float]]:
        agg: dict[str, list[float]] = {}
        for outcome in self.outcomes:
            for name, values in outcome.score_values.items():
                agg.setdefault(name, []).extend(values)
        return agg


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command_output(command: list[str], timeout: float = 5.0) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _collect_environment() -> dict[str, Any]:
    environment: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {},
    }
    for name in ("unirl-reward-service", "ray", "torch", "transformers", "pillow"):
        version = _package_version(name)
        if version is not None:
            environment["packages"][name] = version

    git_root = Path(__file__).resolve().parents[2]
    git_sha = _command_output(["git", "-C", str(git_root), "rev-parse", "HEAD"])
    if git_sha is not None:
        environment["git_sha"] = git_sha

    driver = _command_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    )
    if driver is not None:
        environment["nvidia_driver"] = driver.splitlines()[0].strip()

    mps: dict[str, Any] = {
        "pipe_directory": os.environ.get("CUDA_MPS_PIPE_DIRECTORY"),
        "log_directory": os.environ.get("CUDA_MPS_LOG_DIRECTORY"),
        "active_thread_percentage": os.environ.get(
            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"
        ),
        "pinned_device_memory_limit": os.environ.get(
            "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT"
        ),
    }
    mps["client_environment_present"] = any(mps.values())
    environment["mps"] = mps
    return environment


def _query_gpus() -> list[dict[str, Any]]:
    output = _command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,"
            "utilization.gpu,utilization.memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if output is None:
        return []

    gpus = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 7:
            continue
        try:
            gpus.append(
                {
                    "index": int(fields[0]),
                    "uuid": fields[1],
                    "name": fields[2],
                    "memory_total_mib": float(fields[3]),
                    "memory_used_mib": float(fields[4]),
                    "gpu_utilization_percent": float(fields[5]),
                    "memory_utilization_percent": float(fields[6]),
                }
            )
        except ValueError:
            continue
    return gpus


class _GpuMonitor:
    """Best-effort same-host nvidia-smi sampler."""

    def __init__(self, interval_s: float):
        self.interval_s = interval_s
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples_lock = threading.Lock()

    def start(self) -> None:
        if self.interval_s <= 0 or shutil.which("nvidia-smi") is None:
            return
        self._sample()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        # _query_gpus has a bounded timeout, so a full join is finite and
        # prevents summary() racing a late sampler append.
        self._thread.join()
        self._sample()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()

    def _sample(self) -> None:
        gpus = _query_gpus()
        if gpus:
            with self._samples_lock:
                self.samples.append({"timestamp": _utc_now(), "gpus": gpus})

    def summary(self) -> dict[str, Any] | None:
        with self._samples_lock:
            samples_snapshot = list(self.samples)
        if not samples_snapshot:
            return None
        by_uuid: dict[str, list[dict[str, Any]]] = {}
        for sample in samples_snapshot:
            for gpu in sample["gpus"]:
                by_uuid.setdefault(gpu["uuid"], []).append(gpu)

        devices = []
        for uuid, samples in sorted(by_uuid.items()):
            first = samples[0]
            steady_samples = samples[max(len(samples) // 2, 0):]
            devices.append(
                {
                    "index": first["index"],
                    "uuid": uuid,
                    "name": first["name"],
                    "memory_total_mib": first["memory_total_mib"],
                    "peak_memory_used_mib": max(
                        sample["memory_used_mib"] for sample in samples
                    ),
                    "steady_memory_used_mib": statistics.fmean(
                        sample["memory_used_mib"] for sample in steady_samples
                    ),
                    "mean_gpu_utilization_percent": statistics.fmean(
                        sample["gpu_utilization_percent"] for sample in samples
                    ),
                    "mean_memory_utilization_percent": statistics.fmean(
                        sample["memory_utilization_percent"] for sample in samples
                    ),
                }
            )
        return {
            "sample_interval_s": self.interval_s,
            "sample_count": len(samples_snapshot),
            "devices": devices,
            "limitations": [
                "nvidia-smi samples are node-wide, not attributed per reward",
                "Tensor Core activity is unavailable from this sampler",
                "MPS may attribute client activity to the server process",
            ],
        }


def _score_summary(values: dict[str, list[float]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for name, metric_values in sorted(values.items()):
        if not metric_values:
            continue
        sorted_values = sorted(metric_values)
        summary[name] = {
            "count": len(sorted_values),
            "min": sorted_values[0],
            "mean": statistics.fmean(sorted_values),
            "max": sorted_values[-1],
            # Full sorted vectors make parity tolerance-based and detect
            # missing/corrupted items without relying on aggregate means.
            "values": sorted_values,
        }
    return summary


def _stats_to_dict(
    stats: _RunStats,
    *,
    rewards: list[str],
    batch_size: int,
    label: str | None = None,
    repetition: int = 1,
    gpu_count: float | None = None,
) -> dict[str, Any]:
    latencies = sorted(stats.latencies_ms)
    result: dict[str, Any] = {
        "label": label,
        "repetition": repetition,
        "rewards": rewards,
        "concurrency": stats.concurrency,
        "requests": stats.total,
        "batch_size": batch_size,
        "items": stats.total * batch_size,
        "wall_clock_s": stats.wall_s,
        "successful_requests": stats.ok,
        "failed_requests": stats.fail,
        "reward_failed_requests": stats.reward_failed_requests,
        "requests_per_second": stats.qps,
        "items_per_second": stats.qps * batch_size,
        "transport_errors": dict(sorted(stats.err_kinds.items())),
        "per_reward_errors": dict(sorted(stats.per_reward_errs.items())),
        "scores": _score_summary(stats.score_values),
    }
    if latencies:
        result["latency_ms"] = {
            "min": latencies[0],
            "mean": statistics.fmean(latencies),
            "max": latencies[-1],
            "p50": _percentile(latencies, 50),
            "p90": _percentile(latencies, 90),
            "p95": _percentile(latencies, 95),
            "p99": _percentile(latencies, 99),
        }
    if gpu_count is not None:
        result["reward_evaluations_per_gpu_hour"] = (
            result["items_per_second"] * len(rewards) * 3600.0 / gpu_count
        )
    return result


def _run_one(
    args: argparse.Namespace,
    image: Image.Image,
    rewards: list[str],
    concurrency: int,
    batch_size: int,
) -> _RunStats:
    total = args.total
    print(
        f"\n=== concurrency={concurrency}  batch_size={batch_size}  "
        f"total={total} ===",
        flush=True,
    )

    outcomes: list[_Outcome] = []
    progress_step = max(total // 10, 1)
    heartbeat_every_s = 30.0  # stay noisy when --timeout None leaves the
                               # bench blocked on slow server responses
    t0 = time.perf_counter()
    last_heartbeat = t0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                _fire_once,
                args.url,
                args.timeout,
                args.trust_env,
                args.prompt,
                image,
                rewards,
                batch_size,
            )
            for _ in range(total)
        ]
        for i, fut in enumerate(as_completed(futures), start=1):
            outcomes.append(fut.result())
            now = time.perf_counter()
            if i % progress_step == 0 or i == total:
                print(f"  progress: {i}/{total}", flush=True)
                last_heartbeat = now
            elif now - last_heartbeat >= heartbeat_every_s:
                print(
                    f"  heartbeat: {i}/{total} done, waiting on {total - i} "
                    f"({now - t0:.0f}s elapsed)",
                    flush=True,
                )
                last_heartbeat = now
    wall = time.perf_counter() - t0

    stats = _RunStats(concurrency=concurrency, total=total, wall_s=wall, outcomes=outcomes)
    _print_stats(stats, batch_size=batch_size)
    return stats


def _warm_up(
    args: argparse.Namespace,
    image: Image.Image,
    reward_sets: list[list[str]],
    batch_sizes: list[int],
) -> None:
    if args.warmup_requests == 0:
        return
    print(
        f"\n=== warmup requests={args.warmup_requests} "
        f"batch_sizes={batch_sizes} ===",
        flush=True,
    )
    for rewards in reward_sets:
        for batch_size in batch_sizes:
            for _ in range(args.warmup_requests):
                outcome = _fire_once(
                    args.url,
                    args.timeout,
                    args.trust_env,
                    args.prompt,
                    image,
                    rewards,
                    batch_size,
                )
                if not outcome.ok or outcome.per_reward_errs:
                    detail = outcome.err or dict(outcome.per_reward_errs)
                    raise RuntimeError(f"warmup failed for {rewards}: {detail}")


def _print_stats(s: _RunStats, batch_size: int) -> None:
    lat = sorted(s.latencies_ms)
    print(f"wall time:     {s.wall_s:.2f}s")
    print(
        f"throughput:    {s.qps:.1f} req/s  "
        f"({s.qps * batch_size:.1f} items/s @ batch={batch_size})"
    )
    total = s.ok + s.fail
    success_pct = 100.0 * s.ok / total if total else 0.0
    print(f"success/fail:  {s.ok} / {s.fail}  ({success_pct:.2f}% ok)")
    if lat:
        print(
            f"latency ms:    min={lat[0]:.0f}  mean={statistics.mean(lat):.0f}  "
            f"max={lat[-1]:.0f}"
        )
        print(
            f"               p50={_percentile(lat, 50):.0f}  "
            f"p90={_percentile(lat, 90):.0f}  "
            f"p95={_percentile(lat, 95):.0f}  "
            f"p99={_percentile(lat, 99):.0f}"
        )
    if s.err_kinds:
        print("transport errors:")
        for kind, n in s.err_kinds.most_common():
            print(f"  {kind}: {n}")
    if s.per_reward_errs:
        print("per-reward errors (server reported missing):")
        for name, n in s.per_reward_errs.most_common():
            print(f"  {name}: {n}")


def _print_sweep_summary(all_stats: list[_RunStats]) -> None:
    print("\n=== sweep summary ===")
    header = (
        f"  {'concur':>6s}  {'qps':>8s}  {'min_ms':>7s}  {'mean_ms':>8s}  "
        f"{'max_ms':>7s}  {'p95_ms':>7s}  {'p99_ms':>7s}  {'fail':>5s}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for s in all_stats:
        lat = sorted(s.latencies_ms)
        if lat:
            print(
                f"  {s.concurrency:>6d}  {s.qps:>8.1f}  "
                f"{lat[0]:>7.0f}  {statistics.mean(lat):>8.0f}  "
                f"{lat[-1]:>7.0f}  {_percentile(lat, 95):>7.0f}  "
                f"{_percentile(lat, 99):>7.0f}  {s.fail:>5d}"
            )
        else:
            print(f"  {s.concurrency:>6d}  (all {s.fail} requests failed)")


def _print_per_reward_summary(named: list[tuple[str, _RunStats]]) -> None:
    print("\n=== per-reward isolated summary ===")
    name_w = max(len(n) for n, _ in named)
    header = (
        f"  {'reward':<{name_w}s}  {'qps':>8s}  {'min_ms':>7s}  {'mean_ms':>8s}  "
        f"{'max_ms':>7s}  {'p95_ms':>7s}  {'p99_ms':>7s}  {'fail':>5s}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, s in named:
        lat = sorted(s.latencies_ms)
        if lat:
            print(
                f"  {name:<{name_w}s}  {s.qps:>8.1f}  "
                f"{lat[0]:>7.0f}  {statistics.mean(lat):>8.0f}  "
                f"{lat[-1]:>7.0f}  {_percentile(lat, 95):>7.0f}  "
                f"{_percentile(lat, 99):>7.0f}  {s.fail:>5d}"
            )
        else:
            print(f"  {name:<{name_w}s}  (all {s.fail} requests failed)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", help="Reward service URL.")
    ap.add_argument(
        "--deployment-mode",
        choices=_DEPLOYMENT_MODES,
        default="dedicated",
        help="Label the deployment being measured. This records, but does not "
             "change, service placement. Default: dedicated.",
    )
    ap.add_argument(
        "--deployment-note",
        default=None,
        help="Free-form deployment/configuration note stored in JSON.",
    )
    ap.add_argument(
        "--physical-gpu",
        default=None,
        help="Physical GPU UUID/index used by the service, for same-card checks.",
    )
    ap.add_argument(
        "--gpu-count",
        type=float,
        default=None,
        help="Physical GPUs used by the measured deployment. Enables "
             "reward-evaluations/GPU-hour reporting.",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write deterministic schema-v1 JSON to this path.",
    )
    ap.add_argument(
        "--run-id",
        default=None,
        help="Experiment/run identifier. Defaults to a UTC timestamp.",
    )
    ap.add_argument(
        "--gpu-sample-interval",
        type=float,
        default=0.0,
        help="Seconds between same-host nvidia-smi samples; 0 disables. "
             "Default: 0 (disabled, preserving legacy benchmark overhead).",
    )
    ap.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Repeat every workload point this many times. Default: 1.",
    )
    ap.add_argument(
        "--warmup-requests",
        type=int,
        default=0,
        help="Untimed warm-up requests per reward set and batch size. "
             "Default: 0, preserving legacy behavior.",
    )
    ap.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Number of in-flight requests. Required unless --sweep is given.",
    )
    ap.add_argument(
        "--sweep",
        type=int,
        nargs="+",
        default=None,
        help="Run the bench at each concurrency level in sequence. "
             "e.g. --sweep 100 500 1000 2000.",
    )
    ap.add_argument(
        "--total", type=int, default=1000,
        help="Total requests sent per concurrency level. Default 1000.",
    )
    ap.add_argument(
        "--rewards", nargs="+", default=["clip"],
        help="Rewards to query (space- or comma-separated). Default: clip.",
    )
    ap.add_argument(
        "--batch-size", type=int, default=1,
        help="Items per /score request (same image reused). Default 1.",
    )
    ap.add_argument(
        "--batch-sweep",
        type=int,
        nargs="+",
        default=None,
        help="Run each concurrency at every listed request batch size. "
             "Overrides --batch-size.",
    )
    ap.add_argument("--prompt", default="a cute dog running in the park")
    ap.add_argument(
        "--image", type=Path, default=None,
        help="Candidate image. Defaults to tests/assets/sample.jpg if present.",
    )
    ap.add_argument(
        "--timeout",
        type=lambda x: None if x.lower() == "none" else float(x),
        default=None,
        help="Per-request HTTP timeout (seconds), or 'none' (default) to "
             "wait indefinitely. The bench prints a heartbeat every 30s "
             "while waiting. Giving up on the client does NOT cancel the "
             "server-side Ray task, so finishing the bench early would "
             "leave the server still working — use a number only if you "
             "want partial stats and are OK with server overhang.",
    )
    ap.add_argument(
        "--trust-env", action="store_true",
        help="Honour HTTP(S)_PROXY env vars (default: ignore).",
    )
    ap.add_argument(
        "--per-reward-isolated", action="store_true",
        help="Run one bench round per reward in --rewards (instead of one "
             "round asking for all of them together). Use this to compare "
             "the standalone latency of each reward — when asked for "
             "together, each request's latency is dominated by the slowest "
             "reward (HTTP must return all scores in one response).",
    )
    args = ap.parse_args()

    if args.url is None:
        ap.error("--url is required")
    if args.sweep is None and args.concurrency is None:
        ap.error("either --concurrency or --sweep is required")
    if args.total <= 0:
        ap.error("--total must be positive")
    if args.batch_size <= 0:
        ap.error("--batch-size must be positive")
    if args.batch_sweep is not None and any(size <= 0 for size in args.batch_sweep):
        ap.error("--batch-sweep values must be positive")
    if args.concurrency is not None and args.concurrency <= 0:
        ap.error("--concurrency must be positive")
    if args.sweep is not None and any(level <= 0 for level in args.sweep):
        ap.error("--sweep values must be positive")
    if args.gpu_sample_interval < 0:
        ap.error("--gpu-sample-interval must be non-negative")
    if args.repetitions <= 0:
        ap.error("--repetitions must be positive")
    if args.gpu_count is not None and (
        not math.isfinite(args.gpu_count) or args.gpu_count <= 0
    ):
        ap.error("--gpu-count must be finite and positive")
    if args.warmup_requests < 0:
        ap.error("--warmup-requests must be non-negative")

    rewards = _split_rewards(args.rewards)
    if not rewards:
        ap.error("--rewards is empty after parsing")

    # Resolve image path; pre-decode so worker threads don't race inside PIL.
    if args.image is not None:
        image_path = args.image
    elif _BUNDLED_SAMPLE.is_file():
        image_path = _BUNDLED_SAMPLE
    else:
        print(
            "✗ --image is required (no bundled sample.jpg found)",
            file=sys.stderr,
        )
        return 2
    image = Image.open(image_path).convert("RGB")
    image.load()  # force eager decode so threads only do read-only access

    # Quick probe: is the service up and does it advertise what we ask for?
    probe = RewardClient(args.url, timeout=30.0, trust_env=args.trust_env)
    try:
        advertised = probe.rewards()
    except Exception as e:
        print(f"✗ could not reach {args.url}: {e}", file=sys.stderr)
        return 1
    missing = set(rewards) - set(advertised)
    if missing:
        print(
            f"✗ service doesn't advertise reward(s): {sorted(missing)}\n"
            f"  advertised: {advertised}",
            file=sys.stderr,
        )
        return 1

    levels = args.sweep if args.sweep is not None else [args.concurrency]
    batch_sizes = (
        args.batch_sweep if args.batch_sweep is not None else [args.batch_size]
    )
    print(
        f"url={args.url}  rewards={rewards}  batch_sizes={batch_sizes}  "
        f"image={image_path.name} size={image.size}  "
        f"deployment={args.deployment_mode}"
    )

    reward_sets = (
        [[name] for name in rewards]
        if args.per_reward_isolated
        else [rewards]
    )
    try:
        _warm_up(args, image, reward_sets, batch_sizes)
    except RuntimeError as exc:
        print(f"✗ {exc}", file=sys.stderr)
        return 1

    started_at = _utc_now()
    environment = _collect_environment() if args.output is not None else None
    monitor = _GpuMonitor(
        args.gpu_sample_interval if args.output is not None else 0.0
    )
    recorded_runs: list[dict[str, Any]] = []
    exit_code = 0
    monitor.start()

    completed: list[tuple[str, _RunStats]] = []
    try:
        for repetition in range(1, args.repetitions + 1):
            for batch_size in batch_sizes:
                for reward_set in reward_sets:
                    stats_list = [
                        _run_one(args, image, reward_set, level, batch_size)
                        for level in levels
                    ]
                    label = (
                        f"isolated:{reward_set[0]}"
                        if args.per_reward_isolated
                        else None
                    )
                    for stats in stats_list:
                        recorded_runs.append(
                            _stats_to_dict(
                                stats,
                                rewards=reward_set,
                                batch_size=batch_size,
                                label=label,
                                repetition=repetition,
                                gpu_count=args.gpu_count,
                            )
                        )
                        name = reward_set[0] if label else ",".join(reward_set)
                        completed.append(
                            (f"{name}@b{batch_size}#{repetition}", stats)
                        )
                    if len(stats_list) > 1:
                        _print_sweep_summary(stats_list)
    finally:
        monitor.stop()

    if args.per_reward_isolated and len(levels) == 1:
        _print_per_reward_summary(completed)
    exit_code = (
        0
        if all(stats.fail == 0 and not stats.per_reward_errs for _, stats in completed)
        else 3
    )

    if args.output is not None:
        payload: dict[str, Any] = {
            "schema_version": 1,
            "benchmark": "reward-service-concurrent",
            "run_id": args.run_id or started_at.replace(":", "").replace("-", ""),
            "started_at": started_at,
            "finished_at": _utc_now(),
            "deployment": {
                "mode": args.deployment_mode,
                "note": args.deployment_note,
                "physical_gpu": args.physical_gpu,
                "gpu_count": args.gpu_count,
            },
            "environment": environment,
            "workload": {
                "url": args.url,
                "rewards": rewards,
                "prompt": args.prompt,
                "image": {
                    "path": str(image_path),
                    "sha256": _sha256_file(image_path),
                    "width": image.width,
                    "height": image.height,
                },
                "batch_sizes": batch_sizes,
                "requests_per_run": args.total,
                "concurrency_levels": levels,
                "repetitions": args.repetitions,
                "warmup_requests_per_reward_set_and_batch": args.warmup_requests,
                "per_reward_isolated": args.per_reward_isolated,
            },
            "runs": recorded_runs,
        }
        gpu_observations = monitor.summary()
        if gpu_observations is not None:
            payload["gpu_observations"] = gpu_observations
        _write_json(args.output, payload)
        print(f"\nwrote benchmark JSON: {args.output}")

    # Non-zero exit if anything failed — makes CI / scripted use easier.
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
