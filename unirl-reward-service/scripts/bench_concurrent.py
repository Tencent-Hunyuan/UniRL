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
import statistics
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from reward_service.client import RewardClient, RewardRequest

_BUNDLED_SAMPLE = Path(__file__).resolve().parent.parent / "tests" / "assets" / "sample.jpg"

# Thread-local so each worker reuses one RewardClient (one connection pool).
# Creating a new Session per request would add handshake/pool setup cost
# that swamps the actual scoring latency at high concurrency.
_thread_local = threading.local()


@dataclass
class _Outcome:
    """One request's outcome — what the aggregator needs to know."""

    ok: bool
    latency_s: float
    err: str | None = None
    # Per-reward failures reported in body["errors"] — counted separately
    # from transport failures (HTTP errors, timeouts) which go in ``err``.
    per_reward_errs: Counter = field(default_factory=Counter)


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

    # Server-side per-reward failures come back as missing keys in the
    # result dict. We don't have direct access to body["errors"] here
    # (the client strips it), so missing = failed.
    reward_errs: Counter = Counter()
    for r in results:
        for name in rewards:
            if name not in r:
                reward_errs[name] += 1

    return _Outcome(ok=True, latency_s=elapsed, per_reward_errs=reward_errs)


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
        return Counter(o.err.split(":", 1)[0] for o in self.outcomes if not o.ok)

    @property
    def per_reward_errs(self) -> Counter:
        agg: Counter = Counter()
        for o in self.outcomes:
            agg.update(o.per_reward_errs)
        return agg


def _run_one(
    args: argparse.Namespace,
    image: Image.Image,
    rewards: list[str],
    concurrency: int,
) -> _RunStats:
    total = args.total
    print(f"\n=== concurrency={concurrency}  total={total} ===", flush=True)

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
                args.batch_size,
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
    _print_stats(stats, batch_size=args.batch_size)
    return stats


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
    ap.add_argument("--url", required=True, help="Reward service URL.")
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

    if args.sweep is None and args.concurrency is None:
        ap.error("either --concurrency or --sweep is required")
    if args.total <= 0:
        ap.error("--total must be positive")

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

    print(
        f"url={args.url}  rewards={rewards}  batch_size={args.batch_size}  "
        f"image={image_path.name} size={image.size}"
    )

    levels = args.sweep if args.sweep is not None else [args.concurrency]

    # Three modes:
    #   - sweep (levels > 1) + isolated: for each reward, run the sweep.
    #     Prints one sweep summary per reward. Probably overkill; included
    #     for completeness.
    #   - single level + isolated: run each reward alone at one concurrency,
    #     print one per-reward comparison table. This is the common case.
    #   - not isolated: original behaviour — all rewards together.
    if args.per_reward_isolated:
        if len(levels) == 1:
            concurrency = levels[0]
            named: list[tuple[str, _RunStats]] = []
            for name in rewards:
                print(f"\n--- isolated: reward={name} ---", flush=True)
                stats = _run_one(args, image, [name], concurrency)
                named.append((name, stats))
            _print_per_reward_summary(named)
            return 0 if all(s.fail == 0 for _, s in named) else 3

        # sweep + isolated: nested loop.
        ok = True
        for name in rewards:
            print(f"\n########  reward={name}  ########", flush=True)
            per_level = [_run_one(args, image, [name], c) for c in levels]
            _print_sweep_summary(per_level)
            ok = ok and all(s.fail == 0 for s in per_level)
        return 0 if ok else 3

    all_stats = [_run_one(args, image, rewards, c) for c in levels]

    if len(all_stats) > 1:
        _print_sweep_summary(all_stats)

    # Non-zero exit if anything failed — makes CI / scripted use easier.
    return 0 if all(s.fail == 0 for s in all_stats) else 3


if __name__ == "__main__":
    sys.exit(main())
