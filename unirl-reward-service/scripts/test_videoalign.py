"""VideoAlign smoke test + benchmark.

Run functional validation and performance tests against a running VideoAlign RewardService.
Includes health checks, single-video, batch, video_path modes, and latency measurements.

Usage:
    # Use sample videos from the VideoAlign repository (auto-detected)
    python3 scripts/test_videoalign.py

    # Specify the service URL
    python3 scripts/test_videoalign.py --url http://localhost:8090

    # Specify video files
    python3 scripts/test_videoalign.py --video /path/to/video.mp4

    # Run functional validation only, skipping the benchmark
    python3 scripts/test_videoalign.py --no-bench

    # Run the full benchmark with multiple batch sizes
    python3 scripts/test_videoalign.py --bench-full
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import requests

# Default video paths (VideoAlign repository samples)
_VIDEO_DIRS = [
    Path("/path/to/VideoAlign/datasets/train/videos"),
    Path(__file__).resolve().parent.parent / "tests" / "assets",
]

_PROMPTS = {
    "example_1_A.mp4": "The camera remains still, a girl with braided hair and wearing a pink dress approached the chair in the room and sat on it.",
    "example_1_B.mp4": "The camera remains still, a girl with braided hair and wearing a pink dress approached the chair in the room and sat on it.",
    "example_2_A.mp4": "The camera follows a young explorer through an abandoned urban building at night.",
    "example_2_B.mp4": "The camera follows a young explorer through an abandoned urban building at night.",
    "example_3_A.mp4": "A lively street scene with people walking and cars passing by.",
    "example_3_B.mp4": "A lively street scene with people walking and cars passing by.",
}

_DEFAULT_PROMPT = "A high quality video showing a natural scene."


def _find_videos() -> list[Path]:
    """Search known directories for .mp4 test files."""
    for d in _VIDEO_DIRS:
        if d.is_dir():
            videos = sorted(d.glob("*.mp4"))
            if videos:
                return videos
    return []


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _prompt_for(path: Path) -> str:
    return _PROMPTS.get(path.name, _DEFAULT_PROMPT)


def _post_score(url: str, payload: dict, timeout: float = 300) -> dict:
    resp = requests.post(f"{url}/score", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# Test functions

def test_health(url: str) -> bool:
    print("-- Health Check --")
    try:
        resp = requests.get(f"{url}/health", timeout=10)
        resp.raise_for_status()
        body = resp.json()
        print(f"  Status: {body['status']}")
        for name, replicas in body.get("rewards", {}).items():
            print(f"  {name}: {replicas}")
        print()
        return body["status"] == "ok"
    except Exception as e:
        print(f"  [FAIL] {e}\n")
        return False


def test_rewards_endpoint(url: str) -> list[str]:
    print("-- Registered Rewards --")
    resp = requests.get(f"{url}/rewards", timeout=10)
    resp.raise_for_status()
    rewards = resp.json()["rewards"]
    print(f"  {rewards}")
    if "videoalign" not in rewards:
        print("  [WARN] videoalign is not registered!")
    print()
    return rewards


def test_single_video_b64(url: str, video: Path) -> bool:
    print("-- Single Video Test (video_b64) --")
    print(f"  Video: {video.name} ({video.stat().st_size / 1024:.0f} KB)")

    payload = {
        "requests": [{
            "history": [{"text": _prompt_for(video), "video_b64": _b64(video)}],
            "required_rewards": ["videoalign"],
        }]
    }

    t0 = time.perf_counter()
    body = _post_score(url, payload)
    elapsed = time.perf_counter() - t0

    result = body["results"][0]
    errors = body["errors"][0]

    if errors:
        print(f"  [FAIL] Error: {errors}")
        print(f"  Latency: {elapsed:.2f}s\n")
        return False

    scores = result.get("videoalign", {})
    print(f"  VQ={scores['VQ']:.4f}  MQ={scores['MQ']:.4f}  TA={scores['TA']:.4f}  Overall={scores['Overall']:.4f}")
    print(f"  Latency: {elapsed:.2f}s")
    print()
    return True


def test_single_video_path(url: str, video: Path) -> bool:
    print("-- Single Video Test (video_path) --")
    print(f"  Path: {video}")

    payload = {
        "requests": [{
            "history": [{"text": _prompt_for(video), "video_path": str(video)}],
            "required_rewards": ["videoalign"],
        }]
    }

    t0 = time.perf_counter()
    body = _post_score(url, payload)
    elapsed = time.perf_counter() - t0

    result = body["results"][0]
    errors = body["errors"][0]

    if errors:
        print(f"  [FAIL] Error: {errors}")
        print(f"  Latency: {elapsed:.2f}s\n")
        return False

    scores = result.get("videoalign", {})
    print(f"  VQ={scores['VQ']:.4f}  MQ={scores['MQ']:.4f}  TA={scores['TA']:.4f}  Overall={scores['Overall']:.4f}")
    print(f"  Latency: {elapsed:.2f}s")
    print()
    return True


def test_batch(url: str, videos: list[Path]) -> bool:
    n = min(len(videos), 6)
    batch = videos[:n]
    print(f"-- Batch Test ({n} videos, video_b64) --")

    payload = {
        "requests": [
            {
                "history": [{"text": _prompt_for(v), "video_b64": _b64(v)}],
                "required_rewards": ["videoalign"],
            }
            for v in batch
        ]
    }

    t0 = time.perf_counter()
    body = _post_score(url, payload)
    elapsed = time.perf_counter() - t0

    all_ok = True
    for i, (result, errors) in enumerate(zip(body["results"], body["errors"])):
        if errors:
            print(f"  [{i}] {batch[i].name}: ERROR {errors}")
            all_ok = False
        else:
            s = result.get("videoalign", {})
            print(f"  [{i}] {batch[i].name}: VQ={s['VQ']:.3f} MQ={s['MQ']:.3f} TA={s['TA']:.3f} Overall={s['Overall']:.3f}")

    per_video = elapsed / n
    print(f"  Total latency: {elapsed:.2f}s | Per video: {per_video:.2f}s | Throughput: {n / elapsed:.1f} videos/s")
    print()
    return all_ok


def test_consistency(url: str, video: Path, n_runs: int = 3) -> bool:
    """Score the same video multiple times and check result consistency."""
    print(f"-- Consistency Test ({n_runs} repeated runs) --")

    payload = {
        "requests": [{
            "history": [{"text": _prompt_for(video), "video_b64": _b64(video)}],
            "required_rewards": ["videoalign"],
        }]
    }

    results = []
    for i in range(n_runs):
        body = _post_score(url, payload)
        scores = body["results"][0].get("videoalign", {})
        results.append(scores)

    # Check whether scores are consistent within tolerance.
    ok = True
    for key in ("VQ", "MQ", "TA", "Overall"):
        values = [r[key] for r in results]
        spread = max(values) - min(values)
        status = "OK" if spread < 0.01 else "DRIFT"
        if spread >= 0.01:
            ok = False
        print(f"  {key}: {values} spread={spread:.6f} [{status}]")

    print()
    return ok


# Benchmark

def benchmark(url: str, video: Path, batch_sizes: list[int], warmup: int = 2, repeats: int = 3) -> None:
    print("══════════════════════════════════════")
    print(" VideoAlign Performance Benchmark")
    print("══════════════════════════════════════")

    video_b64 = _b64(video)
    prompt = _prompt_for(video)

    # Warmup
    print(f"\n  Warmup ({warmup} runs)...")
    for _ in range(warmup):
        payload = {
            "requests": [{
                "history": [{"text": prompt, "video_b64": video_b64}],
                "required_rewards": ["videoalign"],
            }]
        }
        _post_score(url, payload)

    # Per batch size
    print(f"\n  {'Batch':>5s}  {'Avg(s)':>8s}  {'Per-vid(s)':>10s}  {'Throughput':>10s}  {'Runs':>20s}")
    print(f"  {'─' * 5}  {'─' * 8}  {'─' * 10}  {'─' * 10}  {'─' * 20}")

    for bs in batch_sizes:
        payload = {
            "requests": [
                {
                    "history": [{"text": prompt, "video_b64": video_b64}],
                    "required_rewards": ["videoalign"],
                }
                for _ in range(bs)
            ]
        }

        latencies = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            _post_score(url, payload)
            latencies.append(time.perf_counter() - t0)

        avg = sum(latencies) / len(latencies)
        per_vid = avg / bs
        throughput = bs / avg
        runs_str = ", ".join(f"{l:.2f}" for l in latencies)
        print(f"  {bs:>5d}  {avg:>8.2f}  {per_vid:>10.2f}  {throughput:>8.1f}/s  [{runs_str}]")

    print()


# Entry point

def main() -> int:
    ap = argparse.ArgumentParser(
        description="VideoAlign RewardService test script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--url", default="http://localhost:8080", help="Service URL")
    ap.add_argument("--video", type=Path, nargs="+", default=None, help="Test video path")
    ap.add_argument("--no-bench", action="store_true", help="Skip performance benchmark")
    ap.add_argument("--bench-full", action="store_true", help="Full benchmark (batch 1-6)")
    args = ap.parse_args()

    url = args.url.rstrip("/")

    # Find videos
    if args.video:
        videos = list(args.video)
        missing = [v for v in videos if not v.is_file()]
        if missing:
            print(f"[ERROR] Video files do not exist: {missing}", file=sys.stderr)
            return 1
    else:
        videos = _find_videos()
        if not videos:
            print("[ERROR] No test videos found. Specify them with --video.", file=sys.stderr)
            return 1
        print(f"[INFO] Auto-detected {len(videos)} test videos: {videos[0].parent}/\n")

    # Functional tests
    passed = 0
    failed = 0

    if test_health(url):
        passed += 1
    else:
        print("[FATAL] Service is unhealthy. Aborting tests.", file=sys.stderr)
        return 1

    test_rewards_endpoint(url)

    for test_fn, test_args in [
        (test_single_video_b64, (url, videos[0])),
        (test_single_video_path, (url, videos[0])),
        (test_batch, (url, videos)),
        (test_consistency, (url, videos[0])),
    ]:
        try:
            if test_fn(*test_args):
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  [ERROR] {e}")
            failed += 1

    print(f"== Functional Test Results: {passed} passed, {failed} failed ==\n")

    # Benchmark
    if not args.no_bench:
        if args.bench_full:
            batch_sizes = [1, 2, 3, 4, 6]
        else:
            batch_sizes = [1, 3, 6]
        benchmark(url, videos[0], batch_sizes)

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
