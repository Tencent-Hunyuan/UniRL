#!/usr/bin/env python
"""Single-GPU CUDA MPS benchmark comparing RolloutPool v1/v2/v3 routing policies."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from unirl.rollout.manager.dispatch import RolloutPool
from unirl.types.primitives import Texts
from unirl.types.sample import Part, Sample


class _FuturePending:
    def __init__(self, future: concurrent.futures.Future[Dict[str, Any]]) -> None:
        self._future = future

    def ready(self) -> bool:
        return self._future.done()

    def result(self) -> Dict[str, Any]:
        return self._future.result()

    def discard_on_completion(self) -> None:
        self._future.cancel()


def _http_post_json(url: str, payload: Dict[str, Any], timeout: float = 120.0) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _wait_for_health(
    ports: List[int], procs: List[subprocess.Popen[bytes]], log_paths: List[Path], timeout_s: float = 180.0
) -> None:
    deadline = time.monotonic() + timeout_s
    pending = set(ports)
    while pending and time.monotonic() < deadline:
        for port, proc, log_path in zip(ports, procs, log_paths):
            if port not in pending:
                continue
            if proc.poll() is not None:
                tail = log_path.read_text(errors="replace")[-1500:] if log_path.exists() else ""
                raise RuntimeError(f"SGLang server on port {port} exited with code {proc.returncode}:\n{tail}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2.0) as resp:
                    if resp.status == 200:
                        pending.remove(port)
            except Exception:
                pass
        if pending:
            time.sleep(1.0)
    if pending:
        raise TimeoutError(f"SGLang servers on ports {sorted(pending)} did not become healthy within {timeout_s}s")


def _flush_caches(ports: List[int]) -> None:
    for port in ports:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/flush_cache", data=b"", method="POST")
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            if resp.status != 200:
                raise RuntimeError(f"flush_cache failed on port {port}: status={resp.status}")


def _build_tasks(num_prompts: int, group_size: int, prefix_repeats: int, *, heterogeneous: bool = True) -> List[Sample]:
    base_unit = (
        "You are an analytical assistant solving multi-step reasoning tasks with tool calls. "
        "Carefully read the retrieved context, verify all intermediate numbers, and avoid guessing. "
    )
    tasks: List[Sample] = []
    for p in range(num_prompts):
        repeats = prefix_repeats if (not heterogeneous or p % 2 == 0) else max(4, prefix_repeats // 4)
        filler = base_unit * repeats
        prompt = (
            f"System Context [Task {p}]:\n{filler}\n"
            f"Question {p}: Compute (17 * {p + 3} + 41) mod 97 and explain each step briefly."
        )
        for _ in range(group_size):
            root = Part.input([f"prompt_{p}"], primitives={"text": Texts(texts=[prompt])})
            tasks.append(Sample.request(root))
    return tasks


def _run_policy(
    *,
    label: str,
    pool_kwargs: Dict[str, Any],
    ports: List[int],
    worker_node_ids: List[int],
    driver_node_id: int,
    intra_node_delay_ms: float,
    cross_node_delay_ms: float,
    straggler_worker: Optional[int],
    straggler_delay_ms: float,
    tasks: List[Sample],
    per_worker_inflight: int,
    max_new_tokens: int,
) -> Dict[str, Any]:
    _flush_caches(ports)
    rng = random.Random(42)
    executors = [
        concurrent.futures.ThreadPoolExecutor(max_workers=per_worker_inflight, thread_name_prefix=f"sgl-{i}")
        for i in range(len(ports))
    ]

    def make_launcher(worker_idx: int, port: int):
        node_id = worker_node_ids[worker_idx]
        net_delay_s = (intra_node_delay_ms if node_id == driver_node_id else cross_node_delay_ms) / 1000.0

        def launch(sample: Sample) -> _FuturePending:
            prompt = sample.parts[0].primitives["text"].texts[0]

            def _call() -> Dict[str, Any]:
                t0 = time.perf_counter()
                if net_delay_s > 0:
                    time.sleep(net_delay_s)
                if straggler_worker is not None and worker_idx == straggler_worker and straggler_delay_ms > 0:
                    time.sleep((straggler_delay_ms / 1000.0) * (0.8 + 0.4 * rng.random()))
                out = _http_post_json(
                    f"http://127.0.0.1:{port}/generate",
                    {
                        "text": prompt,
                        "sampling_params": {
                            "temperature": 0.7,
                            "top_p": 0.95,
                            "max_new_tokens": max_new_tokens,
                        },
                    },
                )
                dt = time.perf_counter() - t0
                meta = out.get("meta_info", {})
                return {
                    "latency_s": dt,
                    "prompt_tokens": int(meta.get("prompt_tokens", 0)),
                    "cached_tokens": int(meta.get("cached_tokens", 0)),
                    "completion_tokens": int(meta.get("completion_tokens", 0)),
                }

            return _FuturePending(executors[worker_idx].submit(_call))

        return launch

    launchers = [make_launcher(i, port) for i, port in enumerate(ports)]
    pool = RolloutPool(launchers, [per_worker_inflight] * len(ports), **pool_kwargs)
    try:
        t0 = time.perf_counter()
        completed = pool.run_to_completion(tasks)
        wall_s = time.perf_counter() - t0
        dispatch_stats = pool.dispatch_stats(reset=True)
    finally:
        pool.close()
        for ex in executors:
            ex.shutdown(wait=True)

    results = [unit.pending.result() for unit in completed]
    latencies = sorted(r["latency_s"] for r in results)
    total_prompt = sum(r["prompt_tokens"] for r in results)
    total_cached = sum(r["cached_tokens"] for r in results)
    total_completion = sum(r["completion_tokens"] for r in results)
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))]
    return {
        "label": label,
        "wall_s": round(wall_s, 3),
        "throughput_tasks_per_s": round(len(results) / wall_s, 3),
        "p50_latency_s": round(p50, 3),
        "p95_latency_s": round(p95, 3),
        "prompt_tokens": total_prompt,
        "cached_tokens": total_cached,
        "completion_tokens": total_completion,
        "prefix_cache_hit_rate": round(total_cached / total_prompt, 4) if total_prompt else 0.0,
        "dispatch_stats": {k: round(v, 4) for k, v in dispatch_stats.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--worker-node-ids", default="0,0,1,1")
    parser.add_argument("--driver-node-id", type=int, default=0)
    parser.add_argument("--intra-node-delay-ms", type=float, default=5.0)
    parser.add_argument("--cross-node-delay-ms", type=float, default=35.0)
    parser.add_argument("--straggler-worker", type=int, default=2)
    parser.add_argument("--straggler-delay-ms", type=float, default=150.0)
    parser.add_argument("--base-port", type=int, default=30001)
    parser.add_argument("--mem-fraction-static", type=float, default=0.21)
    parser.add_argument("--per-worker-inflight", type=int, default=8)
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--prefix-repeats", type=int, default=40)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--leader-stagger-ms", type=float, default=150.0)
    parser.add_argument("--attention-backend", default="triton")
    parser.add_argument("--max-total-tokens", type=int, default=8192)
    parser.add_argument("--use-mps", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    worker_node_ids = [int(x.strip()) for x in args.worker_node_ids.split(",")]
    if len(worker_node_ids) != args.num_workers:
        raise ValueError(f"--worker-node-ids length ({len(worker_node_ids)}) != --num-workers ({args.num_workers})")

    mps_dir = None
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    if args.use_mps:
        mps_dir = tempfile.mkdtemp(prefix="unirl_mps_")
        pipe_dir = str(Path(mps_dir) / "pipe")
        log_dir = str(Path(mps_dir) / "log")
        os.makedirs(pipe_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        env["CUDA_MPS_PIPE_DIRECTORY"] = pipe_dir
        env["CUDA_MPS_LOG_DIRECTORY"] = log_dir
        env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(max(1, 100 // args.num_workers))
        subprocess.run(["nvidia-cuda-mps-control", "-d"], env=env, check=True)

    model_path = args.model
    if not os.path.exists(model_path):
        try:
            from huggingface_hub import snapshot_download

            model_path = snapshot_download(model_path)
        except Exception:
            pass

    def _gpu_mem_mb() -> tuple[float, float]:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"],
            text=True,
        ).strip().splitlines()[0]
        total_s, free_s = out.split(",")
        return float(total_s), float(free_s)

    ports = [args.base_port + i for i in range(args.num_workers)]
    run_dir = Path(mps_dir) if mps_dir is not None else Path(tempfile.mkdtemp(prefix="unirl_sgl_"))
    log_paths: List[Path] = []
    log_files = []
    procs: List[subprocess.Popen[bytes]] = []
    try:
        for i, port in enumerate(ports):
            total_mb, free_mb = _gpu_mem_mb()
            worker_frac = min(0.85, max(args.mem_fraction_static, (total_mb * args.mem_fraction_static) / max(free_mb, 1.0)))
            log_path = run_dir / f"worker_{i}_{port}.log"
            log_paths.append(log_path)
            lf = open(log_path, "w")
            log_files.append(lf)
            cmd = [
                sys.executable,
                "-m",
                "sglang.launch_server",
                "--model-path",
                model_path,
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--mem-fraction-static",
                f"{worker_frac:.4f}",
                "--dtype",
                "bfloat16",
                "--attention-backend",
                args.attention_backend,
                "--max-total-tokens",
                str(args.max_total_tokens),
                "--disable-cuda-graph",
                "--log-level",
                "warning",
            ]
            procs.append(subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT))
            _wait_for_health([port], [procs[-1]], [log_path])
        tasks = _build_tasks(args.num_prompts, args.group_size, args.prefix_repeats, heterogeneous=True)
        # Warmup one short request per worker so JIT/kernels are initialized before timing.
        _run_policy(
            label="warmup",
            pool_kwargs={"affinity_policy": "least_loaded"},
            ports=ports,
            worker_node_ids=worker_node_ids,
            driver_node_id=args.driver_node_id,
            intra_node_delay_ms=0.0,
            cross_node_delay_ms=0.0,
            straggler_worker=None,
            straggler_delay_ms=0.0,
            tasks=_build_tasks(1, args.num_workers, 2, heterogeneous=False),
            per_worker_inflight=1,
            max_new_tokens=8,
        )

        experiments = [
            ("v1_least_loaded", {"affinity_policy": "least_loaded", "worker_node_ids": worker_node_ids}),
            ("v2_group_packing", {"affinity_policy": "group_packing", "worker_node_ids": worker_node_ids}),
            (
                "v3_group_packing_stagger",
                {
                    "affinity_policy": "group_packing",
                    "stagger_leader_prefill": True,
                    "leader_stagger_ms": args.leader_stagger_ms,
                    "worker_node_ids": worker_node_ids,
                },
            ),
            (
                "v3_cache_aware_scored",
                {
                    "affinity_policy": "cache_aware_scored",
                    "stagger_leader_prefill": True,
                    "leader_stagger_ms": args.leader_stagger_ms,
                    "balance_rel_threshold": 1.5,
                    "max_inflight_tokens": 16384,
                    "worker_node_ids": worker_node_ids,
                },
            ),
        ]
        reports = [
            _run_policy(
                label=label,
                pool_kwargs=kwargs,
                ports=ports,
                worker_node_ids=worker_node_ids,
                driver_node_id=args.driver_node_id,
                intra_node_delay_ms=args.intra_node_delay_ms,
                cross_node_delay_ms=args.cross_node_delay_ms,
                straggler_worker=args.straggler_worker,
                straggler_delay_ms=args.straggler_delay_ms,
                tasks=tasks,
                per_worker_inflight=args.per_worker_inflight,
                max_new_tokens=args.max_new_tokens,
            )
            for label, kwargs in experiments
        ]
        print(json.dumps({"config": vars(args), "results": reports}, indent=2))
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
        for proc in procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        for lf in log_files:
            lf.close()
        if args.use_mps and mps_dir is not None:
            subprocess.run(["nvidia-cuda-mps-control"], input=b"quit\n", env=env, check=False)


if __name__ == "__main__":
    main()
