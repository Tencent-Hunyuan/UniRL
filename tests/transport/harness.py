"""Shared helpers for the GPU / multi-GPU transport tests (NOT a ``test_`` module).

Provides a minimal :class:`Remote` role plus driver-side drivers so a test can:
  * put a deterministic tensor onto a CHOSEN device (``make_ref``), and
  * force the resolve/borrow path off a CHOSEN device (``borrow_sum``),

using the raw ``Worker.call`` RPC rather than the SPMD ``Handle`` path — the
latter would split/merge across ranks, whereas these tests need per-device
control (put on device A, localize to device B, verify on B).

``leak_loop`` is the sustained-churn driver behind the no-leak assertion; it
leans on ``TensorWorker.get_store_size()`` / ``memory_allocated()`` (gpu_store).
"""

from __future__ import annotations

import gc
import os
import time
from typing import Optional

import ray
import torch

from unirl.distributed.group.dispatch import distributed
from unirl.distributed.group.remote import RankInfo, Remote
from unirl.distributed.tensor.backend.gpu_store.transport import GPUStoreTransport


class TProbe(Remote):
    """Per-Worker probe. Methods are ``@distributed`` so the same role also works
    through a ``Handle`` (used by the grad/backward test); the raw-``call`` tests
    invoke them directly, which ignores the dispatch metadata."""

    @distributed
    def make(self, n: int, w: int, base: float):
        # element[i, j] = base + i*w + j  → every row uniquely identifiable, so a
        # checksum still pins exact rows after select/slice/cross-device move.
        return torch.arange(n * w, dtype=torch.float32, device=self.device).reshape(n, w) + float(base)

    @distributed
    def checksum(self, t):
        # Plain list (not a tensor) → Worker.call does not re-pack it. The
        # TensorRef arg forces transport.get_batch → _resolve_handles → borrow.
        return [int(t.shape[0]), int(t.shape[1]), float(t.to(torch.float64).sum().item())]

    @distributed
    def scale(self, t):
        # Differentiable elementwise op for the grad/backward test (Handle path).
        return t * 3.0

    @distributed
    def make_ones(self, n: int, w: int):
        # Per-rank ones block; used to seed an output gradient in the grad test.
        return torch.ones(n, w, dtype=torch.float32, device=self.device)


def expected(n: int, w: int, base: float) -> torch.Tensor:
    """CPU reference tensor matching ``TProbe.make`` for assertions."""
    return torch.arange(n * w, dtype=torch.float32).reshape(n, w) + float(base)


def register_probe(pool, role: str = "tprobe") -> str:
    """Register ``TProbe`` on every slot0 Worker of *pool*. Returns the role name."""
    n = pool.num_devices
    ray.get(
        [
            pool.slot0_worker(d).add_remote.remote(role, TProbe, RankInfo(rank=d, world_size=n), {}, {"RANK": str(d)})
            for d in range(n)
        ]
    )
    return role


def make_ref(pool, role: str, device_id: int, n: int, w: int, base: float):
    """Put a deterministic (n, w) tensor on ``device_id``; return a bound TensorRef."""
    worker = pool.slot0_worker(device_id)
    ref = ray.get(worker.call.remote(role, "make", (n, w, base), {}))
    for s in ref.spans:
        s.handle.rebind(worker)  # bypassing Handle._rebind_tree → bind for driver-side .local()
    return ref


def borrow_sum(pool, role: str, device_id: int, ref):
    """Resolve *ref* on ``device_id`` via the borrow path; return [rows, width, sum]."""
    return ray.get(pool.slot0_worker(device_id).call.remote(role, "checksum", (ref,), {}))


def store_size(pool, device_id: int) -> int:
    return ray.get(pool._tw_by_device[device_id].get_store_size.remote())


def mem(pool, device_id: int) -> int:
    return ray.get(pool._tw_by_device[device_id].memory_allocated.remote())


def drain_store(pool, device_id: int, timeout_s: float = 12.0) -> int:
    """Poll until the device's TensorWorker store stops shrinking — async decref
    RPCs + the lazily-thresholded ipc_collect need a moment to settle."""
    deadline = time.time() + timeout_s
    prev = -1
    while time.time() < deadline:
        ray.get(pool._tw_by_device[device_id].empty_cache.remote())
        cur = store_size(pool, device_id)
        if cur == prev:
            return cur
        prev = cur
        time.sleep(0.2)
    return store_size(pool, device_id)


def leak_loop(pool, role: str, iters: Optional[int] = None) -> dict:
    """Sustained churn for the no-leak assertion (gpu_store pool).

    Each iter: make → ``select_ranges`` (partial spans) → borrow → drop. On a
    >=2-device pool it also cross-device ``localize``s the partial spans first,
    so the NCCL move + its received-handle GC are churned too. On a 1-device pool
    it exercises the put/borrow/decref path alone.

    Returns before/after TensorWorker store sizes + memory for the source and
    destination device, plus ``bad`` (incorrect per-iter checksums). A leak shows
    as store/mem failing to return to baseline.
    """
    if iters is None:
        iters = int(os.environ.get("UNIRL_LEAK_ITERS", "300"))
    src = 0
    dst = 1 if pool.num_devices >= 2 else 0
    gc.collect()
    ray.get(pool._tw_by_device[src].empty_cache.remote())
    ray.get(pool._tw_by_device[dst].empty_cache.remote())
    s0b, s1b = store_size(pool, src), store_size(pool, dst)
    m0b, m1b = mem(pool, src), mem(pool, dst)
    bad = 0
    for i in range(iters):
        r = make_ref(pool, role, src, 8, 16, i)
        seg = r.select_ranges([(1, 4), (6, 8)])  # 5 rows, 2 partial spans
        if dst != src:
            target = GPUStoreTransport.localize([((seg,), {})], pool, [dst], [f"dw{dst}"])[0][0][0]
        else:
            target = seg
        n, w, csum = borrow_sum(pool, role, dst, target)
        exp = float(torch.cat([expected(8, 16, i)[1:4], expected(8, 16, i)[6:8]]).to(torch.float64).sum())
        if not (n == 5 and abs(csum - exp) < 1e-2):
            bad += 1
        del r, seg, target
        if i % 50 == 0:
            gc.collect()
    gc.collect()
    s0a = drain_store(pool, src)
    s1a = drain_store(pool, dst)
    m0a, m1a = mem(pool, src), mem(pool, dst)
    return dict(s0b=s0b, s0a=s0a, s1b=s1b, s1a=s1a, m0b=m0b, m0a=m0a, m1b=m1b, m1a=m1a, bad=bad, iters=iters)
