"""Lifecycle GC + no-leak for the refcount-driven gpu_store borrow GC.

Marker: gpu (the 300-iter churn is additionally ``slow``). The churn runs
make -> select_segments -> [localize if >=2 GPU] -> borrow -> drop, repeated, and
asserts the per-GPU TensorWorker store size and allocated memory return to
baseline — i.e. neither the put handles nor the borrowed IPC views leak.
"""

from __future__ import annotations

import copy
import gc

import pytest

pytestmark = pytest.mark.gpu


def test_copy_increfs_keeps_alive_then_frees(gpu1_probe):
    p = gpu1_probe
    base = p.h.store_size(p.pool, 0)
    ref = p.h.make_ref(p.pool, p.role, 0, 4, 4, 0)
    clone = copy.copy(ref.spans[0].handle)  # __copy__ on a bound CUDA handle increfs
    assert p.h.store_size(p.pool, 0) == base + 1
    del ref
    gc.collect()
    # ref's handle dropped (refcount 2->1); the clone still holds it → not freed.
    assert p.h.store_size(p.pool, 0) == base + 1
    del clone
    gc.collect()
    assert p.h.drain_store(p.pool, 0) == base  # last holder gone → freed


@pytest.mark.slow
def test_no_leak_under_churn(gpu1_probe):
    p = gpu1_probe
    r = p.h.leak_loop(p.pool, p.role)
    assert r["bad"] == 0, f"{r['bad']}/{r['iters']} bad checksums"
    assert r["s0a"] <= r["s0b"] + 2, f"dw0 store {r['s0b']}->{r['s0a']} over {r['iters']} iters"
    assert r["s1a"] <= r["s1b"] + 2, f"dw1 store {r['s1b']}->{r['s1a']} over {r['iters']} iters"
    assert r["m0a"] - r["m0b"] <= 16 * 1024 * 1024, f"dw0 mem grew {(r['m0a'] - r['m0b']) / 1e6:.1f} MB"
    assert r["m1a"] - r["m1b"] <= 16 * 1024 * 1024, f"dw1 mem grew {(r['m1a'] - r['m1b']) / 1e6:.1f} MB"
