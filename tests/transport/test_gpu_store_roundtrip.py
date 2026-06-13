"""gpu_store IPC put / batch_borrow / get + refcount on a single real GPU.

Drives the real per-GPU TensorWorker through ``TProbe`` (the SPMD Handle path
would split/merge; these need per-device control). Marker: gpu.
"""

from __future__ import annotations

import gc

import pytest
import torch

pytestmark = pytest.mark.gpu


def test_put_then_borrow_roundtrip(gpu1_probe):
    p = gpu1_probe
    ref = p.h.make_ref(p.pool, p.role, 0, 6, 4, 100)
    assert ref.batch_size == 6 and ref.sizes == [6]
    # driver-side materialize (get_cpu path)
    assert torch.equal(ref.local(), p.h.expected(6, 4, 100))
    # worker-side resolve (batch_borrow IPC view path)
    n, w, s = p.h.borrow_sum(p.pool, p.role, 0, ref)
    assert n == 6 and w == 4 and abs(s - float(p.h.expected(6, 4, 100).sum())) < 1e-3


def test_select_segments_borrow(gpu1_probe):
    p = gpu1_probe
    ref = p.h.make_ref(p.pool, p.role, 0, 8, 4, 200)
    seg = ref.select_segments([(5, 8), (0, 2)])  # rows 5,6,7,0,1 — partial spans
    exp = p.h.expected(8, 4, 200)[[5, 6, 7, 0, 1]]
    assert torch.equal(seg.local(), exp)
    n, w, s = p.h.borrow_sum(p.pool, p.role, 0, seg)
    assert n == 5 and abs(s - float(exp.sum())) < 1e-3


def test_refcount_decref_frees(gpu1_probe):
    p = gpu1_probe
    base = p.h.store_size(p.pool, 0)
    ref = p.h.make_ref(p.pool, p.role, 0, 4, 4, 300)
    assert p.h.store_size(p.pool, 0) == base + 1
    del ref
    gc.collect()
    assert p.h.drain_store(p.pool, 0) == base  # decref finalizer freed it
