"""Cross-device ``localize`` MOVE on real GPUs — the NCCL hop the unit tests stub.

Parametrized over the two worker-local backends (gpu_store, colocate) via
``multigpu_probe``. Covers FULL-block + PARTIAL-span moves (the send ships only
``[start:stop)`` rows, the recv allocates the sliced shape), the same-device
no-op, on-wire dedup, and grad / ``_packed_cu_seqlens`` preservation when a
sibling moves but this ref stays. Marker: multigpu.
"""

from __future__ import annotations

import pytest
import torch

from unirl.distributed.tensor.backend.colocate_store.transport import ColocateStoreTransport
from unirl.distributed.tensor.backend.gpu_store.transport import GPUStoreTransport

pytestmark = pytest.mark.multigpu


def _cls(backend: str):
    return GPUStoreTransport if backend == "gpu_store" else ColocateStoreTransport


def test_move_full_block(multigpu_probe):
    p = multigpu_probe
    ref = p.h.make_ref(p.pool, p.role, 0, 6, 4, 100)  # produced on dw0
    moved = _cls(p.backend).localize([((ref,), {})], p.pool, [1], ["dw1"])[0][0][0]
    assert torch.equal(moved.materialize(), p.h.expected(6, 4, 100))
    n, w, s = p.h.borrow_sum(p.pool, p.role, 1, moved)  # resolvable on dw1
    assert n == 6 and abs(s - float(p.h.expected(6, 4, 100).sum())) < 1e-3


def test_move_partial_spans(multigpu_probe):
    p = multigpu_probe
    ref = p.h.make_ref(p.pool, p.role, 0, 6, 4, 100)
    part = ref.select_ranges([(2, 5), (0, 1)])  # rows 2,3,4,0 — two partial spans
    exp = p.h.expected(6, 4, 100)[[2, 3, 4, 0]]
    moved = _cls(p.backend).localize([((part,), {})], p.pool, [1], ["dw1"])[0][0][0]
    assert moved.batch_size == 4  # only the selected rows shipped
    assert torch.equal(moved.materialize(), exp)
    n, w, s = p.h.borrow_sum(p.pool, p.role, 1, moved)
    assert n == 4 and abs(s - float(exp.sum())) < 1e-3


def test_same_device_noop(multigpu_probe):
    p = multigpu_probe
    ref = p.h.make_ref(p.pool, p.role, 0, 4, 4, 0)
    same = _cls(p.backend).localize([((ref,), {})], p.pool, [0], ["dw0"])
    assert same[0][0][0] is ref  # already local → same object, no NCCL


def test_dedup_shared_received_handle(multigpu_probe):
    p = multigpu_probe
    ref = p.h.make_ref(p.pool, p.role, 0, 4, 4, 500)
    out = _cls(p.backend).localize([((ref,), {}), ((ref,), {})], p.pool, [1, 1], ["dw1", "dw1"])
    assert out[0][0][0].spans[0].handle is out[1][0][0].spans[0].handle  # one transfer, shared
    assert torch.equal(out[0][0][0].materialize(), p.h.expected(4, 4, 500))


def test_dedup_one_transfer_store(multigpu_probe):
    p = multigpu_probe
    if p.backend != "gpu_store":
        pytest.skip("TensorWorker store_size introspection is gpu_store-only")
    ref = p.h.make_ref(p.pool, p.role, 0, 4, 4, 800)
    before = p.h.store_size(p.pool, 1)
    GPUStoreTransport.localize([((ref,), {}), ((ref,), {})], p.pool, [1, 1], ["dw1", "dw1"])
    assert p.h.store_size(p.pool, 1) == before + 1  # dedup → exactly ONE received tensor


def test_grad_preserved_when_sibling_moves(multigpu_probe):
    p = multigpu_probe
    local1 = p.h.make_ref(p.pool, p.role, 1, 2, 4, 600)  # on dst dw1 → stays
    foreign0 = p.h.make_ref(p.pool, p.role, 0, 3, 4, 700)  # on dw0 → moves
    local1.grad = p.h.make_ref(p.pool, p.role, 1, 2, 4, 0)
    local1.retain_grad()
    cu = torch.tensor([0, 1, 2])
    object.__setattr__(local1, "_packed_cu_seqlens", cu)
    out = _cls(p.backend).localize([((foreign0, local1), {})], p.pool, [1], ["dw1"])
    g_for, g_loc = out[0][0][0], out[0][0][1]
    assert g_loc is local1  # same object back (REPLACE no-op for the local sibling)
    assert g_loc.grad is local1.grad and g_loc.retain_grad_flag is True
    assert getattr(g_loc, "_packed_cu_seqlens", None) is cu
    assert torch.equal(g_for.materialize(), p.h.expected(3, 4, 700))
