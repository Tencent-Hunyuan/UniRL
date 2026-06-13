"""Unit tests for TensorRef span views (select/slice without hydration).

Selection emits :class:`TensorSpan` spans — contiguous row-windows over the
parent handles — instead of moving data. Pure-CPU: handles are faked with a
minimal protocol (.local/.shape/.dtype/.device), no transport backend required
(materialize falls back to per-span fetch when backend is None).
"""

from dataclasses import dataclass
from typing import Optional

import torch

from unirl.distributed.tensor.backend.gpu_store.transport import GPUStoreTransport
from unirl.distributed.tensor.batch import Batch, shared_field
from unirl.distributed.tensor import (
    TensorRef,
    TensorSpan,
    WorkerLocalTransport,
    cat_rows,
    map_tree,
)


class _FakeHandle:
    def __init__(self, t: torch.Tensor, source_id: str = "w0", store_key: str = "k0", object_ref=None):
        self.t = t
        self.shape = t.shape
        self.dtype = t.dtype
        self.device = t.device
        # Worker-local identity fields read by localize's _move_key (unused by the view tests).
        self.source_id = source_id
        self.store_key = store_key
        self.object_ref = object_ref

    def local(self) -> torch.Tensor:
        return self.t


def _meta(*tensors: torch.Tensor) -> TensorRef:
    return TensorRef(
        spans=[TensorSpan(_FakeHandle(t), 0, int(t.shape[0])) for t in tensors],
        shape=(sum(int(t.shape[0]) for t in tensors), *tensors[0].shape[1:]),
        dtype=tensors[0].dtype,
        device="cpu",
    )


def test_select_permutation_with_ragged_pad():
    t0 = torch.arange(12).reshape(3, 4).float()
    t1 = torch.arange(100, 112).reshape(2, 6).float()
    t2 = torch.arange(200, 210).reshape(2, 5).float()
    tm = _meta(t0, t1, t2)
    perm = [5, 0, 3, 6, 2, 1, 4]
    v = tm.select(perm)
    assert v.batch_size == 7
    assert any(isinstance(r, TensorSpan) for r in v.spans)
    out = v.materialize(backend=None)
    assert out.shape == (7, 6)  # ragged spans right-padded to the max width
    assert torch.equal(out[0, :5], t2[0])
    assert torch.equal(out[1, :4], t0[0])
    assert torch.all(out[1, 4:] == 0)


def test_view_slice_matches_materialized_rows():
    t0 = torch.arange(12).reshape(3, 4).float()
    t1 = torch.arange(100, 108).reshape(2, 4).float()
    v = _meta(t0, t1).select([4, 0, 2, 1])
    full = v.materialize(backend=None)
    half = v.slice(1, 3)
    assert torch.equal(half.materialize(backend=None), full[1:3])


def test_aligned_slice_passes_spans_through():
    # A span-boundary-aligned slice is the structural inverse of concat:
    # the original span object (and its handle) comes back untouched.
    t0 = torch.arange(12).reshape(3, 4).float()
    t1 = torch.arange(100, 108).reshape(2, 4).float()
    tm = _meta(t0, t1)
    head = tm.slice(0, 3)
    assert head.spans == [tm.spans[0]] and head.sizes == [3]
    assert head.spans[0] is tm.spans[0]
    assert head.spans[0].handle is tm.spans[0].handle


def test_misaligned_slice_wraps_boundary_spans():
    t0 = torch.arange(12).reshape(3, 4).float()
    t1 = torch.arange(100, 108).reshape(2, 4).float()
    tm = _meta(t0, t1)
    mid = tm.slice(1, 4)  # crosses the span boundary off-alignment
    assert mid.batch_size == 3
    assert isinstance(mid.spans[0], TensorSpan) and isinstance(mid.spans[1], TensorSpan)
    assert torch.equal(mid.materialize(backend=None), torch.cat([t0[1:], t1[:1]]))


def test_packed_segment_view():
    p0 = torch.arange(10).float()
    p1 = torch.arange(100, 106).float()
    pm = _meta(p0, p1)
    pv = pm.select_ranges([(12, 16), (0, 3)])  # out-of-order token ranges
    assert pv.batch_size == 7
    assert torch.equal(pv.materialize(backend=None), torch.cat([p1[2:6], p0[0:3]]))


def test_nested_views_flatten():
    # A span of a span flattens to a single TensorSpan over the handle —
    # repeated selection never builds an indirection chain.
    t0 = torch.arange(40).reshape(8, 5).float()
    v1 = _meta(t0).select([3, 4, 5, 6])  # rows 3..6 (one coalesced span)
    v2 = v1.select([1, 2])  # rows 4..5 of the original
    assert all(isinstance(r, TensorSpan) and isinstance(r.handle, _FakeHandle) for r in v2.spans)
    assert torch.equal(v2.materialize(backend=None), t0[4:6])


def test_with_spans_preserves_sizes():
    t0 = torch.arange(8).reshape(2, 4).float()
    v = _meta(t0).select([1, 0])
    v2 = v.with_spans(list(v.spans))
    assert v2.sizes == v.sizes and v2.batch_size == v.batch_size


def test_empty_selection():
    p0 = torch.arange(10).float()
    e = _meta(p0).select_ranges([])
    assert e.batch_size == 0
    assert e.materialize(backend=None).numel() == 0


def test_span_shape_and_local():
    t0 = torch.arange(12).reshape(3, 4).float()
    h = _FakeHandle(t0)
    v = TensorSpan(h, 1, 3)
    assert v.shape == (2, 4) and v.dtype == t0.dtype
    assert torch.equal(v.local(), t0[1:3])


def test_cat_rows_ragged_pad_contract():
    a = torch.ones(2, 3)
    b = torch.full((1, 5), 2.0)
    out = cat_rows([a, b])
    assert out.shape == (3, 5)
    assert torch.all(out[:2, 3:] == 0)  # right-pad with zeros


# ── localize: find / move / replace ──────────────────────────────────────────
# FIND (_move_key) and REPLACE (_replace_leaf) are pure — they depend only on a
# pool's device_id_of + the backend _is_local predicate, so they unit-test with no
# Ray/NCCL. MOVE (the NCCL hop) stays integration-only (real multi-GPU).


class _FakePool:
    """Minimal pool exposing only what _move_key needs: worker_id → device_id."""

    def __init__(self, mapping: dict):
        self._m = mapping

    def device_id_of(self, worker_id: str) -> int:
        return self._m[worker_id]


def _span(t: torch.Tensor, *, source_id="w0", store_key="k0", start=0, stop=None, object_ref=None) -> TensorSpan:
    h = _FakeHandle(t, source_id=source_id, store_key=store_key, object_ref=object_ref)
    return TensorSpan(h, start, int(t.shape[0]) if stop is None else stop)


def _ref(*spans: TensorSpan) -> TensorRef:
    return TensorRef(
        spans=list(spans),
        shape=(sum(s.stop - s.start for s in spans), *spans[0].shape[1:]),
        dtype=spans[0].dtype,
        device="cpu",
    )


@dataclass
class _Holder(Batch):
    """Non-TensorRef Batch holding a nested TensorRef — exercises map_tree's Batch branch."""

    ref: Optional[TensorRef] = shared_field(default=None)


# FIND — _move_key classification

def test_move_key_local_by_source():
    # base _is_local: a ref produced by the dst worker is already resolvable → None
    pool = _FakePool({"w0": 0})
    span = _span(torch.zeros(2, 4), source_id="w0")
    assert WorkerLocalTransport._move_key(span, ("w0", 0), pool) is None


def test_move_key_local_by_device_gpu_override():
    # gpu _is_local also accepts same physical device (shared per-GPU TensorWorker)
    pool = _FakePool({"w1_s0": 1, "w1_s1": 1})
    span = _span(torch.zeros(2, 4), source_id="w1_s0")
    # produced by a different worker, but on the dst's device → local for gpu
    assert GPUStoreTransport._move_key(span, ("w1_s1", 1), pool) is None
    # base backend does NOT accept same-device → foreign
    assert WorkerLocalTransport._move_key(span, ("w1_s1", 1), pool) == (1, 1, "k0", 0, 2)


def test_move_key_object_ref_resolvable_anywhere():
    # a CPU/plasma handle (object_ref set) resolves anywhere, even from a foreign worker
    pool = _FakePool({"w9": 9})
    span = _span(torch.zeros(2, 4), source_id="w9", object_ref=object())
    assert WorkerLocalTransport._move_key(span, ("w0", 0), pool) is None


def test_move_key_foreign_value_key():
    pool = _FakePool({"w0": 0, "w1": 1})
    span = _span(torch.zeros(5, 4), source_id="w1", store_key="k7", start=2, stop=5)
    assert WorkerLocalTransport._move_key(span, ("w0", 0), pool) == (1, 0, "k7", 2, 5)


def test_move_key_dedup_identical_slice():
    # two distinct span objects over the same foreign slice → equal keys → setdefault keeps one
    pool = _FakePool({"w0": 0, "w1": 1})
    a = _span(torch.zeros(4, 4), source_id="w1", store_key="kb")
    b = _span(torch.zeros(4, 4), source_id="w1", store_key="kb")
    dst = ("w0", 0)
    ka = WorkerLocalTransport._move_key(a, dst, pool)
    kb = WorkerLocalTransport._move_key(b, dst, pool)
    assert ka == kb and a is not b
    to_move: dict = {}
    to_move.setdefault(ka, a)
    to_move.setdefault(kb, b)
    assert len(to_move) == 1 and to_move[ka] is a


def test_move_key_same_device_two_workers_dedup():
    # the key carries dst_device, not dst_worker → two workers on one device dedup
    pool = _FakePool({"w3": 3, "w5_s0": 5, "w5_s1": 5})
    a = _span(torch.zeros(4, 4), source_id="w3", store_key="k")
    b = _span(torch.zeros(4, 4), source_id="w3", store_key="k")
    ka = GPUStoreTransport._move_key(a, ("w5_s0", 5), pool)
    kb = GPUStoreTransport._move_key(b, ("w5_s1", 5), pool)
    assert ka == kb == (3, 5, "k", 0, 4)


# REPLACE — _replace_leaf substitution

def test_replace_substitutes_foreign_keeps_local():
    pool = _FakePool({"w0": 0, "w1": 1})
    local = _span(torch.zeros(2, 4), source_id="w0", store_key="ka")
    foreign = _span(torch.ones(3, 4), source_id="w1", store_key="kb")
    ref = _ref(local, foreign)
    key = WorkerLocalTransport._move_key(foreign, ("w0", 0), pool)
    moved_span = _span(torch.full((3, 4), 7.0), source_id="w0", store_key="kb_recv")
    out = WorkerLocalTransport._replace_leaf({key: moved_span}, ("w0", 0), pool)(ref)
    assert out is not ref
    assert out.spans[0] is local  # local span untouched
    assert out.spans[1] is moved_span  # foreign span replaced


def test_replace_nomove_returns_identical_ref():
    # nothing moves → SAME object back, preserving grad / retain_grad / _packed_cu_seqlens
    # (which with_spans would drop). This case is dropped on the pre-refactor code.
    pool = _FakePool({"w0": 0})
    ref = _ref(_span(torch.zeros(2, 4), source_id="w0"))
    ref.grad = _ref(_span(torch.zeros(2, 4), source_id="w0"))
    ref.retain_grad()
    cu = torch.tensor([0, 1, 2])
    object.__setattr__(ref, "_packed_cu_seqlens", cu)
    out = WorkerLocalTransport._replace_leaf({}, ("w0", 0), pool)(ref)
    assert out is ref
    assert out.grad is ref.grad and out.retain_grad_flag is True
    assert out._packed_cu_seqlens is cu


def test_replace_rewrites_nested_containers():
    pool = _FakePool({"w0": 0, "w1": 1})
    foreign = _span(torch.ones(3, 4), source_id="w1", store_key="kb")
    ref = _ref(foreign)
    key = WorkerLocalTransport._move_key(foreign, ("w0", 0), pool)
    moved_span = _span(torch.full((3, 4), 7.0), source_id="w0", store_key="kb_recv")
    leaf = WorkerLocalTransport._replace_leaf({key: moved_span}, ("w0", 0), pool)
    assert map_tree({"x": ref}, leaf)["x"].spans[0] is moved_span  # nested in dict
    assert map_tree([ref], leaf)[0].spans[0] is moved_span  # nested in list
    assert map_tree(_Holder(ref=ref), leaf).ref.spans[0] is moved_span  # nested in Batch field


def test_localize_all_local_returns_shards_untouched():
    # FIND finds nothing foreign → localize returns the SAME shards (no MOVE, no Ray)
    pool = _FakePool({"w0": 0, "w1": 1})
    shard0 = ((_ref(_span(torch.zeros(2, 4), source_id="w0")),), {})
    shard1 = ((_ref(_span(torch.zeros(2, 4), source_id="w1")),), {})
    shards = [shard0, shard1]
    out = WorkerLocalTransport.localize(shards, pool, device_ids=[0, 1], worker_ids=["w0", "w1"])
    assert out is shards
