"""Complementary pure-CPU coverage of localize FIND/REPLACE for MULTI-source and
partial spans. The single-span cases live in ``tests/test_tensorref_spans.py``;
this adds the cases that need a ref whose spans span different workers. No
Ray/GPU — fake handles + a fake pool. Marker: cpu.
"""

from __future__ import annotations

import pytest
import torch

from unirl.distributed.tensor.transport import TensorRef, TensorSpan, WorkerLocalTransport

pytestmark = pytest.mark.cpu


class _FakeHandle:
    def __init__(self, t, source_id="w0", store_key="k0", object_ref=None):
        self.t = t
        self.shape = t.shape
        self.dtype = t.dtype
        self.device = t.device
        self.source_id = source_id
        self.store_key = store_key
        self.object_ref = object_ref

    def local(self):
        return self.t


class _FakePool:
    def __init__(self, mapping):
        self._m = mapping

    def device_id_of(self, worker_id):
        return self._m[worker_id]


def _span(t, source_id, store_key, start=0, stop=None):
    h = _FakeHandle(t, source_id=source_id, store_key=store_key)
    return TensorSpan(h, start, int(t.shape[0]) if stop is None else stop)


def _ref(*spans):
    total = sum(s.stop - s.start for s in spans)
    return TensorRef(spans=list(spans), shape=(total, *spans[0].shape[1:]), dtype=spans[0].dtype, device="cpu")


def test_move_key_partial_span_carries_start_stop():
    # a partial (sliced) span keys by its [start, stop) rows, so two different
    # row-windows of one foreign block do NOT collapse together.
    pool = _FakePool({"w0": 0, "w1": 1})
    s = _span(torch.zeros(8, 4), "w1", "kp", start=2, stop=5)
    assert WorkerLocalTransport._move_key(s, ("w0", 0), pool) == (1, 0, "kp", 2, 5)
    s2 = _span(torch.zeros(8, 4), "w1", "kp", start=0, stop=3)
    assert WorkerLocalTransport._move_key(s2, ("w0", 0), pool) != WorkerLocalTransport._move_key(s, ("w0", 0), pool)


def test_replace_mixed_local_foreign_spans():
    # concat(dw0-ref, dw1-ref) → two spans of different sources; localize to dw0
    # swaps only the foreign span, leaving the local one as the same object.
    pool = _FakePool({"w0": 0, "w1": 1})
    local = _span(torch.zeros(2, 4), "w0", "ka")
    foreign = _span(torch.ones(3, 4), "w1", "kb")
    ref = _ref(local, foreign)
    key = WorkerLocalTransport._move_key(foreign, ("w0", 0), pool)
    moved = _span(torch.full((3, 4), 7.0), "w0", "kb_recv")
    out = WorkerLocalTransport._replace_leaf({key: moved}, ("w0", 0), pool)(ref)
    assert out is not ref
    assert out.spans[0] is local and out.spans[1] is moved
    assert WorkerLocalTransport._move_key(local, ("w0", 0), pool) is None  # local span never queued


def test_find_dedups_two_spans_same_foreign_slice():
    # two distinct span objects over the same foreign block+range → equal keys.
    pool = _FakePool({"w0": 0, "w1": 1})
    a = _span(torch.zeros(4, 4), "w1", "kc")
    b = _span(torch.zeros(4, 4), "w1", "kc")
    dst = ("w0", 0)
    to_move = {}
    to_move.setdefault(WorkerLocalTransport._move_key(a, dst, pool), a)
    to_move.setdefault(WorkerLocalTransport._move_key(b, dst, pool), b)
    assert len(to_move) == 1 and a is not b
