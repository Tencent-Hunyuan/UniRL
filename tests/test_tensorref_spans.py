"""Unit tests for TensorRef ref views (select/slice without hydration).

Selection emits :class:`TensorSpan` refs — unit-range views of the parent
refs — instead of moving data. Pure-CPU: parent refs are faked with a minimal
handle protocol (.local/.shape/.dtype/.device), no transport backend required
(materialize falls back to per-ref fetch when backend is None).
"""

import torch

from unirl.distributed.tensor.transport import TensorSpan, TensorRef, cat_rows


class _FakeHandle:
    def __init__(self, t: torch.Tensor):
        self.t = t
        self.shape = t.shape
        self.dtype = t.dtype
        self.device = t.device

    def local(self) -> torch.Tensor:
        return self.t


def _meta(*tensors: torch.Tensor) -> TensorRef:
    return TensorRef(
        refs=[_FakeHandle(t) for t in tensors],
        sizes=[int(t.shape[0]) for t in tensors],
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
    assert any(isinstance(r, TensorSpan) for r in v.refs)
    out = v.materialize(backend=None)
    assert out.shape == (7, 6)  # ragged refs right-padded to the max width
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


def test_aligned_slice_passes_refs_through():
    # A ref-boundary-aligned slice is the structural inverse of concat:
    # the original ref objects come back untouched (no TensorSpan wrapping).
    t0 = torch.arange(12).reshape(3, 4).float()
    t1 = torch.arange(100, 108).reshape(2, 4).float()
    tm = _meta(t0, t1)
    head = tm.slice(0, 3)
    assert head.refs == [tm.refs[0]] and head.sizes == [3]


def test_misaligned_slice_wraps_boundary_refs():
    t0 = torch.arange(12).reshape(3, 4).float()
    t1 = torch.arange(100, 108).reshape(2, 4).float()
    tm = _meta(t0, t1)
    mid = tm.slice(1, 4)  # crosses the ref boundary off-alignment
    assert mid.batch_size == 3
    assert isinstance(mid.refs[0], TensorSpan) and isinstance(mid.refs[1], TensorSpan)
    assert torch.equal(mid.materialize(backend=None), torch.cat([t0[1:], t1[:1]]))


def test_packed_segment_view():
    p0 = torch.arange(10).float()
    p1 = torch.arange(100, 106).float()
    pm = _meta(p0, p1)
    pv = pm.select_segments([(12, 16), (0, 3)])  # out-of-order token ranges
    assert pv.batch_size == 7
    assert torch.equal(pv.materialize(backend=None), torch.cat([p1[2:6], p0[0:3]]))


def test_nested_views_flatten():
    # A view of a view flattens to a single TensorSpan over the parent ref —
    # repeated selection never builds an indirection chain.
    t0 = torch.arange(40).reshape(8, 5).float()
    v1 = _meta(t0).select([3, 4, 5, 6])  # rows 3..6 (one coalesced view)
    v2 = v1.select([1, 2])  # rows 4..5 of the original
    assert all(isinstance(r, TensorSpan) and isinstance(r.base, _FakeHandle) for r in v2.refs)
    assert torch.equal(v2.materialize(backend=None), t0[4:6])


def test_with_refs_preserves_sizes():
    t0 = torch.arange(8).reshape(2, 4).float()
    v = _meta(t0).select([1, 0])
    v2 = v.with_refs(list(v.refs))
    assert v2.sizes == v.sizes and v2.batch_size == v.batch_size


def test_empty_selection():
    p0 = torch.arange(10).float()
    e = _meta(p0).select_segments([])
    assert e.batch_size == 0
    assert e.materialize(backend=None).numel() == 0


def test_view_shape_and_local():
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
