"""Unit tests for the ``Batch`` field-kind container — concat/select/slice/pack.

``Batch`` is a ``@dataclass`` mixin whose fields are annotated with field-kind
declarators (``concat_field`` / ``shared_field`` / ``packed_field`` / the four
reduction fields). The base class derives generic concat / chunk / select /
slice / repeat_interleave / clone / to_device / map implementations that
dispatch on the per-field kind and the value type. This module exercises those
field-kind semantics and the per-instance ops with small CPU int/float tensors,
so every concatenated / reduced / re-indexed value is asserted exactly.

Pure-CPU: plain ``torch`` tensors and locally-declared ``Batch`` subclasses,
no transport backend, no Ray, no GPU.
"""

from dataclasses import dataclass
from typing import List, Optional

import pytest
import torch

from unirl.distributed.tensor.batch import (
    Batch,
    concat_field,
    max_field,
    mean_field,
    min_field,
    packed_field,
    shared_field,
    sum_field,
)
from unirl.distributed.tensor.batch import (
    _concat_cu_seqlens,
    _select_cu_seqlens,
    _slice_cu_seqlens,
)

pytestmark = pytest.mark.cpu


# ── sample Batch subclasses ──────────────────────────────────────────────────


@dataclass
class SimpleBatch(Batch):
    """concat + shared + every reduction kind on one container."""

    data: torch.Tensor = concat_field(default=None)
    labels: List[str] = concat_field(default_factory=list)
    config: str = shared_field(default="cfg")
    total: torch.Tensor = sum_field(default=None)
    avg: torch.Tensor = mean_field(default=None)
    lo: torch.Tensor = min_field(default=None)
    hi: torch.Tensor = max_field(default=None)
    wall_clock: float = max_field(default=0.0)


@dataclass
class PackedBatch(Batch):
    """Two packed (varlen) fields that share one instance-level cu_seqlens,
    plus a per-sample concat field to cross-check batch_size inference."""

    tokens: Optional[torch.Tensor] = packed_field(default=None)
    log_probs: Optional[torch.Tensor] = packed_field(default=None)
    sample_idx: Optional[torch.Tensor] = concat_field(default=None)


def _mk(data, labels, **kw):
    return SimpleBatch(data=torch.tensor(data, dtype=torch.float32), labels=list(labels), **kw)


# ── concat_field: per-sample, concatenated; split on chunk/slice/select ───────


def test_concat_field_concatenates_per_sample():
    a = _mk([1.0, 2.0], ["a", "b"])
    b = _mk([3.0, 4.0, 5.0], ["c", "d", "e"])
    out = SimpleBatch.concat([a, b])
    assert torch.equal(out.data, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert out.labels == ["a", "b", "c", "d", "e"]  # list concat fields merge too
    assert out.batch_size == 5


def test_concat_field_split_on_chunk_slice_select():
    out = _mk([10.0, 11.0, 12.0, 13.0], ["w", "x", "y", "z"])
    # chunk -> equal contiguous shards
    c0, c1 = out.chunk(2)
    assert torch.equal(c0.data, torch.tensor([10.0, 11.0])) and c0.labels == ["w", "x"]
    assert torch.equal(c1.data, torch.tensor([12.0, 13.0])) and c1.labels == ["y", "z"]
    # slice
    s = out.slice(1, 3)
    assert torch.equal(s.data, torch.tensor([11.0, 12.0])) and s.labels == ["x", "y"]
    # select (gather/permute)
    g = out.select([3, 1])
    assert torch.equal(g.data, torch.tensor([13.0, 11.0])) and g.labels == ["z", "x"]


# ── shared_field: identical across samples; FIRST taken on concat ────────────


def test_shared_field_takes_first_on_concat():
    a = _mk([1.0], ["a"], config="alpha")
    b = _mk([2.0], ["b"], config="beta")  # differing shared value
    out = SimpleBatch.concat([a, b])
    assert out.config == "alpha"  # first instance's shared value wins


def test_shared_field_passes_through_chunk_slice_select():
    out = _mk([1.0, 2.0, 3.0, 4.0], ["a", "b", "c", "d"], config="zeta")
    assert out.slice(1, 3).config == "zeta"
    assert out.select([2, 0]).config == "zeta"
    assert all(sh.config == "zeta" for sh in out.chunk(2))


# ── reduction fields: sum/mean/min/max applied across instances on concat ────


def test_reduction_fields_reduce_across_instances():
    a = _mk(
        [1.0],
        ["a"],
        total=torch.tensor([1.0, 2.0]),
        avg=torch.tensor([2.0, 4.0]),
        lo=torch.tensor([5.0, 1.0]),
        hi=torch.tensor([5.0, 1.0]),
        wall_clock=3.0,
    )
    b = _mk(
        [2.0],
        ["b"],
        total=torch.tensor([3.0, 4.0]),
        avg=torch.tensor([4.0, 8.0]),
        lo=torch.tensor([2.0, 9.0]),
        hi=torch.tensor([2.0, 9.0]),
        wall_clock=7.0,
    )
    out = SimpleBatch.concat([a, b])
    assert torch.equal(out.total, torch.tensor([4.0, 6.0]))  # elementwise sum
    assert torch.equal(out.avg, torch.tensor([3.0, 6.0]))  # elementwise mean
    assert torch.equal(out.lo, torch.tensor([2.0, 1.0]))  # elementwise min
    assert torch.equal(out.hi, torch.tensor([5.0, 9.0]))  # elementwise max
    assert out.wall_clock == 7.0  # scalar (python) max reduction


def test_reduction_fields_pass_through_slice_select():
    out = _mk(
        [1.0, 2.0, 3.0],
        ["a", "b", "c"],
        total=torch.tensor([9.0]),
        wall_clock=5.0,
    )
    # reductions are batch-shared metadata: untouched by re-indexing
    assert torch.equal(out.slice(0, 1).total, torch.tensor([9.0]))
    assert out.select([2, 1, 0]).wall_clock == 5.0


# ── packed_field + cu_seqlens ────────────────────────────────────────────────


def test_pack_computes_cu_seqlens_and_lengths():
    seg = PackedBatch.pack(
        tokens=[torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0]), torch.tensor([5.0, 6.0])],
        log_probs=[torch.tensor([0.1, 0.2, 0.3]), torch.tensor([0.4]), torch.tensor([0.5, 0.6])],
        sample_idx=torch.arange(3),
    )
    # packed along dim 0
    assert torch.equal(seg.tokens, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    # cu_seqlens are cumulative per-sample offsets [N+1], lengths derived
    assert torch.equal(seg.cu_seqlens, torch.tensor([0, 3, 4, 6]))
    assert torch.equal(seg.lengths, torch.tensor([3, 1, 2]))
    assert seg.batch_size == 3  # inferred from cu (and matches sample_idx concat field)


def test_packed_concat_merges_cu_offsets_with_shift():
    a = PackedBatch.pack(tokens=[torch.tensor([1.0, 2.0]), torch.tensor([3.0])])  # cu [0,2,3]
    b = PackedBatch.pack(tokens=[torch.tensor([4.0, 5.0, 6.0])])  # cu [0,3]
    out = PackedBatch.concat([a, b])
    assert torch.equal(out.tokens, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    # b's cu (minus its leading 0) shifted by a's running total (3): [0,2,3] + [3+3]=6
    assert torch.equal(out.cu_seqlens, torch.tensor([0, 2, 3, 6]))
    assert out.batch_size == 3


def test_packed_slice_rebuilds_cu_seqlens():
    seg = PackedBatch.pack(
        tokens=[torch.tensor([1.0]), torch.tensor([2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0])]
    )  # cu [0,1,3,6]
    sl = seg.slice(1, 3)  # samples 1,2
    assert torch.equal(sl.tokens, torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0]))
    assert torch.equal(sl.cu_seqlens, torch.tensor([0, 2, 5]))  # re-zeroed
    assert sl.batch_size == 2


def test_packed_select_rebuilds_cu_seqlens():
    seg = PackedBatch.pack(
        tokens=[torch.tensor([1.0]), torch.tensor([2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0])]
    )  # cu [0,1,3,6]
    g = seg.select([2, 0])  # sample 2 (size 3) then sample 0 (size 1)
    assert torch.equal(g.tokens, torch.tensor([4.0, 5.0, 6.0, 1.0]))
    assert torch.equal(g.cu_seqlens, torch.tensor([0, 3, 4]))


def test_packed_roundtrip_pack_then_slice():
    parts = [torch.tensor([1.0, 2.0]), torch.tensor([3.0]), torch.tensor([4.0, 5.0, 6.0]), torch.tensor([7.0])]
    seg = PackedBatch.pack(tokens=parts, log_probs=parts)
    # slicing out a single sample reproduces that original per-sample chunk exactly,
    # for both packed fields sharing the one cu_seqlens
    for i, part in enumerate(parts):
        one = seg.slice(i, i + 1)
        assert torch.equal(one.tokens, part)
        assert torch.equal(one.log_probs, part)
        assert torch.equal(one.cu_seqlens, torch.tensor([0, int(part.shape[0])]))


def test_packed_repeat_interleave_rebuilds_cu():
    seg = PackedBatch.pack(tokens=[torch.tensor([1.0]), torch.tensor([2.0, 3.0])])  # cu [0,1,3]
    r = seg.repeat_interleave(2)  # each chunk duplicated, group-by-parent
    assert torch.equal(r.tokens, torch.tensor([1.0, 1.0, 2.0, 3.0, 2.0, 3.0]))
    assert torch.equal(r.cu_seqlens, torch.tensor([0, 1, 2, 4, 6]))
    assert r.batch_size == 4


# ── standalone cu_seqlens helpers ────────────────────────────────────────────


def test_concat_cu_seqlens_helper():
    # shard 0 taken as-is; each later shard contributes cu[1:] + running_total
    merged = _concat_cu_seqlens([torch.tensor([0, 2, 3]), torch.tensor([0, 1, 4])])
    assert torch.equal(merged, torch.tensor([0, 2, 3, 4, 7]))
    # None shards are skipped; all-None -> None
    assert torch.equal(_concat_cu_seqlens([None, torch.tensor([0, 5])]), torch.tensor([0, 5]))
    assert _concat_cu_seqlens([None, None]) is None


def test_slice_cu_seqlens_helper():
    cu = torch.tensor([0, 1, 3, 6])
    assert torch.equal(_slice_cu_seqlens(cu, 1, 3), torch.tensor([0, 2, 5]))  # re-zeroed
    assert torch.equal(_slice_cu_seqlens(cu, 0, 0), torch.tensor([0]))  # empty range -> [0]
    assert torch.equal(_slice_cu_seqlens(cu, 0, 3), torch.tensor([0, 1, 3, 6]))  # full == identity


def test_select_cu_seqlens_helper():
    cu = torch.tensor([0, 1, 3, 6])  # per-sample sizes 1, 2, 3
    # rebuild from selected sizes, in selection order
    assert torch.equal(_select_cu_seqlens(cu, [2, 0]), torch.tensor([0, 3, 4]))
    assert torch.equal(_select_cu_seqlens(cu, []), torch.tensor([0]))
    assert torch.equal(_select_cu_seqlens(cu, [1, 1]), torch.tensor([0, 2, 4]))  # dup index


# ── concat / chunk ───────────────────────────────────────────────────────────


def test_concat_single_item_returns_same_object():
    a = _mk([1.0, 2.0], ["a", "b"])
    assert SimpleBatch.concat([a]) is a  # len==1 short-circuits


def test_concat_empty_raises():
    with pytest.raises(ValueError):
        SimpleBatch.concat([])


def test_concat_with_method():
    a = _mk([1.0], ["a"])
    b = _mk([2.0], ["b"])
    c = _mk([3.0], ["c"])
    out = a.concat_with(b, c)
    assert torch.equal(out.data, torch.tensor([1.0, 2.0, 3.0]))


def test_chunk_divisible():
    out = _mk([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], list("abcdef"))
    shards = out.chunk(3)
    assert len(shards) == 3
    assert torch.equal(shards[0].data, torch.tensor([0.0, 1.0]))
    assert torch.equal(shards[2].data, torch.tensor([4.0, 5.0]))
    # chunk is the structural inverse of concat
    assert torch.equal(SimpleBatch.concat(shards).data, out.data)


def test_chunk_non_divisible_raises():
    out = _mk([0.0, 1.0, 2.0], ["a", "b", "c"])
    with pytest.raises(ValueError):
        out.chunk(2)


def test_chunk_non_positive_raises():
    out = _mk([0.0, 1.0], ["a", "b"])
    with pytest.raises(ValueError):
        out.chunk(0)
    with pytest.raises(ValueError):
        out.chunk(-1)


# ── select: sparse / dup / reverse ───────────────────────────────────────────


def test_select_sparse():
    out = _mk([0.0, 1.0, 2.0, 3.0, 4.0], list("abcde"))
    g = out.select([0, 4])
    assert torch.equal(g.data, torch.tensor([0.0, 4.0])) and g.labels == ["a", "e"]


def test_select_duplicate_indices():
    out = _mk([0.0, 1.0, 2.0], ["a", "b", "c"])
    g = out.select([1, 1, 1])
    assert torch.equal(g.data, torch.tensor([1.0, 1.0, 1.0])) and g.labels == ["b", "b", "b"]


def test_select_reverse():
    out = _mk([0.0, 1.0, 2.0], ["a", "b", "c"])
    g = out.select([2, 1, 0])
    assert torch.equal(g.data, torch.tensor([2.0, 1.0, 0.0])) and g.labels == ["c", "b", "a"]


def test_select_accepts_tensor_indices():
    out = _mk([0.0, 1.0, 2.0], ["a", "b", "c"])
    g = out.select(torch.tensor([2, 0]))
    assert torch.equal(g.data, torch.tensor([2.0, 0.0]))


# ── slice: aligned / mid / empty / full ──────────────────────────────────────


def test_slice_variants():
    out = _mk([0.0, 1.0, 2.0, 3.0], list("abcd"))
    # full
    full = out.slice(0, 4)
    assert torch.equal(full.data, out.data) and full.labels == out.labels
    # mid
    mid = out.slice(1, 3)
    assert torch.equal(mid.data, torch.tensor([1.0, 2.0])) and mid.labels == ["b", "c"]
    # aligned head
    head = out.slice(0, 2)
    assert torch.equal(head.data, torch.tensor([0.0, 1.0]))
    # empty
    empty = out.slice(2, 2)
    assert empty.data.numel() == 0 and empty.labels == []


# ── repeat_interleave: n=0 / 1 / >1 ──────────────────────────────────────────


def test_repeat_interleave_gt1():
    out = _mk([0.0, 1.0], ["a", "b"])
    r = out.repeat_interleave(3)
    assert torch.equal(r.data, torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]))  # group-by-parent
    assert r.labels == ["a", "a", "a", "b", "b", "b"]
    assert r.batch_size == 6


def test_repeat_interleave_one_clones():
    out = _mk([0.0, 1.0], ["a", "b"])
    r = out.repeat_interleave(1)
    assert torch.equal(r.data, out.data) and r.labels == out.labels
    assert r.data is not out.data  # clone, not alias


def test_repeat_interleave_zero_empties():
    out = _mk([0.0, 1.0], ["a", "b"])
    r = out.repeat_interleave(0)
    assert r.data.numel() == 0 and r.labels == []
    assert r.batch_size == 0


def test_repeat_interleave_negative_raises():
    out = _mk([0.0, 1.0], ["a", "b"])
    with pytest.raises(ValueError):
        out.repeat_interleave(-1)


# ── batch_size inference ─────────────────────────────────────────────────────


def test_batch_size_from_first_concat_field():
    out = _mk([0.0, 1.0, 2.0], ["a", "b", "c"])
    assert out.batch_size == 3


def test_batch_size_from_packed_cu_when_no_concat():
    seg = PackedBatch.pack(tokens=[torch.tensor([1.0, 2.0]), torch.tensor([3.0])])
    # no concat field populated -> falls back to len(cu_seqlens) - 1
    assert seg.sample_idx is None
    assert seg.batch_size == 2


def test_batch_size_zero_when_empty():
    out = SimpleBatch()
    assert out.batch_size == 0


# ── clone / to_device / map ──────────────────────────────────────────────────


def test_clone_deep_copies_tensors():
    out = _mk([0.0, 1.0], ["a", "b"])
    cl = out.clone()
    assert torch.equal(cl.data, out.data)
    assert cl.data is not out.data  # tensor cloned
    cl.data[0] = 99.0
    assert out.data[0] == 0.0  # mutation does not leak back


def test_clone_carries_packed_cu():
    seg = PackedBatch.pack(tokens=[torch.tensor([1.0]), torch.tensor([2.0, 3.0])])
    cl = seg.clone()
    assert torch.equal(cl.cu_seqlens, seg.cu_seqlens)
    assert cl.cu_seqlens is not seg.cu_seqlens
    assert torch.equal(cl.tokens, seg.tokens) and cl.tokens is not seg.tokens


def test_to_device_cpu_moves_tensors():
    out = _mk([0.0, 1.0], ["a", "b"])
    moved = out.to_device("cpu")
    assert moved.data.device.type == "cpu"
    assert torch.equal(moved.data, out.data)


def test_to_device_carries_packed_cu():
    seg = PackedBatch.pack(tokens=[torch.tensor([1.0, 2.0]), torch.tensor([3.0])])
    moved = seg.to_device("cpu")
    assert torch.equal(moved.cu_seqlens, seg.cu_seqlens)
    assert moved.cu_seqlens.device.type == "cpu"


def test_map_applies_fn_per_field_value():
    out = _mk([1.0, 2.0], ["a", "b"])
    mapped = out.map(lambda v: v * 2 if isinstance(v, torch.Tensor) else v)
    assert torch.equal(mapped.data, torch.tensor([2.0, 4.0]))
    assert mapped.labels == ["a", "b"]  # non-tensor field untouched


def test_map_preserves_packed_cu():
    seg = PackedBatch.pack(tokens=[torch.tensor([1.0]), torch.tensor([2.0, 3.0])])
    # representation-only transform (no batch-dim change) -> cu carries over
    mapped = seg.map(lambda v: v.clone() if isinstance(v, torch.Tensor) else v)
    assert torch.equal(mapped.cu_seqlens, seg.cu_seqlens)
    assert torch.equal(mapped.tokens, seg.tokens)


# ── pack() validation errors ─────────────────────────────────────────────────


def test_pack_rejects_already_packed_tensor():
    # an already-packed Tensor is a TypeError directing to the plain constructor
    with pytest.raises(TypeError):
        PackedBatch.pack(tokens=torch.tensor([1.0, 2.0, 3.0]))


def test_pack_rejects_non_sequence():
    with pytest.raises(TypeError):
        PackedBatch.pack(tokens=42)


def test_pack_rejects_non_tensor_element():
    with pytest.raises(TypeError):
        PackedBatch.pack(tokens=[torch.tensor([1.0]), "not a tensor"])


def test_pack_rejects_size_mismatch_across_fields():
    # two packed fields must agree on per-sample sizes (shared cu_seqlens)
    with pytest.raises(ValueError):
        PackedBatch.pack(
            tokens=[torch.tensor([1.0, 2.0]), torch.tensor([3.0])],  # sizes [2, 1]
            log_probs=[torch.tensor([0.1]), torch.tensor([0.2, 0.3])],  # sizes [1, 2]
        )


def test_pack_none_field_is_valid():
    # None is a valid packed-field value: that field stays empty, contributes no sizes
    seg = PackedBatch.pack(tokens=[torch.tensor([1.0, 2.0]), torch.tensor([3.0])], log_probs=None)
    assert seg.log_probs is None
    assert torch.equal(seg.cu_seqlens, torch.tensor([0, 2, 3]))
