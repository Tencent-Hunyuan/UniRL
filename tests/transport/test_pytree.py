"""Unit tests for the pytree batch-axis walkers: chunk / cat / infer_batch_size.

``pytree_chunk`` shards one same-structured tree into ``dp_size`` per-rank trees
along axis 0; ``pytree_cat`` is its inverse, merging ``N`` same-structured trees
back into one. ``infer_batch_size`` (via ``_value_batch_size``) derives the
batch-axis size those two split/merge along, returning the FIRST batch-axis size
found and treating ``Broadcast``-wrapped values as opt-outs.

Pure-CPU and backend-agnostic: plain torch CPU tensors / numpy arrays / Python
containers, plus ``TensorRef`` proxies built over faked store handles (the
``_FakeHandle`` / ``_meta`` pattern copied from ``test_tensorref_spans.py``) so
the TensorRef branches exercise without any transport backend.
"""

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from unirl.distributed.tensor import TensorRef, TensorSpan
from unirl.distributed.tensor.batch import Batch, concat_field, shared_field
from unirl.distributed.tensor.pytree import (
    _value_batch_size,
    infer_batch_size,
    pytree_cat,
    pytree_chunk,
)
from unirl.distributed.utils import Broadcast

pytestmark = pytest.mark.cpu


# ── TensorRef construction helpers (copied from test_tensorref_spans.py) ──────


class _FakeHandle:
    def __init__(self, t: torch.Tensor):
        self.t = t
        self.shape = t.shape
        self.dtype = t.dtype
        self.device = t.device

    def local(self) -> torch.Tensor:
        return self.t


def _meta(*tensors: torch.Tensor) -> TensorRef:
    """A TensorRef with one full-range span per tensor (spans are the concat axis)."""
    return TensorRef(
        spans=[TensorSpan(_FakeHandle(t), 0, int(t.shape[0])) for t in tensors],
        shape=(sum(int(t.shape[0]) for t in tensors), *tensors[0].shape[1:]),
        dtype=tensors[0].dtype,
        device="cpu",
    )


@dataclass
class _Sample(Batch):
    """A minimal concat-field Batch: ``data`` is batch-aligned, ``tag`` is shared."""

    data: torch.Tensor = concat_field(default=None)
    tag: str = shared_field(default="t")


# ══════════════════════════════════════════════════════════════════════════════
# infer_batch_size / _value_batch_size
# ══════════════════════════════════════════════════════════════════════════════


def test_value_batch_size_tensor_uses_dim0():
    assert _value_batch_size(torch.zeros(7, 3)) == 7


def test_value_batch_size_scalar_tensor_is_none():
    # 0-dim tensor has an empty shape -> no batch axis.
    assert _value_batch_size(torch.tensor(5.0)) is None


def test_value_batch_size_ndarray_uses_axis0():
    assert _value_batch_size(np.zeros((4, 2))) == 4


def test_value_batch_size_list_uses_len():
    assert _value_batch_size([10, 11, 12]) == 3


def test_value_batch_size_tensorref_uses_batch_size():
    ref = _meta(torch.zeros(2, 4), torch.zeros(3, 4))  # 5 rows over 2 spans
    assert _value_batch_size(ref) == 5


def test_value_batch_size_batch_uses_batch_size():
    assert _value_batch_size(_Sample(data=torch.zeros(6, 2))) == 6


def test_value_batch_size_scalar_str_none_are_none():
    assert _value_batch_size(3) is None
    assert _value_batch_size(1.5) is None
    assert _value_batch_size("hello") is None
    assert _value_batch_size(None) is None


def test_value_batch_size_broadcast_opts_out():
    # Broadcast contributes no size even when its inner value has a batch axis.
    assert _value_batch_size(Broadcast(torch.zeros(9, 3))) is None


def test_value_batch_size_tuple_and_dict_recurse_first_hit():
    # str scalar contributes nothing -> first real batch axis (the tensor) wins.
    assert _value_batch_size(("skip", torch.zeros(5, 2))) == 5
    assert _value_batch_size({"a": None, "b": [1, 2, 3, 4]}) == 4


def test_infer_batch_size_scans_args_then_kwargs():
    # args are scanned before kwargs; first batch-axis size found wins.
    assert infer_batch_size((torch.zeros(8, 3),), {}) == 8
    assert infer_batch_size((), {"x": torch.zeros(2, 3)}) == 2
    # leading non-batched arg (Broadcast) skipped, next arg supplies the size.
    assert infer_batch_size((Broadcast(0.01), torch.zeros(4, 1)), {}) == 4


def test_infer_batch_size_pure_broadcast_is_none():
    # No batched field anywhere -> None (dispatch replicates the whole payload).
    assert infer_batch_size((Broadcast(torch.zeros(4, 2)), 3, "s"), {"lr": 0.1}) is None


# ══════════════════════════════════════════════════════════════════════════════
# pytree_chunk
# ══════════════════════════════════════════════════════════════════════════════


def test_chunk_tensor_splits_into_equal_dim0_chunks():
    t = torch.arange(8).reshape(4, 2).float()
    shards = pytree_chunk(t, dp_size=2, batch_size=4)
    assert len(shards) == 2
    assert torch.equal(shards[0], t[:2])
    assert torch.equal(shards[1], t[2:])


def test_chunk_broadcast_replicates_inner_value():
    inner = torch.zeros(4, 2)
    shards = pytree_chunk(Broadcast(inner), dp_size=3, batch_size=4)
    assert len(shards) == 3
    assert all(s is inner for s in shards)  # the unwrapped value, replicated


def test_chunk_scalar_tensor_replicated():
    t = torch.tensor(7.0)  # 0-dim -> replicated, not split
    shards = pytree_chunk(t, dp_size=2, batch_size=4)
    assert shards == [t, t]


def test_chunk_tensor_with_mismatched_dim0_replicated():
    # dim0 != batch_size -> per-rollout metadata, replicated rather than split.
    t = torch.zeros(3, 5)
    shards = pytree_chunk(t, dp_size=2, batch_size=8)
    assert len(shards) == 2
    assert all(s is t for s in shards)


def test_chunk_ndarray_splits_along_axis0():
    a = np.arange(12).reshape(4, 3)
    shards = pytree_chunk(a, dp_size=2, batch_size=4)
    assert len(shards) == 2
    assert np.array_equal(shards[0], a[:2])
    assert np.array_equal(shards[1], a[2:])


def test_chunk_list_of_batch_len_is_sliced():
    lst = [0, 1, 2, 3, 4, 5]
    shards = pytree_chunk(lst, dp_size=3, batch_size=6)
    assert shards == [[0, 1], [2, 3], [4, 5]]


def test_chunk_list_of_other_len_replicated():
    lst = [0, 1, 2]  # len != batch_size -> replicated whole
    shards = pytree_chunk(lst, dp_size=2, batch_size=8)
    assert shards == [lst, lst]


def test_chunk_scalar_and_none_replicated():
    assert pytree_chunk(42, dp_size=2, batch_size=4) == [42, 42]
    assert pytree_chunk("s", dp_size=2, batch_size=4) == ["s", "s"]
    assert pytree_chunk(None, dp_size=3, batch_size=4) == [None, None, None]


def test_chunk_dict_reassembled_per_shard():
    d = {"x": torch.arange(4).float(), "lr": 0.1}
    shards = pytree_chunk(d, dp_size=2, batch_size=4)
    assert len(shards) == 2
    # batched value sliced per shard; scalar replicated into each shard dict.
    assert torch.equal(shards[0]["x"], torch.tensor([0.0, 1.0]))
    assert torch.equal(shards[1]["x"], torch.tensor([2.0, 3.0]))
    assert shards[0]["lr"] == 0.1 and shards[1]["lr"] == 0.1


def test_chunk_tuple_reassembled_per_shard():
    tup = (torch.arange(4).float(), "meta")
    shards = pytree_chunk(tup, dp_size=2, batch_size=4)
    assert len(shards) == 2
    assert isinstance(shards[0], tuple) and len(shards[0]) == 2
    assert torch.equal(shards[0][0], torch.tensor([0.0, 1.0]))
    assert shards[0][1] == "meta" and shards[1][1] == "meta"


def test_chunk_batch_recursion_per_shard():
    b = _Sample(data=torch.arange(8).reshape(4, 2).float(), tag="g")
    shards = pytree_chunk(b, dp_size=2, batch_size=4)
    assert len(shards) == 2
    # CONCAT field sliced, SHARED field carried through untouched.
    assert torch.equal(shards[0].data, b.data[:2])
    assert torch.equal(shards[1].data, b.data[2:])
    assert shards[0].tag == "g" and shards[1].tag == "g"


def test_chunk_batch_with_mismatched_size_replicated():
    b = _Sample(data=torch.zeros(3, 2))  # batch_size 3 != dispatch 8
    shards = pytree_chunk(b, dp_size=2, batch_size=8)
    assert all(s is b for s in shards)


def test_chunk_divisibility_error():
    t = torch.zeros(5, 2)  # 5 rows, dp_size 2 -> not divisible
    with pytest.raises(ValueError):
        pytree_chunk(t, dp_size=2, batch_size=5)


def test_chunk_tensorref_splits_spans_across_shards():
    # 4 spans (1 row each), batch_size 4, dp_size 2 -> 2 spans per shard.
    rows = [torch.tensor([[float(i)]]) for i in range(4)]  # each (1,1)
    ref = _meta(*rows)
    assert ref.batch_size == 4 and len(ref.spans) == 4
    shards = pytree_chunk(ref, dp_size=2, batch_size=4)
    assert len(shards) == 2
    assert all(isinstance(s, TensorRef) for s in shards)
    assert shards[0].batch_size == 2 and shards[1].batch_size == 2
    # shard contents are the matching row-windows of the original.
    assert torch.equal(shards[0].materialize(backend=None), torch.tensor([[0.0], [1.0]]))
    assert torch.equal(shards[1].materialize(backend=None), torch.tensor([[2.0], [3.0]]))


def test_chunk_tensorref_mismatched_size_replicated():
    ref = _meta(torch.zeros(3, 2))  # batch_size 3 != dispatch 8
    shards = pytree_chunk(ref, dp_size=2, batch_size=8)
    assert all(s is ref for s in shards)


def test_chunk_tensorref_span_indivisible_raises():
    # batch_size divisible by dp_size, but span COUNT is not (1 span, dp_size 2).
    ref = _meta(torch.zeros(4, 2))  # 1 span holding all 4 rows
    assert len(ref.spans) == 1
    with pytest.raises(ValueError):
        pytree_chunk(ref, dp_size=2, batch_size=4)


# ══════════════════════════════════════════════════════════════════════════════
# pytree_cat
# ══════════════════════════════════════════════════════════════════════════════


def test_cat_empty_is_none():
    assert pytree_cat([]) is None


def test_cat_all_none_is_none():
    assert pytree_cat([None, None]) is None


def test_cat_tensors_concat_dim0():
    a = torch.arange(4).reshape(2, 2).float()
    b = torch.arange(4, 8).reshape(2, 2).float()
    out = pytree_cat([a, b])
    assert torch.equal(out, torch.cat([a, b], dim=0))


def test_cat_ndarrays_concat_axis0():
    a = np.arange(4).reshape(2, 2)
    b = np.arange(4, 8).reshape(2, 2)
    out = pytree_cat([a, b])
    assert np.array_equal(out, np.concatenate([a, b], axis=0))


def test_cat_lists_flatten():
    out = pytree_cat([[0, 1], [2, 3], [4]])
    assert out == [0, 1, 2, 3, 4]


def test_cat_tuples_recurse_elementwise():
    r0 = (torch.tensor([0.0, 1.0]), [10])
    r1 = (torch.tensor([2.0, 3.0]), [11])
    out = pytree_cat([r0, r1])
    assert isinstance(out, tuple) and len(out) == 2
    assert torch.equal(out[0], torch.tensor([0.0, 1.0, 2.0, 3.0]))  # tensor element concatenated
    assert out[1] == [10, 11]  # list element flattened


def test_cat_dicts_recurse_per_key():
    out = pytree_cat([{"x": torch.tensor([0.0]), "n": [1]}, {"x": torch.tensor([1.0]), "n": [2]}])
    assert torch.equal(out["x"], torch.tensor([0.0, 1.0]))
    assert out["n"] == [1, 2]


def test_cat_scalar_takes_first():
    # non-container, non-None leaf -> first value (all shards expected to match).
    assert pytree_cat([0.1, 0.1]) == 0.1
    assert pytree_cat(["a", "a"]) == "a"


def test_cat_tensorref_merges_spans():
    a = _meta(torch.tensor([[0.0], [1.0]]))  # 2 rows, 1 span
    b = _meta(torch.tensor([[2.0]]))  # 1 row, 1 span
    out = pytree_cat([a, b])
    assert isinstance(out, TensorRef)
    assert out.spans == a.spans + b.spans  # spans concatenated in order
    assert out.batch_size == 3
    assert torch.equal(out.materialize(backend=None), torch.tensor([[0.0], [1.0], [2.0]]))


# ══════════════════════════════════════════════════════════════════════════════
# round-trip: chunk then cat reconstructs the original
# ══════════════════════════════════════════════════════════════════════════════


def test_roundtrip_tensor():
    t = torch.arange(12).reshape(6, 2).float()
    out = pytree_cat(pytree_chunk(t, dp_size=3, batch_size=6))
    assert torch.equal(out, t)


def test_roundtrip_list():
    lst = [0, 1, 2, 3, 4, 5]
    out = pytree_cat(pytree_chunk(lst, dp_size=2, batch_size=6))
    assert out == lst


def test_roundtrip_dict():
    d = {"x": torch.arange(4).float(), "y": np.arange(4)}
    out = pytree_cat(pytree_chunk(d, dp_size=2, batch_size=4))
    assert torch.equal(out["x"], d["x"])
    assert np.array_equal(out["y"], d["y"])
