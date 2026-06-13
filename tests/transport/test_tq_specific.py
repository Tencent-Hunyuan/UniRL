"""TQ-unique behaviors exercised through TQTransport's public put/get surface.

The Transfer Queue backend differs from the worker-local stores in two ways these
tests pin down:

* **Shape padding/restore.** TransferQueue stores row-wise, so ``_store_shape``
  pads a field to >=2 dims on ``put`` (0-dim → ``(1, 1)``, 1-dim ``(N,)`` →
  ``(N, 1)``) and records the original shape on the :class:`TQTensorHandle`;
  ``_restore`` reshapes back to ``orig_shape`` when the handle is resolved.
* **Column-union / batch-size grouping.** ``put_batch`` groups named tensors by
  leading-dim batch size into one ``async_put`` per distinct size, so same-size
  tensors share one originating put (``_gkey``) and column-union into a single
  ``async_get_data`` round-trip on fetch; different sizes split into separate puts.

Only correctness + the cheap, observable ``_gkey`` grouping invariant are asserted
(no RPC-count probing). Requires the ``transfer_queue`` lib and the in-Ray
``tq_simple`` runtime from the conftest.
"""

import pytest
import torch

pytest.importorskip("transfer_queue")

from unirl.distributed.tensor.backend.transfer_queue.transport import (  # noqa: E402
    TQTensorHandle,
    _restore,
    _store_shape,
)
from unirl.distributed.tensor.transport import TensorRef  # noqa: E402

pytestmark = pytest.mark.cpu


# ── shape padding / restore ──────────────────────────────────────────────────


def test_store_shape_pads_scalar_and_1d():
    # The padding contract _store_shape enforces: a per-row slice must stay >=1-dim,
    # so a 0-dim field is padded to (1, 1) and a 1-dim (N,) to (N, 1). >=2-dim is
    # left untouched.
    assert _store_shape(torch.tensor(7.0)).shape == (1, 1)
    assert _store_shape(torch.arange(5)).shape == (5, 1)
    assert _store_shape(torch.zeros(3, 4)).shape == (3, 4)


def test_restore_reverses_padding():
    # _restore undoes the pad back to the recorded orig_shape (a no-op once shapes match).
    padded_scalar = torch.tensor(7.0).reshape(1, 1)
    assert _restore(padded_scalar, ()).shape == ()
    padded_1d = torch.arange(5).unsqueeze(1)
    assert _restore(padded_1d, (5,)).shape == (5,)
    already = torch.zeros(3, 4)
    assert _restore(already, (3, 4)) is already  # unchanged when shape already matches


def test_scalar_handle_resolves_to_original_shape(tq_simple_transport):
    # SOURCE NOTE: a 0-dim original round-trips its ORIGINAL shape at the handle
    # resolution layer — put records orig_shape=() and _resolve_handles restores it.
    # We assert restoration there (and via handle.local()) rather than through get():
    # get() slices b[start:stop] off the resolved base, which is illegal once the
    # base has been restored to 0-dim, so the row-slice path does not apply to scalars.
    t = torch.tensor(3.5)
    handle = tq_simple_transport.put(t)
    assert isinstance(handle, TQTensorHandle)
    assert handle.orig_shape == ()
    base = tq_simple_transport._resolve_handles([handle])[0]
    assert base.shape == ()
    assert torch.equal(base, t)
    assert torch.equal(handle.local(), t)  # local() == _resolve_handles([self])[0]


def test_1d_tensor_roundtrips_through_get(tq_simple_transport):
    # A 1-dim field DOES round-trip through get/hydrate: orig_shape=(N,) keeps a
    # sliceable leading dim after restore ((N,1) -> (N,)).
    t = torch.arange(6, dtype=torch.float32)
    ref = tq_simple_transport.dehydrate(t)
    assert isinstance(ref, TensorRef)
    assert ref.shape == (6,)
    assert torch.equal(tq_simple_transport.get(ref.spans), t)
    assert torch.equal(tq_simple_transport.hydrate(ref), t)


def test_2d_tensor_roundtrips_through_get(tq_simple_transport):
    t = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    ref = tq_simple_transport.dehydrate(t)
    assert torch.equal(tq_simple_transport.get(ref.spans), t)


# ── column-union / batch-size grouping ───────────────────────────────────────


def test_put_batch_same_leading_dim_shares_one_put(tq_simple_transport):
    # Same leading dim → one async_put → both handles share global_indexes, so they
    # column-union into a single get. _gkey() equality is the cheap, observable proxy
    # for "dedup to one fetch".
    a = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    b = torch.arange(100, 108, dtype=torch.float32).reshape(4, 2)
    refs = tq_simple_transport.put_batch({"a": a, "b": b})
    assert refs["a"].spans[0].handle._gkey() == refs["b"].spans[0].handle._gkey()
    got = tq_simple_transport.get_batch(refs)
    assert torch.equal(got["a"], a)
    assert torch.equal(got["b"], b)


def test_put_batch_mixed_leading_dims_split_and_each_roundtrips(tq_simple_transport):
    # Mixed leading dims must split into separate puts (a TensorDict needs one
    # uniform batch_size). Different-size handles get disjoint global_indexes →
    # distinct _gkey → fetched separately; same-size pair still shares a _gkey.
    a = torch.arange(8, dtype=torch.float32).reshape(4, 2)  # leading dim 4
    b = torch.arange(100, 108, dtype=torch.float32).reshape(4, 2)  # leading dim 4
    c = torch.arange(6, dtype=torch.float32).reshape(2, 3)  # leading dim 2
    refs = tq_simple_transport.put_batch({"a": a, "b": b, "c": c})

    gk_a = refs["a"].spans[0].handle._gkey()
    gk_b = refs["b"].spans[0].handle._gkey()
    gk_c = refs["c"].spans[0].handle._gkey()
    assert gk_a == gk_b  # same leading dim → one put-group
    assert gk_a != gk_c  # different leading dim → separate put-group

    got = tq_simple_transport.get_batch(refs)
    assert torch.equal(got["a"], a)
    assert torch.equal(got["b"], b)
    assert torch.equal(got["c"], c)


def test_get_batch_unions_then_slices_per_key(tq_simple_transport):
    # get_batch flattens every key's spans into ONE _resolve_handles call (column-union
    # per put-group) and slices each key back out; assert each key is correct end-to-end.
    tensors = {
        "x": torch.arange(9, dtype=torch.float32).reshape(3, 3),
        "y": torch.arange(50, 59, dtype=torch.float32).reshape(3, 3),
        "z": torch.arange(6, dtype=torch.float32),  # 1-dim, restored to (6,)
    }
    refs = tq_simple_transport.put_batch(tensors)
    got = tq_simple_transport.get_batch(refs)
    assert torch.equal(got["x"], tensors["x"])
    assert torch.equal(got["y"], tensors["y"])
    assert torch.equal(got["z"], tensors["z"])
