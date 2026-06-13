"""dehydrate / hydrate / session / put_batch / get_batch round-trips.

Parametrized over every driver-resolvable backend available on this host
(colocate, tq_simple, [tq_mooncake]) via the ``transport`` fixture — one source
of truth for "a transport round-trips a tensor unchanged". Test IDs read
``[colocate]`` / ``[tq_simple]``. Marker: cpu.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch

from unirl.distributed.tensor import Batch, concat_field
from unirl.distributed.tensor.transport import TensorRef

pytestmark = pytest.mark.cpu


@dataclass
class _Pair(Batch):
    x: Any = concat_field(default=None)
    y: Any = concat_field(default=None)


def test_put_get_batch_roundtrip(transport):
    t1 = torch.arange(12).reshape(3, 4).float()
    t2 = torch.arange(100, 106).reshape(3, 2).float()
    refs = transport.put_batch({"a": t1, "b": t2})
    assert transport.is_ref(refs["a"]) and isinstance(refs["a"], TensorRef)
    assert refs["a"].batch_size == 3
    out = transport.get_batch(refs)
    assert torch.equal(out["a"], t1) and torch.equal(out["b"], t2)


def test_dehydrate_hydrate_tensor(transport):
    t = torch.randn(5, 7)
    ref = transport.dehydrate(t)
    assert isinstance(ref, TensorRef) and transport.is_ref(ref)
    assert torch.equal(transport.hydrate(ref), t)


def test_get_resolves_spans(transport):
    t = torch.arange(24).reshape(6, 4).float()
    ref = transport.dehydrate(t)
    # get over the raw spans is the universal materialization path
    assert torch.equal(transport.get(ref.spans), t)


def test_dehydrate_hydrate_dict_mixed(transport):
    d = {"a": torch.randn(2, 3), "b": torch.randn(2, 5), "scalar": 7, "name": "x"}
    a0, b0 = d["a"].clone(), d["b"].clone()
    transport.dehydrate(d)
    assert isinstance(d["a"], TensorRef) and isinstance(d["b"], TensorRef)
    assert d["scalar"] == 7 and d["name"] == "x"  # non-tensors pass through
    transport.hydrate(d)
    assert torch.equal(d["a"], a0) and torch.equal(d["b"], b0)


def test_dehydrate_hydrate_batch(transport):
    b = _Pair(x=torch.arange(6).reshape(3, 2).float(), y=torch.arange(9).reshape(3, 3).float())
    xs, ys = b.x.clone(), b.y.clone()
    transport.dehydrate(b)
    assert isinstance(b.x, TensorRef) and isinstance(b.y, TensorRef)
    transport.hydrate(b)
    assert torch.equal(b.x, xs) and torch.equal(b.y, ys)


def test_hydrate_fields_filter(transport):
    d = {"keep": torch.randn(2, 2), "drop": torch.randn(2, 2)}
    transport.dehydrate(d)
    transport.hydrate(d, fields={"keep"})
    assert torch.is_tensor(d["keep"]) and isinstance(d["drop"], TensorRef)


def test_session_batches_dehydrate(transport):
    d = {"a": torch.randn(2, 3), "b": torch.randn(2, 4)}
    a0, b0 = d["a"].clone(), d["b"].clone()
    with transport.session() as sess:
        sess.dehydrate(d)
    assert isinstance(d["a"], TensorRef) and isinstance(d["b"], TensorRef)
    transport.hydrate(d)
    assert torch.equal(d["a"], a0) and torch.equal(d["b"], b0)


def test_roundtrip_1d(transport):
    t = torch.arange(9).float()
    assert torch.equal(transport.hydrate(transport.dehydrate(t)), t)


def test_duplicate_tensor_in_tree(transport):
    # The same tensor object appearing twice must round-trip both positions.
    t = torch.arange(8).reshape(2, 4).float()
    d = {"a": t, "b": t}
    transport.dehydrate(d)
    transport.hydrate(d)
    assert torch.equal(d["a"], torch.arange(8).reshape(2, 4).float())
    assert torch.equal(d["b"], torch.arange(8).reshape(2, 4).float())
