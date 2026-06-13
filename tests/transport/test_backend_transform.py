"""Backend-agnostic transform/reshape/permute round-trips over a live transport.

Every :class:`TensorTransport` shares the same compute-proxy surface at the ABC /
TensorRef level: ``transport.transform(ref, fn)`` (the ABC default hydrate → fn →
dehydrate), and ``TensorRef.reshape`` / ``.permute`` which route through
``transform`` via the active ``TensorTransportRuntime`` the ``transport`` fixture
installs. These tests parametrize over every driver-resolvable backend on the host
(``colocate`` / ``tq_simple``) and assert the produced ref *materializes to the
right values*, not just the right shape.

The remote-compute path (``tensor_op`` / ``get_cpu``) only exists on
:class:`WorkerLocalTransport` backends — the global TQ transport implements none of
it — so those assertions are guarded by ``isinstance(transport, WorkerLocalTransport)``.
"""

import pytest
import torch

from unirl.distributed.tensor.transport import TensorRef, WorkerLocalTransport

pytestmark = pytest.mark.cpu


def _dehydrate(transport, t):
    """Round a CPU tensor into a TensorRef through the live backend's put surface."""
    ref = transport.dehydrate(t)
    assert isinstance(ref, TensorRef)
    return ref


@pytest.mark.cpu
def test_transform_roundtrips_values(transport):
    # transform = get → fn → put; the result ref must materialize to fn(tensor).
    t = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    ref = _dehydrate(transport, t)
    out = transport.transform(ref, lambda x: x * 2 + 1)
    assert isinstance(out, TensorRef)
    got = transport.get(out.spans)
    assert torch.equal(got, t * 2 + 1)


@pytest.mark.cpu
def test_transform_result_hydrates(transport):
    # hydrate(ref) on a bare TensorRef == materialize(backend=self): same tensor as get.
    t = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    ref = _dehydrate(transport, t)
    out = transport.transform(ref, lambda x: x - 5)
    got = transport.hydrate(out)
    assert torch.equal(got, t - 5)


@pytest.mark.cpu
def test_transform_can_change_shape(transport):
    # fn may reshape the tensor; the new ref's shape/values follow fn's output.
    t = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    ref = _dehydrate(transport, t)
    out = transport.transform(ref, lambda x: x.reshape(2, 4))
    assert out.shape == (2, 4)
    assert torch.equal(transport.get(out.spans), t.reshape(2, 4))


@pytest.mark.cpu
def test_ref_reshape_via_runtime(transport):
    # TensorRef.reshape → transform via TensorTransportRuntime.current() (installed
    # by the fixture); shape AND values must survive the reshape.
    t = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    ref = _dehydrate(transport, t)
    out = ref.reshape(2, 6)
    assert isinstance(out, TensorRef)
    assert out.shape == (2, 6)
    assert torch.equal(transport.get(out.spans), t.reshape(2, 6))


@pytest.mark.cpu
def test_ref_permute_via_runtime(transport):
    t = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    ref = _dehydrate(transport, t)
    out = ref.permute(2, 0, 1)
    assert isinstance(out, TensorRef)
    assert tuple(out.shape) == (4, 2, 3)
    assert torch.equal(transport.get(out.spans), t.permute(2, 0, 1))


@pytest.mark.cpu
def test_transform_preserves_dtype(transport):
    t = torch.arange(6, dtype=torch.int64).reshape(2, 3)
    ref = _dehydrate(transport, t)
    out = transport.transform(ref, lambda x: x + 10)
    got = transport.get(out.spans)
    assert got.dtype == torch.int64
    assert torch.equal(got, t + 10)


@pytest.mark.cpu
def test_tensor_op_reshape_worker_local_only(transport):
    # tensor_op is a WorkerLocalTransport-only capability (the global TQ backend has
    # no on-worker compute path), so guard the remote-compute assertion on locality.
    if not isinstance(transport, WorkerLocalTransport):
        pytest.skip("tensor_op is worker-local only; not on the global TQ backend")
    t = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    handle = transport.put(t)
    new_handle = transport.tensor_op(handle, "reshape", (2, 6))
    # get_cpu resolves the produced handle back to a CPU tensor.
    got = transport.get_cpu(new_handle)
    assert tuple(got.shape) == (2, 6)
    assert torch.equal(got, t.reshape(2, 6))


@pytest.mark.cpu
def test_get_cpu_worker_local_only(transport):
    if not isinstance(transport, WorkerLocalTransport):
        pytest.skip("get_cpu is worker-local only; not on the global TQ backend")
    t = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    handle = transport.put(t)
    got = transport.get_cpu(handle)
    assert torch.equal(got, t)
