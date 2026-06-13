"""TransferQueue + Mooncake round-trip — the same put/get surface as
test_backend_roundtrip, over the RDMA Mooncake backend. Auto-skipped unless an
external Mooncake master is provided (env ``UNIRL_MOONCAKE_MASTER``) plus the
``transfer_queue`` lib and an RDMA NIC. Marker: mooncake.

To run::

    bash examples/mooncake_master.sh start
    UNIRL_MOONCAKE_MASTER=<master_addr> pytest tests/transport/test_tq_mooncake.py

The cfg below is the minimal driver bootstrap; a real master may need extra
fields (metadata_server, zero-copy sizing) — extend the ``_target_`` block to
match your deployment.
"""

from __future__ import annotations

import os

import pytest
import torch

pytest.importorskip("transfer_queue")
pytestmark = pytest.mark.mooncake


@pytest.fixture
def mooncake_transport(ray_session):
    master = os.environ.get("UNIRL_MOONCAKE_MASTER")
    if not master:
        pytest.skip("set UNIRL_MOONCAKE_MASTER to an external Mooncake master")
    from omegaconf import OmegaConf

    from unirl.distributed.tensor.backend.transfer_queue.runtime import _DEFAULT_PARTITION_ID, TransferQueueRuntime
    from unirl.distributed.tensor.backend.transfer_queue.transport import TQTransport
    from unirl.distributed.tensor.transport import TensorTransportRuntime

    cfg = OmegaConf.create(
        {
            "transfer_queue": {
                "_target_": "unirl.distributed.tensor.backend.transfer_queue.mooncake.MooncakeBackend",
                "master_server_address": master,
            }
        }
    )
    rt = TransferQueueRuntime().install()
    controller_handoff, _actor_handoff = rt.init(cfg)
    rt.create_client("driver", controller_handoff)
    t = TQTransport(rt, partition_id=_DEFAULT_PARTITION_ID)
    TensorTransportRuntime.install(t)
    yield t
    TensorTransportRuntime.clear_current()
    try:
        rt.clear_partition()
    finally:
        TransferQueueRuntime.clear_current()


def test_mooncake_put_get_roundtrip(mooncake_transport):
    t = torch.arange(12).reshape(3, 4).float()
    refs = mooncake_transport.put_batch({"a": t})
    assert torch.equal(mooncake_transport.get_batch(refs)["a"], t)


def test_mooncake_shape_padding_1d(mooncake_transport):
    t = torch.arange(5).float()  # 1-d → padded to (N, 1) on the wire, restored on fetch
    assert torch.equal(mooncake_transport.hydrate(mooncake_transport.dehydrate(t)), t)
