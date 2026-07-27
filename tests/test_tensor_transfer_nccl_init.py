from __future__ import annotations

import pytest

from unirl.distributed.tensor.backend.colocate_store import store as colocate_store
from unirl.distributed.tensor.backend.gpu_store import worker as gpu_store_worker


class _FakeProcessGroup:
    def __init__(self, _store, rank: int, world_size: int) -> None:
        self.rank = rank
        self.world_size = world_size
        self.eager_devices = []

    def eager_connect_single_device(self, device) -> None:
        self.eager_devices.append(device)


@pytest.mark.parametrize(
    ("module", "owner"),
    [
        (
            colocate_store,
            lambda: colocate_store.TensorStore(
                worker_id="worker-0",
                device="cuda:0",
            ),
        ),
        (
            gpu_store_worker,
            lambda: _gpu_store_owner(),
        ),
    ],
)
def test_tensor_transfer_process_group_is_initialized_eagerly(
    monkeypatch,
    module,
    owner,
) -> None:
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")
    monkeypatch.setattr(module.dist, "TCPStore", lambda **_kwargs: object())
    monkeypatch.setattr(module.dist, "ProcessGroupNCCL", _FakeProcessGroup)

    instance = owner()
    instance.setup_global_pg(global_rank=1, global_world_size=8)

    assert instance._global_pg.rank == 1
    assert instance._global_pg.world_size == 8
    assert instance._global_pg.eager_devices == [module.torch.device("cuda:0")]


def _gpu_store_owner():
    owner = gpu_store_worker.TensorWorker.__new__(gpu_store_worker.TensorWorker)
    owner.device = "cuda:0"
    owner._global_pg = None
    return owner
