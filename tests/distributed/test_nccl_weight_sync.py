from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch import nn

from unirl.distributed.weight_sync.full.nccl import NCCLWeightSync
from unirl.trainer.base import BaseTrainer
from unirl.utils.distributed_utils import eager_connect_process_group


def _sync(**kwargs) -> NCCLWeightSync:
    backend = SimpleNamespace(
        model=nn.Linear(1, 1),
        rollout_adapter_name="default",
        expert_weight_export_transform=lambda: None,
    )
    return NCCLWeightSync(backend=backend, **kwargs)


def test_operation_timeout_must_be_finite_and_positive() -> None:
    for value in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="operation_timeout_s"):
            _sync(operation_timeout_s=value)
    for value in (True, "300"):
        with pytest.raises(TypeError, match="operation_timeout_s"):
            _sync(operation_timeout_s=value)


def test_rank_zero_phase_error_poison_sync_before_next_bucket() -> None:
    sync = _sync()
    with pytest.raises(RuntimeError, match="prepare bucket failed"):
        sync._raise_if_rank0_phase_failed("receiver OOM", phase="prepare bucket")
    assert sync._broken


def test_teardown_releases_sender_group_and_connected_state(monkeypatch) -> None:
    sync = _sync()
    group = object()
    destroyed = []
    sync._model_update_group = group
    sync._connected = True
    monkeypatch.setattr(dist, "destroy_process_group", destroyed.append)
    sync._teardown_groups()
    assert destroyed == [group]
    assert sync._model_update_group is None
    assert not sync._connected


def test_base_trainer_finish_invokes_weight_sync_cleanup() -> None:
    calls = []
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.backend = None
    trainer.weight_sync = SimpleNamespace(cleanup=lambda: calls.append("cleanup"))
    trainer.wandb_logger = None
    trainer._finish_wandb()
    assert calls == ["cleanup"]


def test_eager_connect_resolves_generic_process_group_backend() -> None:
    calls = []
    backend = SimpleNamespace(eager_connect_single_device=lambda device: calls.append(device))
    wrapper = SimpleNamespace(_get_backend=lambda device: backend)
    device = torch.device("cuda", 0)
    eager_connect_process_group(wrapper, device)
    assert calls == [device]
