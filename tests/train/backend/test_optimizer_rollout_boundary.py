from __future__ import annotations

import torch

import unirl.train.backend.base_backend as backend_module
from unirl.train.backend.base_backend import BaseFSDP2Backend


class _Optimizer:
    def __init__(self, parameter: torch.nn.Parameter) -> None:
        self.param_groups = [{"params": [parameter]}]
        self.state = {
            parameter: {
                "step": torch.tensor(1, dtype=torch.int64),
                "exp_avg": torch.ones(4, dtype=torch.float32),
                "exp_avg_sq": torch.ones(4, dtype=torch.float32),
            }
        }
        self.zero_grad_calls: list[bool] = []

    def zero_grad(self, *, set_to_none: bool = False) -> None:
        self.zero_grad_calls.append(set_to_none)
        for group in self.param_groups:
            for parameter in group["params"]:
                if set_to_none:
                    parameter.grad = None
                elif parameter.grad is not None:
                    parameter.grad.zero_()


def _backend() -> tuple[BaseFSDP2Backend, torch.nn.Parameter, _Optimizer]:
    parameter = torch.nn.Parameter(torch.ones(4, dtype=torch.float32))
    parameter.grad = torch.ones_like(parameter)
    optimizer = _Optimizer(parameter)
    backend = BaseFSDP2Backend.__new__(BaseFSDP2Backend)
    backend._device = torch.device("cpu")
    backend.model = torch.nn.Module()
    backend.model.register_parameter("weight", parameter)
    backend.optimizer = optimizer
    return backend, parameter, optimizer


def test_optimizer_boundary_clears_grads_and_moves_only_optimizer_state(monkeypatch) -> None:
    backend, parameter, optimizer = _backend()
    model_parameter_ids = tuple(id(value) for value in backend.model.parameters())
    model_parameter_devices = tuple(value.device for value in backend.model.parameters())
    moves: list[object] = []

    original_move = backend_module.move_optimizer_state

    def recorded_move(state_optimizer, device) -> None:
        moves.append(device)
        original_move(state_optimizer, device)

    monkeypatch.setattr(backend_module, "move_optimizer_state", recorded_move)
    park = backend.park_optimizer_state_for_rollout()

    assert optimizer.zero_grad_calls == [True]
    assert parameter.grad is None
    assert moves == ["cpu"]
    assert park["grad_bytes_cleared"] == 16.0
    assert park["optimizer_state_bytes"] == 40.0
    assert park["optimizer_state_bytes_parked"] == 40.0
    assert park["optimizer_park_host_time_s"] >= 0.0
    assert tuple(id(value) for value in backend.model.parameters()) == model_parameter_ids
    assert tuple(value.device for value in backend.model.parameters()) == model_parameter_devices

    restore = backend.restore_optimizer_state_after_rollout()
    assert moves == ["cpu", torch.device("cpu")]
    assert restore["optimizer_state_bytes_restored"] == 40.0
    assert restore["optimizer_restore_host_time_s"] >= 0.0
    assert all(value.device.type == "cpu" for state in optimizer.state.values() for value in state.values())


def test_optimizer_boundary_cleanup_is_idempotent() -> None:
    backend, parameter, optimizer = _backend()

    first_park = backend.park_optimizer_state_for_rollout()
    second_park = backend.park_optimizer_state_for_rollout()
    first_restore = backend.restore_optimizer_state_after_rollout()
    second_restore = backend.restore_optimizer_state_after_rollout()

    assert parameter.grad is None
    assert optimizer.zero_grad_calls == [True, True]
    assert first_park["grad_bytes_cleared"] == 16.0
    assert second_park["grad_bytes_cleared"] == 0.0
    assert first_restore["optimizer_state_bytes_restored"] == 40.0
    assert second_restore["optimizer_state_bytes_restored"] == 40.0
    assert all(value.device == backend._device for state in optimizer.state.values() for value in state.values())
