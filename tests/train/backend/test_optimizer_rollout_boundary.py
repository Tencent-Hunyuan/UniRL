from __future__ import annotations

import pytest
import torch

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


def _backend(
    parameter: torch.nn.Parameter,
    optimizer,
    *,
    device: torch.device,
) -> BaseFSDP2Backend:
    backend = BaseFSDP2Backend.__new__(BaseFSDP2Backend)
    backend._device = device
    backend.model = torch.nn.Module()
    backend.model.register_parameter("weight", parameter)
    backend.optimizer = optimizer
    backend._rollout_optimizer_state_restore_plan = {}
    return backend


def test_cpu_native_optimizer_state_is_never_recorded_or_moved() -> None:
    parameter = torch.nn.Parameter(torch.ones(4, dtype=torch.float32))
    parameter.grad = torch.ones_like(parameter)
    optimizer = _Optimizer(parameter)
    backend = _backend(parameter, optimizer, device=torch.device("cpu"))
    state_ids = {key: id(value) for key, value in optimizer.state[parameter].items()}
    parameter_id = id(parameter)

    first_park = backend.park_optimizer_state_for_rollout()
    second_park = backend.park_optimizer_state_for_rollout()
    first_restore = backend.restore_optimizer_state_after_rollout()
    second_restore = backend.restore_optimizer_state_after_rollout()

    assert parameter.grad is None
    assert id(parameter) == parameter_id
    assert optimizer.zero_grad_calls == [True, True]
    assert first_park["grad_bytes_cleared"] == 16.0
    assert first_park["optimizer_state_bytes"] == 40.0
    assert first_park["optimizer_state_bytes_parked"] == 0.0
    assert second_park["grad_bytes_cleared"] == 0.0
    assert second_park["optimizer_state_bytes_parked"] == 0.0
    assert first_restore["optimizer_state_bytes_restored"] == 0.0
    assert second_restore["optimizer_state_bytes_restored"] == 0.0
    assert first_restore["optimizer_state_restore_slots_pending"] == 0.0
    assert {key: id(value) for key, value in optimizer.state[parameter].items()} == state_ids


def test_partial_restore_consumes_only_successful_slots_and_is_retryable() -> None:
    parameter = torch.nn.Parameter(torch.ones(4, dtype=torch.float32))
    optimizer = _Optimizer(parameter)
    backend = _backend(parameter, optimizer, device=torch.device("cpu"))
    state = optimizer.state[parameter]
    first_slot = (id(state), "exp_avg")
    second_slot = (id(state), "exp_avg_sq")
    backend._rollout_optimizer_state_restore_plan = {
        first_slot: (state, "exp_avg", torch.device("cpu")),
        second_slot: (state, "exp_avg_sq", torch.device("cpu")),
    }
    saved_second = state["exp_avg_sq"]
    state["exp_avg_sq"] = None

    with pytest.raises(RuntimeError, match="disappeared while parked"):
        backend.restore_optimizer_state_after_rollout()
    assert first_slot not in backend._rollout_optimizer_state_restore_plan
    assert second_slot in backend._rollout_optimizer_state_restore_plan

    state["exp_avg_sq"] = saved_second
    restored = backend.restore_optimizer_state_after_rollout()
    repeated = backend.restore_optimizer_state_after_rollout()
    assert restored["optimizer_state_restore_slots_pending"] == 0.0
    assert repeated["optimizer_state_bytes_restored"] == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_adamw_round_trip_preserves_cpu_step_and_allows_next_step() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    parameter = torch.nn.Parameter(torch.arange(1, 5, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW([parameter], lr=1.0e-2, foreach=False)
    parameter.square().sum().backward()
    optimizer.step()
    state = optimizer.state[parameter]
    step = state["step"]
    moment_values = {key: state[key].detach().clone() for key in ("exp_avg", "exp_avg_sq")}
    parameter_id = id(parameter)
    parameter_device = parameter.device

    assert step.device.type == "cpu"
    assert state["exp_avg"].device == device
    assert state["exp_avg_sq"].device == device

    backend = _backend(parameter, optimizer, device=device)
    parked = backend.park_optimizer_state_for_rollout()
    assert parameter.grad is None
    assert id(parameter) == parameter_id
    assert parameter.device == parameter_device
    assert state["step"] is step
    assert state["step"].device.type == "cpu"
    assert state["exp_avg"].device.type == "cpu"
    assert state["exp_avg_sq"].device.type == "cpu"
    assert parked["optimizer_state_bytes_parked"] > 0.0

    repeated_park = backend.park_optimizer_state_for_rollout()
    assert repeated_park["optimizer_state_bytes_parked"] == 0.0

    restored = backend.restore_optimizer_state_after_rollout()
    repeated_restore = backend.restore_optimizer_state_after_rollout()
    assert state["step"] is step
    assert state["step"].device.type == "cpu"
    assert state["exp_avg"].device == device
    assert state["exp_avg_sq"].device == device
    assert restored["optimizer_state_bytes_restored"] == parked["optimizer_state_bytes_parked"]
    assert restored["optimizer_state_restore_slots_pending"] == 0.0
    assert repeated_restore["optimizer_state_bytes_restored"] == 0.0
    for key, expected in moment_values.items():
        torch.testing.assert_close(state[key], expected)

    before = parameter.detach().clone()
    parameter.square().sum().backward()
    optimizer.step()
    assert not torch.equal(parameter, before)
    assert state["step"].device.type == "cpu"


class _FailOnceToCPU(torch.Tensor):
    failed = False

    @staticmethod
    def __new__(cls, value: torch.Tensor):
        return torch.Tensor._make_subclass(cls, value, require_grad=False)

    def to(self, *args, **kwargs):
        target = kwargs.get("device", args[0] if args else None)
        if target is not None and torch.device(target).type == "cpu" and not self.failed:
            self.failed = True
            raise RuntimeError("injected optimizer-state transfer failure")
        return super().to(*args, **kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_partial_cuda_park_is_retryable_and_restores_every_recorded_slot() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    parameter = torch.nn.Parameter(torch.ones(4, device=device))
    optimizer = torch.optim.AdamW([parameter], foreach=False)
    parameter.sum().backward()
    optimizer.step()
    state = optimizer.state[parameter]
    state["exp_avg_sq"] = _FailOnceToCPU(state["exp_avg_sq"])
    backend = _backend(parameter, optimizer, device=device)

    with pytest.raises(RuntimeError, match="injected optimizer-state transfer failure"):
        backend.park_optimizer_state_for_rollout()
    assert state["step"].device.type == "cpu"
    assert state["exp_avg"].device.type == "cpu"
    assert state["exp_avg_sq"].device == device
    assert len(backend._rollout_optimizer_state_restore_plan) == 2

    parked = backend.park_optimizer_state_for_rollout()
    assert parked["optimizer_state_bytes_parked"] > 0.0
    assert state["exp_avg_sq"].device.type == "cpu"
    restored = backend.restore_optimizer_state_after_rollout()
    assert restored["optimizer_state_restore_slots_pending"] == 0.0
    assert state["step"].device.type == "cpu"
    assert state["exp_avg"].device == device
    assert state["exp_avg_sq"].device == device
