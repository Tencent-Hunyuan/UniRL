from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest
import torch

from unirl.train.stack import TrainStepResult
from unirl.train.unified_model_stack import UnifiedModelTrainStack, _collect_unified_train_results


class _Algorithm:
    prepares_update_batch = False


class _Backend:
    _device = torch.device("cpu")
    _persistent_cpu_offload = False

    def __init__(self, events: list[str], *, fail_restore_once: bool = False) -> None:
        self.events = events
        self.parked = False
        self.fail_restore_once = fail_restore_once
        self.restore_calls = 0
        self.parameter = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
        self.parameter_snapshot = self.parameter.detach().clone()
        self.parameter_id = id(self.parameter)
        self.step = torch.tensor(1, dtype=torch.int64)
        self.optimizer = SimpleNamespace(param_groups=[{"params": [self.parameter], "lr": 1.0e-6}])
        self.scheduler = None

    def zero_grad(self) -> None:
        self.events.append("zero_grad")
        self.parameter.grad = None

    def park_optimizer_state_for_rollout(self):
        self.events.append("park")
        assert not self.parked
        self.parked = True
        self._assert_parameter_unchanged()
        assert self.step.device.type == "cpu"
        return {
            "optimizer_state_bytes": 40.0,
            "optimizer_state_bytes_parked": 32.0,
            "optimizer_park_host_time_s": 2.0,
        }

    def restore_optimizer_state_after_rollout(self):
        self.events.append("restore")
        self.restore_calls += 1
        assert self.parked
        if self.fail_restore_once:
            self.fail_restore_once = False
            raise RuntimeError("injected restore failure")
        self.parked = False
        self._assert_parameter_unchanged()
        assert self.step.device.type == "cpu"
        return {
            "optimizer_state_bytes_restored": 32.0,
            "optimizer_state_restore_slots_pending": 0.0,
            "optimizer_restore_host_time_s": 1.0,
        }

    def optimizer_step(self, *, max_grad_norm: float) -> float:
        del max_grad_norm
        self.events.append("optimizer_step")
        assert not self.parked
        assert self.step.device.type == "cpu"
        self._assert_parameter_unchanged()
        return 0.5

    def _assert_parameter_unchanged(self) -> None:
        assert id(self.parameter) == self.parameter_id
        assert self.parameter.device.type == "cpu"
        torch.testing.assert_close(self.parameter, self.parameter_snapshot)


def _stack(backend: _Backend) -> UnifiedModelTrainStack:
    stack = object.__new__(UnifiedModelTrainStack)
    stack.fsdp_backend = backend
    stack.algorithms = {"ar": _Algorithm(), "image": _Algorithm()}
    stack.max_grad_norm = 1.0
    stack.num_updates_per_batch = 2
    stack.park_optimizer_state_during_train = True
    stack.empty_cache_after_optimizer = False
    stack.cuda_peak_telemetry = False
    return stack


def _result() -> TrainStepResult:
    return TrainStepResult(
        loss=1.0,
        grad_norm=0.0,
        lr=0.0,
        has_backward=True,
        micros=[],
        metrics={},
    )


def _install_backward(stack: UnifiedModelTrainStack, events: list[str], *, fail_image_once: bool = False) -> None:
    failed = False

    def backward(self, name, resp_track, micro_slices, *, training_progress):
        nonlocal failed
        del self, resp_track, micro_slices, training_progress
        events.append(f"backward_{name}")
        assert stack.fsdp_backend.parked
        if name == "image" and fail_image_once and not failed:
            failed = True
            raise RuntimeError("injected backward failure")
        return _result(), True

    stack._backward_track = MethodType(backward, stack)


def _run_update(stack: UnifiedModelTrainStack, update_index: int = 0):
    return stack._train_one_step(
        {"ar": object(), "image": object()},
        {"ar": [(0, 1)], "image": [(0, 1)]},
        training_progress=0.0,
        update_index=update_index,
    )


def test_train_optimizer_parking_wraps_two_updates_without_moving_parameters() -> None:
    events: list[str] = []
    backend = _Backend(events)
    stack = _stack(backend)
    _install_backward(stack, events)

    first = _run_update(stack, update_index=0)
    second = _run_update(stack, update_index=1)

    expected_update = ["zero_grad", "park", "backward_ar", "backward_image", "restore", "optimizer_step"]
    assert events == expected_update * 2
    assert events[-1] == "optimizer_step"
    assert backend.parked is False
    backend._assert_parameter_unchanged()
    assert backend.step.device.type == "cpu"
    for result in (first["image"], second["image"]):
        assert result.metrics["train_optimizer_state_bytes"] == 40.0
        assert result.metrics["train_optimizer_state_bytes_parked"] == 32.0
        assert result.metrics["train_optimizer_state_bytes_restored"] == 32.0
        assert result.metrics["train_optimizer_state_restore_slots_pending"] == 0.0
        assert result.metrics["train_optimizer_park_host_time_s"] == 2.0
        assert result.metrics["train_optimizer_restore_host_time_s"] == 1.0


def test_train_optimizer_parking_rejects_persistent_cpu_offload() -> None:
    with pytest.raises(ValueError, match="cpu_offload=false"):
        UnifiedModelTrainStack(
            fsdp_backend=SimpleNamespace(_persistent_cpu_offload=True),
            ar_algorithm=object(),
            image_algorithm=object(),
            micro_batch_size=1,
            max_grad_norm=1.0,
            park_optimizer_state_during_train=True,
        )


def test_backward_failure_restores_optimizer_and_allows_retry() -> None:
    events: list[str] = []
    backend = _Backend(events)
    stack = _stack(backend)
    _install_backward(stack, events, fail_image_once=True)

    with pytest.raises(RuntimeError, match="injected backward failure"):
        _run_update(stack)

    assert events == ["zero_grad", "park", "backward_ar", "backward_image", "zero_grad", "restore"]
    assert backend.parked is False

    _run_update(stack)
    assert events[-6:] == ["zero_grad", "park", "backward_ar", "backward_image", "restore", "optimizer_step"]
    assert backend.parked is False


def test_restore_failure_is_retried_in_cleanup_and_next_update_can_run() -> None:
    events: list[str] = []
    backend = _Backend(events, fail_restore_once=True)
    stack = _stack(backend)
    _install_backward(stack, events)

    with pytest.raises(RuntimeError, match="injected restore failure"):
        _run_update(stack)

    assert events == [
        "zero_grad",
        "park",
        "backward_ar",
        "backward_image",
        "restore",
        "zero_grad",
        "restore",
    ]
    assert "optimizer_step" not in events
    assert backend.restore_calls == 2
    assert backend.parked is False

    _run_update(stack)
    assert events[-6:] == ["zero_grad", "park", "backward_ar", "backward_image", "restore", "optimizer_step"]


def test_train_optimizer_metrics_sum_dp_shards_and_keep_critical_path_time() -> None:
    class Rank:
        tp_rank = 0
        is_pipeline_last_stage = True
        sp_rank = 0

    class WorkerGroup:
        rank_infos = [Rank(), Rank()]

    def result(bytes_by_update: tuple[float, float], times: tuple[float, float]) -> dict[str, TrainStepResult]:
        updates = tuple(
            {
                "train_optimizer_state_bytes_parked": byte_count,
                "train_optimizer_state_bytes_restored": byte_count,
                "train_optimizer_state_restore_slots_pending": 0.0,
                "train_optimizer_park_host_time_s": host_time,
                "loss": float(index),
            }
            for index, (byte_count, host_time) in enumerate(zip(bytes_by_update, times))
        )
        return {
            "image": TrainStepResult(
                loss=1.0,
                grad_norm=0.5,
                lr=1.0e-6,
                has_backward=True,
                micros=[],
                metrics={
                    "train_optimizer_state_bytes_parked": sum(bytes_by_update) / 2.0,
                    "train_optimizer_state_bytes_restored": sum(bytes_by_update) / 2.0,
                    "train_optimizer_state_restore_slots_pending": 0.0,
                    "train_optimizer_park_host_time_s": sum(times) / 2.0,
                },
                per_update=updates,
            )
        }

    collected = _collect_unified_train_results(
        WorkerGroup(),
        [result((10.0, 12.0), (2.0, 5.0)), result((20.0, 28.0), (4.0, 3.0))],
    )

    assert collected["image"].per_update == (
        {
            "train_optimizer_state_bytes_parked": 30.0,
            "train_optimizer_state_bytes_restored": 30.0,
            "train_optimizer_state_restore_slots_pending": 0.0,
            "train_optimizer_park_host_time_s": 4.0,
            "loss": 0.0,
        },
        {
            "train_optimizer_state_bytes_parked": 40.0,
            "train_optimizer_state_bytes_restored": 40.0,
            "train_optimizer_state_restore_slots_pending": 0.0,
            "train_optimizer_park_host_time_s": 5.0,
            "loss": 1.0,
        },
    )
    assert collected["image"].metrics == {
        "train_optimizer_state_bytes_parked": 35.0,
        "train_optimizer_state_bytes_restored": 35.0,
        "train_optimizer_state_restore_slots_pending": 0.0,
        "train_optimizer_park_host_time_s": 4.5,
    }
