from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest
import torch

from unirl.train.stack import TrainStepResult
from unirl.train.unified_model_stack import UnifiedModelTrainStack, _collect_unified_train_results
from unirl.trainer.unified_model import UnifiedModelTrainer
from unirl.types.rollout_resp import RolloutTrack
from unirl.types.segments import make_image_segment


class _Algorithm:
    def __init__(self, events: list[str] | None = None, *, prepares_update_batch: bool = False) -> None:
        self.events = events
        self.prepares_update_batch = prepares_update_batch

    def finish_update_batch(self, *, succeeded: bool) -> None:
        if self.events is not None:
            self.events.append(f"finish_image_{succeeded}")


class _Backend:
    _device = torch.device("cpu")
    _persistent_cpu_offload = False

    def __init__(
        self,
        events: list[str],
        *,
        fail_restore_times: int = 0,
        fail_optimizer_step: bool = False,
        fail_reclaim: bool = False,
    ) -> None:
        self.events = events
        self.parked = False
        self.fail_restore_times = fail_restore_times
        self.fail_optimizer_step = fail_optimizer_step
        self.fail_reclaim = fail_reclaim
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
        if self.fail_restore_times:
            self.fail_restore_times -= 1
            raise RuntimeError("injected restore failure")
        if not self.parked:
            return {
                "optimizer_state_bytes_restored": 0.0,
                "optimizer_state_restore_slots_pending": 0.0,
                "optimizer_restore_host_time_s": 0.0,
            }
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
        if self.fail_optimizer_step:
            raise RuntimeError("injected optimizer step failure")
        return 0.5

    def reclaim_cuda_allocator(self) -> None:
        self.events.append("reclaim")
        if self.fail_reclaim:
            raise RuntimeError("injected reclaim failure")

    def on_rollout_end(self) -> None:
        self.events.append("rollout_end")

    def _assert_parameter_unchanged(self) -> None:
        assert id(self.parameter) == self.parameter_id
        assert self.parameter.device.type == "cpu"
        torch.testing.assert_close(self.parameter, self.parameter_snapshot)


def _stack(backend: _Backend) -> UnifiedModelTrainStack:
    stack = object.__new__(UnifiedModelTrainStack)
    stack.fsdp_backend = backend
    stack.algorithms = {"ar": _Algorithm(), "image": _Algorithm()}
    stack.micro_batch_size = 1
    stack.max_grad_norm = 1.0
    stack.num_updates_per_batch = 2
    stack.park_optimizer_state_during_train = True
    stack.empty_cache_after_optimizer = False
    stack.cuda_peak_telemetry = False
    return stack


def _result(*, has_backward: bool = True) -> TrainStepResult:
    return TrainStepResult(
        loss=1.0,
        grad_norm=0.0,
        lr=0.0,
        has_backward=has_backward,
        micros=[],
        metrics={},
    )


def _install_backward(
    stack: UnifiedModelTrainStack,
    events: list[str],
    *,
    fail_image_once: bool = False,
    has_backward: bool = True,
) -> None:
    failed = False

    def backward(self, name, resp_track, micro_slices, *, training_progress):
        nonlocal failed
        del self, resp_track, micro_slices, training_progress
        events.append(f"backward_{name}")
        assert stack.fsdp_backend.parked
        if name == "image" and fail_image_once and not failed:
            failed = True
            raise RuntimeError("injected backward failure")
        return _result(has_backward=has_backward), has_backward

    stack._backward_track = MethodType(backward, stack)


def _run_update(stack: UnifiedModelTrainStack, update_index: int = 0):
    return stack._train_one_step(
        {"ar": object(), "image": object()},
        {"ar": [(0, 1)], "image": [(0, 1)]},
        training_progress=0.0,
        update_index=update_index,
    )


def _track(prefix: str, count: int = 2) -> RolloutTrack:
    return RolloutTrack(
        sample_ids=[f"{prefix}-{index}" for index in range(count)],
        conditions={},
        segment=make_image_segment(latents=torch.zeros(count, 1, 1, 1)),
        advantages=torch.ones(count),
    )


def test_train_optimizer_parking_wraps_two_updates_without_moving_parameters() -> None:
    events: list[str] = []
    backend = _Backend(events)
    stack = _stack(backend)
    _install_backward(stack, events)

    first = _run_update(stack, update_index=0)
    second = _run_update(stack, update_index=1)

    expected_update = [
        "zero_grad",
        "park",
        "backward_ar",
        "backward_image",
        "reclaim",
        "restore",
        "optimizer_step",
    ]
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

    assert events == ["zero_grad", "park", "backward_ar", "backward_image", "zero_grad", "reclaim", "restore"]
    assert backend.parked is False

    _run_update(stack)
    assert events[-7:] == [
        "zero_grad",
        "park",
        "backward_ar",
        "backward_image",
        "reclaim",
        "restore",
        "optimizer_step",
    ]
    assert backend.parked is False


def test_restore_failure_is_retried_in_cleanup_and_next_update_can_run() -> None:
    events: list[str] = []
    backend = _Backend(events, fail_restore_times=1)
    stack = _stack(backend)
    _install_backward(stack, events)

    with pytest.raises(RuntimeError, match="injected restore failure"):
        _run_update(stack)

    assert events == [
        "zero_grad",
        "park",
        "backward_ar",
        "backward_image",
        "reclaim",
        "restore",
        "zero_grad",
        "reclaim",
        "restore",
    ]
    assert "optimizer_step" not in events
    assert backend.restore_calls == 2
    assert backend.parked is False

    _run_update(stack)
    assert events[-7:] == [
        "zero_grad",
        "park",
        "backward_ar",
        "backward_image",
        "reclaim",
        "restore",
        "optimizer_step",
    ]


def test_single_update_reclaims_allocator_before_restore() -> None:
    events: list[str] = []
    backend = _Backend(events)
    stack = _stack(backend)
    stack.num_updates_per_batch = 1
    _install_backward(stack, events)

    _run_update(stack)

    assert events == [
        "zero_grad",
        "park",
        "backward_ar",
        "backward_image",
        "reclaim",
        "restore",
        "optimizer_step",
    ]


def test_no_backward_releases_prepared_state_before_restore() -> None:
    events: list[str] = []
    backend = _Backend(events)
    stack = _stack(backend)
    stack.algorithms = {
        "ar": _Algorithm(),
        "image": _Algorithm(events, prepares_update_batch=True),
    }
    _install_backward(stack, events, has_backward=False)

    def prepare(self, name, track, slices, **kwargs):
        del self, track, slices, kwargs
        events.append(f"prepare_{name}")

    stack._prepare_update_batch = MethodType(prepare, stack)
    results = _run_update(stack)

    assert events == [
        "zero_grad",
        "park",
        "backward_ar",
        "prepare_image",
        "backward_image",
        "finish_image_True",
        "zero_grad",
        "reclaim",
        "restore",
    ]
    assert results["image"].grad_norm == 0.0
    assert backend.parked is False


def test_optimizer_step_failure_cleans_grads_without_reparking_or_masking() -> None:
    events: list[str] = []
    backend = _Backend(events, fail_optimizer_step=True)
    stack = _stack(backend)
    _install_backward(stack, events)

    with pytest.raises(RuntimeError, match="injected optimizer step failure"):
        _run_update(stack)

    assert events == [
        "zero_grad",
        "park",
        "backward_ar",
        "backward_image",
        "reclaim",
        "restore",
        "optimizer_step",
        "zero_grad",
        "reclaim",
    ]
    assert backend.parked is False


def test_cleanup_failures_do_not_mask_backward_and_leave_state_safely_parked(caplog) -> None:
    events: list[str] = []
    backend = _Backend(events, fail_restore_times=10)
    stack = _stack(backend)

    class FailingCleanup(_Algorithm):
        def finish_update_batch(self, *, succeeded: bool) -> None:
            events.append(f"finish_image_{succeeded}")
            raise RuntimeError("injected finish failure")

    stack.algorithms = {
        "ar": _Algorithm(),
        "image": FailingCleanup(events, prepares_update_batch=True),
    }
    _install_backward(stack, events, fail_image_once=True)

    def prepare(self, name, track, slices, **kwargs):
        del self, track, slices, kwargs
        events.append(f"prepare_{name}")

    stack._prepare_update_batch = MethodType(prepare, stack)
    with pytest.raises(RuntimeError, match="injected backward failure"):
        _run_update(stack)

    assert events == [
        "zero_grad",
        "park",
        "backward_ar",
        "prepare_image",
        "backward_image",
        "finish_image_False",
        "zero_grad",
        "reclaim",
        "restore",
    ]
    assert backend.parked is True
    assert "injected finish failure" in caplog.text
    assert "injected restore failure" in caplog.text


def test_restore_phase_telemetry_is_ordered_around_restore_and_step() -> None:
    events: list[str] = []
    backend = _Backend(events)
    stack = _stack(backend)
    stack.cuda_peak_telemetry = True
    _install_backward(stack, events)

    def phase_metrics(self, phase: str):
        del self
        events.append(f"memory_{phase}")
        return {
            f"cuda_{phase}_allocated_gb": float(len(events)),
            f"cuda_{phase}_reserved_gb": float(len(events) + 1),
        }

    stack._cuda_phase_memory_metrics = MethodType(phase_metrics, stack)
    stack._cuda_peak_memory_metrics = MethodType(
        lambda self: {"cuda_peak_allocated_gb": 90.0, "cuda_peak_reserved_gb": 91.0}, stack
    )
    results = _run_update(stack)

    assert events == [
        "zero_grad",
        "park",
        "backward_ar",
        "backward_image",
        "reclaim",
        "memory_pre_optimizer_restore",
        "restore",
        "memory_post_optimizer_restore",
        "optimizer_step",
        "memory_post_optimizer_step",
    ]
    metrics = results["image"].metrics
    for phase in ("pre_optimizer_restore", "post_optimizer_restore", "post_optimizer_step"):
        assert f"cuda_{phase}_allocated_gb" in metrics
        assert f"cuda_{phase}_reserved_gb" in metrics
    assert metrics["cuda_peak_reserved_gb"] == 91.0


def test_peak_counter_reset_precedes_track_hydration_and_anchor(monkeypatch) -> None:
    events: list[str] = []
    backend = _Backend(events)
    stack = _stack(backend)
    stack.num_updates_per_batch = 1
    stack.cuda_peak_telemetry = True
    stack._reset_cuda_peak_memory_stats = MethodType(lambda self: events.append("reset_peak"), stack)

    original_to_device = RolloutTrack.to_device

    def tracked_to_device(self, device):
        events.append(f"hydrate_{self.sample_ids[0].split('-')[0]}")
        return original_to_device(self, device)

    monkeypatch.setattr(RolloutTrack, "to_device", tracked_to_device)

    def prepare_anchor(self, name, track):
        del self, track
        events.append(f"anchor_{name}")

    def train_one_step(self, *args, **kwargs):
        del self, args, kwargs
        return {"ar": _result(), "image": _result()}

    stack.prepare_segment = MethodType(prepare_anchor, stack)
    stack._train_one_step = MethodType(train_one_step, stack)
    stack.train_track(_track("ar"), _track("image"), training_progress=0.0)

    assert events[:5] == ["reset_peak", "hydrate_ar", "hydrate_image", "anchor_ar", "anchor_image"]


def test_update_peaks_reset_per_step_while_train_window_keeps_anchor_and_all_updates() -> None:
    events: list[str] = []
    backend = _Backend(events)
    stack = _stack(backend)
    stack.cuda_peak_telemetry = True
    stack._reset_cuda_peak_memory_stats = MethodType(lambda self: events.append("reset_peak"), stack)
    update_peaks = iter(((60.0, 61.0), (70.0, 71.0)))
    window_peaks = iter(((50.0, 51.0), (90.0, 91.0), (80.0, 81.0)))

    def update_peak_metrics(self):
        del self
        allocated, reserved = next(update_peaks)
        return {"cuda_peak_allocated_gb": allocated, "cuda_peak_reserved_gb": reserved}

    def window_peak_metrics(self):
        del self
        allocated, reserved = next(window_peaks)
        return {
            "cuda_train_window_peak_allocated_gb": allocated,
            "cuda_train_window_peak_reserved_gb": reserved,
        }

    stack._cuda_peak_memory_metrics = MethodType(update_peak_metrics, stack)
    stack._cuda_train_window_peak_memory_metrics = MethodType(window_peak_metrics, stack)
    stack._cuda_phase_memory_metrics = MethodType(lambda self, phase: {}, stack)

    def prepare_anchor(self, name, track):
        del self, track
        events.append(f"anchor_{name}")

    stack.prepare_segment = MethodType(prepare_anchor, stack)
    _install_backward(stack, events)
    results = stack.train_track(_track("ar", 4), _track("image", 4), training_progress=0.0)

    assert [event for event in events if event == "reset_peak"] == ["reset_peak"] * 3
    assert events[:3] == ["reset_peak", "anchor_ar", "anchor_image"]
    assert [update["cuda_peak_reserved_gb"] for update in results["image"].per_update] == [61.0, 71.0]
    assert "cuda_train_window_peak_reserved_gb" not in results["image"].per_update[0]
    assert results["image"].per_update[1]["cuda_train_window_peak_reserved_gb"] == 91.0
    assert results["image"].metrics["cuda_peak_reserved_gb"] == 71.0
    assert results["image"].metrics["cuda_train_window_peak_allocated_gb"] == 90.0
    assert results["image"].metrics["cuda_train_window_peak_reserved_gb"] == 91.0


def test_anchor_failure_releases_anchor_state_before_restoring_inherited_optimizer() -> None:
    events: list[str] = []
    backend = _Backend(events)
    backend.park_optimizer_state_for_rollout()
    events.clear()

    class FailingAnchor(_Algorithm):
        prepares_anchor_plan = True

        def prepare_anchor_batch(self, *, updates) -> None:
            del updates
            events.append("prepare_anchor_image")
            raise RuntimeError("injected anchor failure")

        def finish_anchor_batch(self, *, succeeded: bool) -> None:
            events.append(f"finish_anchor_image_{succeeded}")

    stack = _stack(backend)
    stack.algorithms = {"ar": _Algorithm(), "image": FailingAnchor()}
    stack.prepare_segment = MethodType(lambda self, name, track: events.append(f"anchor_{name}"), stack)

    with pytest.raises(RuntimeError, match="injected anchor failure"):
        stack.train_track(
            _track("ar"),
            _track("image"),
            training_progress=0.0,
            optimizer_state_already_parked=True,
        )

    assert events == [
        "anchor_ar",
        "prepare_anchor_image",
        "finish_anchor_image_False",
        "zero_grad",
        "reclaim",
        "restore",
    ]
    assert backend.parked is False


def test_rollout_anchor_two_updates_and_next_boundary_share_one_parked_lifecycle(caplog) -> None:
    events: list[str] = []
    backend = _Backend(events)

    class Rollout:
        def wake_up(self) -> None:
            events.append("wake")

        def sleep(self) -> None:
            events.append("sleep")

    trainer = UnifiedModelTrainer.__new__(UnifiedModelTrainer)
    trainer._single_engine = True
    trainer._rollout_is_trainside = False
    trainer._single_engine_staged_sync = False
    trainer._enable_fsdp_offload = False
    trainer._park_optimizer_state_during_rollout = True
    trainer._park_optimizer_state_during_train = True
    trainer.weight_sync = None
    trainer.rollout = Rollout()
    trainer.backend = backend

    with caplog.at_level("INFO"):
        with trainer._external_single_engine_session(
            sync_weights=False,
            onload_trainer_after=True,
            defer_optimizer_restore=True,
        ) as boundary:
            events.append("generate")
    assert boundary.optimizer_restore_deferred is True
    assert "restore=deferred" in caplog.text
    assert "restored=0.000" not in caplog.text
    assert backend.parked is True

    stack = _stack(backend)

    def prepare_anchor(self, name, track):
        del self, track
        assert backend.parked
        events.append(f"anchor_{name}")

    stack.prepare_segment = MethodType(prepare_anchor, stack)
    _install_backward(stack, events)
    stack.train_track(
        _track("ar", 4),
        _track("image", 4),
        training_progress=0.0,
        optimizer_state_already_parked=True,
    )
    assert backend.parked is False

    with trainer._external_single_engine_session(
        sync_weights=False,
        onload_trainer_after=True,
        defer_optimizer_restore=True,
    ) as next_boundary:
        events.append("next_generate")
    assert next_boundary.optimizer_restore_deferred is True
    assert backend.parked is True
    backend.reclaim_cuda_allocator()
    backend.restore_optimizer_state_after_rollout()

    assert events == [
        "park",
        "wake",
        "generate",
        "sleep",
        "anchor_ar",
        "anchor_image",
        "zero_grad",
        "backward_ar",
        "backward_image",
        "reclaim",
        "restore",
        "optimizer_step",
        "zero_grad",
        "park",
        "backward_ar",
        "backward_image",
        "reclaim",
        "restore",
        "optimizer_step",
        "rollout_end",
        "park",
        "wake",
        "next_generate",
        "sleep",
        "reclaim",
        "restore",
    ]


def test_train_optimizer_metrics_sum_dp_shards_and_keep_critical_path_time() -> None:
    class Rank:
        tp_rank = 0
        is_pipeline_last_stage = True
        sp_rank = 0

    class WorkerGroup:
        rank_infos = [Rank(), Rank()]

    def result(
        bytes_by_update: tuple[float, float],
        times: tuple[float, float],
        pending_by_update: tuple[float, float],
    ) -> dict[str, TrainStepResult]:
        updates = tuple(
            {
                "train_optimizer_state_bytes_parked": byte_count,
                "train_optimizer_state_bytes_restored": byte_count,
                "train_optimizer_state_restore_slots_pending": pending,
                "train_optimizer_park_host_time_s": host_time,
                "loss": float(index),
            }
            for index, (byte_count, host_time, pending) in enumerate(zip(bytes_by_update, times, pending_by_update))
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
                    "train_optimizer_state_restore_slots_pending": max(pending_by_update),
                    "train_optimizer_park_host_time_s": sum(times) / 2.0,
                },
                per_update=updates,
            )
        }

    collected = _collect_unified_train_results(
        WorkerGroup(),
        [
            result((10.0, 12.0), (2.0, 5.0), (0.0, 1.0)),
            result((20.0, 28.0), (4.0, 3.0), (0.0, 2.0)),
        ],
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
            "train_optimizer_state_restore_slots_pending": 2.0,
            "train_optimizer_park_host_time_s": 5.0,
            "loss": 1.0,
        },
    )
    assert collected["image"].metrics == {
        "train_optimizer_state_bytes_parked": 35.0,
        "train_optimizer_state_bytes_restored": 35.0,
        "train_optimizer_state_restore_slots_pending": 2.0,
        "train_optimizer_park_host_time_s": 4.5,
    }
