from __future__ import annotations

from contextlib import contextmanager
from inspect import signature
from types import MethodType, SimpleNamespace

import pytest
import torch

from unirl.distributed.group.dispatch import DISTRIBUTED_CONFIG_ATTR
from unirl.train.stack import TrainStepResult
from unirl.train.unified_model_stack import UnifiedModelTrainStack, _collect_unified_train_results
from unirl.types.rollout_resp import RolloutTrack
from unirl.types.segments import make_image_segment


def _train_result(
    *,
    metrics: dict[str, float],
    per_update: tuple[dict[str, float], ...] = (),
) -> TrainStepResult:
    return TrainStepResult(
        loss=1.0,
        grad_norm=0.5,
        lr=1.0e-6,
        has_backward=True,
        micros=[],
        metrics=metrics,
        per_update=per_update,
    )


def test_unified_train_collector_uses_dp_critical_path_phase_times(monkeypatch) -> None:
    class Rank:
        def __init__(self, *, sp_rank: int) -> None:
            self.tp_rank = 0
            self.is_pipeline_last_stage = True
            self.sp_rank = sp_rank

    class WorkerGroup:
        # Two DP groups with two SP ranks each. Only ranks 0 and 2 are DP heads.
        rank_infos = [Rank(sp_rank=0), Rank(sp_rank=1), Rank(sp_rank=0), Rank(sp_rank=1)]

    rank_zero = {
        "ar": _train_result(
            metrics={"ar_backward_host_time_s": 3.0, "ratio_mean": 1.0},
            per_update=(
                {"ar_backward_host_time_s": 2.0, "loss": 10.0},
                {"ar_backward_host_time_s": 4.0, "loss": 11.0},
            ),
        ),
        "image": _train_result(
            metrics={
                "optimizer_host_time_s": 6.0,
                "cuda_peak_allocated_gb": 91.0,
                "cuda_peak_reserved_gb": 93.0,
                "cuda_train_window_peak_allocated_gb": 98.0,
                "cuda_train_window_peak_reserved_gb": 99.0,
                "image_micro_empty_cache_call_count": 2.5,
                "image_micro_empty_cache_skip_count": 45.5,
                "image_micro_empty_cache_min_free_gb": 6.0,
                "ratio_mean": 1.0,
            },
            per_update=(
                {
                    "optimizer_host_time_s": 5.0,
                    "cuda_peak_allocated_gb": 80.0,
                    "cuda_peak_reserved_gb": 85.0,
                    "image_micro_empty_cache_call_count": 2.0,
                    "image_micro_empty_cache_skip_count": 46.0,
                    "image_micro_empty_cache_min_free_gb": 7.0,
                    "loss": 20.0,
                },
                {
                    "optimizer_host_time_s": 7.0,
                    "cuda_peak_allocated_gb": 91.0,
                    "cuda_peak_reserved_gb": 93.0,
                    "image_micro_empty_cache_call_count": 3.0,
                    "image_micro_empty_cache_skip_count": 45.0,
                    "image_micro_empty_cache_min_free_gb": 6.0,
                    "loss": 21.0,
                },
            ),
        ),
    }
    other_dp_head = {
        "ar": _train_result(
            metrics={"ar_backward_host_time_s": 5.0, "ratio_mean": 9.0},
            per_update=(
                {"ar_backward_host_time_s": 8.0, "loss": 90.0},
                {"ar_backward_host_time_s": 3.0, "loss": 91.0},
            ),
        ),
        "image": _train_result(
            metrics={
                "optimizer_host_time_s": 8.0,
                "cuda_peak_allocated_gb": 96.0,
                "cuda_peak_reserved_gb": 97.0,
                "cuda_train_window_peak_allocated_gb": 100.0,
                "cuda_train_window_peak_reserved_gb": 101.0,
                "image_micro_empty_cache_call_count": 3.0,
                "image_micro_empty_cache_skip_count": 45.0,
                "image_micro_empty_cache_min_free_gb": 4.0,
                "ratio_mean": 9.0,
            },
            per_update=(
                {
                    "optimizer_host_time_s": 6.0,
                    "cuda_peak_allocated_gb": 96.0,
                    "cuda_peak_reserved_gb": 97.0,
                    "image_micro_empty_cache_call_count": 4.0,
                    "image_micro_empty_cache_skip_count": 44.0,
                    "image_micro_empty_cache_min_free_gb": 8.0,
                    "loss": 92.0,
                },
                {
                    "optimizer_host_time_s": 10.0,
                    "cuda_peak_allocated_gb": 89.0,
                    "cuda_peak_reserved_gb": 94.0,
                    "image_micro_empty_cache_call_count": 2.0,
                    "image_micro_empty_cache_skip_count": 46.0,
                    "image_micro_empty_cache_min_free_gb": 4.0,
                    "loss": 93.0,
                },
            ),
        ),
    }
    ignored_sp_rank = {
        "ar": _train_result(metrics={"ar_backward_host_time_s": 1_000.0}),
        "image": _train_result(metrics={"optimizer_host_time_s": 1_000.0}),
    }

    def fail_sync(*args, **kwargs):
        del args, kwargs
        raise AssertionError("controller reduction must not synchronize CUDA")

    monkeypatch.setattr(torch.cuda, "synchronize", fail_sync)
    collected = _collect_unified_train_results(
        WorkerGroup(),
        [rank_zero, ignored_sp_rank, other_dp_head, ignored_sp_rank],
    )

    assert collected["ar"].per_update == (
        {"ar_backward_host_time_s": 8.0, "loss": 10.0},
        {"ar_backward_host_time_s": 4.0, "loss": 11.0},
    )
    assert collected["ar"].metrics == {"ar_backward_host_time_s": 6.0, "ratio_mean": 1.0}
    assert collected["image"].per_update == (
        {
            "optimizer_host_time_s": 6.0,
            "cuda_peak_allocated_gb": 96.0,
            "cuda_peak_reserved_gb": 97.0,
            "image_micro_empty_cache_call_count": 4.0,
            "image_micro_empty_cache_skip_count": 44.0,
            "image_micro_empty_cache_min_free_gb": 7.0,
            "loss": 20.0,
        },
        {
            "optimizer_host_time_s": 10.0,
            "cuda_peak_allocated_gb": 91.0,
            "cuda_peak_reserved_gb": 94.0,
            "image_micro_empty_cache_call_count": 3.0,
            "image_micro_empty_cache_skip_count": 45.0,
            "image_micro_empty_cache_min_free_gb": 4.0,
            "loss": 21.0,
        },
    )
    assert collected["image"].metrics == {
        "optimizer_host_time_s": 8.0,
        "cuda_peak_allocated_gb": 96.0,
        "cuda_peak_reserved_gb": 97.0,
        "cuda_train_window_peak_allocated_gb": 100.0,
        "cuda_train_window_peak_reserved_gb": 101.0,
        "image_micro_empty_cache_call_count": 3.5,
        "image_micro_empty_cache_skip_count": 44.5,
        "image_micro_empty_cache_min_free_gb": 4.0,
        "ratio_mean": 1.0,
    }


def test_unified_train_collector_reduces_single_update_phase_times() -> None:
    class Rank:
        tp_rank = 0
        is_pipeline_last_stage = True
        sp_rank = 0

    class WorkerGroup:
        rank_infos = [Rank(), Rank()]

    collected = _collect_unified_train_results(
        WorkerGroup(),
        [
            {
                "image": _train_result(
                    metrics={"anchor_image_host_time_s": 4.0, "cuda_peak_reserved_gb": 77.0, "ratio_mean": 1.0}
                )
            },
            {
                "image": _train_result(
                    metrics={"anchor_image_host_time_s": 9.0, "cuda_peak_reserved_gb": 88.0, "ratio_mean": 2.0}
                )
            },
        ],
    )

    assert collected["image"].metrics == {
        "anchor_image_host_time_s": 9.0,
        "cuda_peak_reserved_gb": 88.0,
        "ratio_mean": 1.0,
    }


def test_unified_train_track_registers_critical_path_collector() -> None:
    config = getattr(UnifiedModelTrainStack.train_track, DISTRIBUTED_CONFIG_ATTR)

    assert config["collect_fn"] is _collect_unified_train_results


def test_cuda_reclamation_and_peak_telemetry_default_off() -> None:
    parameters = signature(UnifiedModelTrainStack.__init__).parameters

    assert parameters["empty_cache_after_image_micro"].default is False
    assert parameters["park_optimizer_state_during_train"].default is False
    assert parameters["image_micro_empty_cache_interval"].default == 1
    assert parameters["image_micro_empty_cache_min_free_gb"].default == 0.0
    assert parameters["empty_cache_after_optimizer"].default is False
    assert parameters["cuda_peak_telemetry"].default is False


def test_image_micro_reclamation_runs_after_each_image_backward_only(monkeypatch) -> None:
    events: list[str] = []

    class FakeAlgorithm:
        def compute_loss_and_backward(self, **kwargs):
            del kwargs
            events.append("backward")
            return SimpleNamespace(loss=1.0, metrics={}, num_steps_or_tokens=1, has_backward=True)

    stack = object.__new__(UnifiedModelTrainStack)
    stack.algorithms = {"ar": FakeAlgorithm(), "image": FakeAlgorithm()}
    stack.empty_cache_after_image_micro = True
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("empty_cache"))
    track = RolloutTrack(
        sample_ids=["sample-0", "sample-1"],
        conditions={},
        segment=make_image_segment(latents=torch.zeros(2, 1, 1, 1)),
        advantages=torch.ones(2),
    )

    image_result, _ = stack._backward_track("image", track, [(0, 1), (1, 2)], training_progress=0.0)
    stack._backward_track("ar", track, [(0, 1), (1, 2)], training_progress=0.0)

    assert events == ["backward", "empty_cache", "backward", "empty_cache", "backward", "backward"]
    assert float(image_result.metrics["image_micro_empty_cache_host_time_s"]) >= 0.0
    assert image_result.metrics["image_micro_empty_cache_call_count"] == 2.0
    assert image_result.metrics["image_micro_empty_cache_skip_count"] == 0.0
    assert image_result.metrics["image_micro_empty_cache_pressure_call_count"] == 0.0
    assert image_result.metrics["image_micro_empty_cache_pressure_check_count"] == 0.0


def test_image_micro_reclamation_uses_bounded_cadence_and_final_boundary(monkeypatch) -> None:
    events: list[str] = []

    class FakeAlgorithm:
        def compute_loss_and_backward(self, **kwargs):
            del kwargs
            events.append("backward")
            return SimpleNamespace(loss=1.0, metrics={}, num_steps_or_tokens=1, has_backward=True)

    stack = object.__new__(UnifiedModelTrainStack)
    stack.algorithms = {"image": FakeAlgorithm()}
    stack.empty_cache_after_image_micro = True
    stack.image_micro_empty_cache_interval = 3
    stack.image_micro_empty_cache_min_free_gb = 0.0
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("empty_cache"))
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda: (_ for _ in ()).throw(AssertionError("disabled pressure floor must not query CUDA memory")),
    )
    track = RolloutTrack(
        sample_ids=[f"sample-{index}" for index in range(7)],
        conditions={},
        segment=make_image_segment(latents=torch.zeros(7, 1, 1, 1)),
        advantages=torch.ones(7),
    )

    result, _ = stack._backward_track(
        "image",
        track,
        [(index, index + 1) for index in range(7)],
        training_progress=0.0,
    )

    assert events == [
        "backward",
        "backward",
        "backward",
        "empty_cache",
        "backward",
        "backward",
        "backward",
        "empty_cache",
        "backward",
        "empty_cache",
    ]
    assert result.metrics["image_micro_empty_cache_call_count"] == 3.0
    assert result.metrics["image_micro_empty_cache_skip_count"] == 4.0
    assert result.metrics["image_micro_empty_cache_pressure_call_count"] == 0.0
    assert result.metrics["image_micro_empty_cache_pressure_check_count"] == 0.0
    assert "image_micro_empty_cache_min_free_gb" not in result.metrics


def test_image_micro_reclamation_pressure_floor_can_only_add_calls(monkeypatch) -> None:
    events: list[str] = []
    free_gb = iter((8.0, 5.0, 9.0))

    class FakeAlgorithm:
        def compute_loss_and_backward(self, **kwargs):
            del kwargs
            events.append("backward")
            return SimpleNamespace(loss=1.0, metrics={}, num_steps_or_tokens=1, has_backward=True)

    stack = object.__new__(UnifiedModelTrainStack)
    stack.algorithms = {"image": FakeAlgorithm()}
    stack.empty_cache_after_image_micro = True
    stack.image_micro_empty_cache_interval = 4
    stack.image_micro_empty_cache_min_free_gb = 6.0
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("empty_cache"))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (int(next(free_gb) * 2**30), 96 * 2**30))
    track = RolloutTrack(
        sample_ids=[f"sample-{index}" for index in range(5)],
        conditions={},
        segment=make_image_segment(latents=torch.zeros(5, 1, 1, 1)),
        advantages=torch.ones(5),
    )

    result, _ = stack._backward_track(
        "image",
        track,
        [(index, index + 1) for index in range(5)],
        training_progress=0.0,
    )

    # Micro 2 is reclaimed early at 5 GiB free, micro 4 is the cadence
    # boundary, and micro 5 is always reclaimed as the final boundary.
    assert events == [
        "backward",
        "backward",
        "empty_cache",
        "backward",
        "backward",
        "empty_cache",
        "backward",
        "empty_cache",
    ]
    assert result.metrics["image_micro_empty_cache_call_count"] == 3.0
    assert result.metrics["image_micro_empty_cache_skip_count"] == 2.0
    assert result.metrics["image_micro_empty_cache_pressure_call_count"] == 1.0
    assert result.metrics["image_micro_empty_cache_pressure_check_count"] == 3.0
    assert result.metrics["image_micro_empty_cache_min_free_gb"] == 5.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"image_micro_empty_cache_interval": 0}, "image_micro_empty_cache_interval must be >= 1"),
        ({"image_micro_empty_cache_min_free_gb": -1.0}, "image_micro_empty_cache_min_free_gb must be >= 0"),
        ({"image_micro_empty_cache_min_free_gb": float("nan")}, "must be finite"),
    ],
)
def test_image_micro_reclamation_config_validation(kwargs, message) -> None:
    algorithm = SimpleNamespace(supports_multi_update=True)

    with pytest.raises(ValueError, match=message):
        UnifiedModelTrainStack(
            fsdp_backend=object(),
            ar_algorithm=algorithm,
            image_algorithm=algorithm,
            micro_batch_size=1,
            max_grad_norm=1.0,
            **kwargs,
        )


def test_post_optimizer_reclamation_and_peak_telemetry(monkeypatch) -> None:
    events: list[str] = []

    class FakeBackend:
        optimizer = SimpleNamespace(param_groups=[{"lr": 1.0e-6}])
        scheduler = None

        def zero_grad(self):
            events.append("zero_grad")

        def optimizer_step(self, *, max_grad_norm):
            assert max_grad_norm == 1.0
            events.append("optimizer_step")
            return 0.5

    stack = object.__new__(UnifiedModelTrainStack)
    stack.fsdp_backend = FakeBackend()
    stack.algorithms = {
        "ar": SimpleNamespace(prepares_update_batch=False),
        "image": SimpleNamespace(prepares_update_batch=False),
    }
    stack.num_updates_per_batch = 1
    stack.max_grad_norm = 1.0
    stack.empty_cache_after_optimizer = True
    stack.cuda_peak_telemetry = True

    def fake_backward(self, name, resp_track, micro_slices, *, training_progress):
        del self, resp_track, micro_slices, training_progress
        events.append(f"backward_{name}")
        return _train_result(metrics={}), True

    stack._backward_track = MethodType(fake_backward, stack)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: events.append("reset_peak"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("empty_cache"))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 80 * 2**30)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 91 * 2**30)

    results = stack._train_one_step(
        {"ar": object(), "image": object()},
        {"ar": [(0, 1)], "image": [(0, 1)]},
        training_progress=0.0,
    )

    assert events == ["reset_peak", "zero_grad", "backward_ar", "backward_image", "optimizer_step", "empty_cache"]
    assert results["image"].metrics["cuda_peak_allocated_gb"] == 80.0
    assert results["image"].metrics["cuda_peak_reserved_gb"] == 91.0
    assert float(results["image"].metrics["post_optimizer_empty_cache_host_time_s"]) >= 0.0
    assert "cuda_peak_allocated_gb" not in results["ar"].metrics


def test_update_preparation_runs_immediately_before_its_backward(monkeypatch) -> None:
    events: list[str] = []
    profile_ranges: list[str] = []

    class FakeProfiler:
        @contextmanager
        def record(self, name: str):
            profile_ranges.append(name)
            yield

    class FakeBackend:
        _device = torch.device("cpu")
        optimizer = SimpleNamespace(param_groups=[{"lr": 1.0e-6}])
        scheduler = None

        def zero_grad(self):
            events.append("zero_grad")

        def optimizer_step(self, *, max_grad_norm):
            assert max_grad_norm == 1.0
            events.append("optimizer_step")
            return 0.5

    class FakeAlgorithm:
        def __init__(self, name: str, *, prepares_update_batch: bool) -> None:
            self.name = name
            self.prepares_update_batch = prepares_update_batch
            self.prepares_phased_update_batch = prepares_update_batch
            self.prepares_indexed_update_batch = prepares_update_batch

        def prepare_update_batch(self, *, micro_batches, training_progress, loss_scale, update_index):
            assert len(micro_batches) == 2
            assert all(segment.batch_size == 1 for _, segment, _ in micro_batches)
            assert all(torch.equal(advantages, torch.ones(1)) for _, _, advantages in micro_batches)
            assert training_progress == 0.0
            assert loss_scale == 0.5
            assert update_index == 1
            events.append(f"prepare_{self.name}")

        def finish_update_batch(self, *, succeeded):
            events.append(f"finish_{self.name}_{succeeded}")

    def track(prefix: str) -> RolloutTrack:
        return RolloutTrack(
            sample_ids=[f"{prefix}-0", f"{prefix}-1"],
            conditions={},
            segment=make_image_segment(latents=torch.zeros(2, 1, 1, 1)),
            advantages=torch.ones(2),
        )

    stack = object.__new__(UnifiedModelTrainStack)
    stack.fsdp_backend = FakeBackend()
    stack.algorithms = {
        "ar": FakeAlgorithm("ar", prepares_update_batch=False),
        "image": FakeAlgorithm("image", prepares_update_batch=True),
    }
    stack.empty_cache_after_optimizer = False
    stack.cuda_peak_telemetry = False
    stack.num_updates_per_batch = 1
    stack.max_grad_norm = 1.0
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "reset_peak_memory_stats",
        lambda: (_ for _ in ()).throw(AssertionError("default-off telemetry must not reset CUDA peaks")),
    )
    monkeypatch.setattr(
        torch.cuda,
        "empty_cache",
        lambda: (_ for _ in ()).throw(AssertionError("default-off reclamation must not empty the CUDA cache")),
    )

    def fake_backward(self, name, resp_track, micro_slices, *, training_progress):
        del self, resp_track, training_progress
        assert micro_slices == [(0, 1), (1, 2)]
        events.append(f"backward_{name}")
        return (
            TrainStepResult(
                loss=1.0,
                grad_norm=0.0,
                lr=0.0,
                has_backward=True,
                micros=[],
                metrics={},
            ),
            True,
        )

    stack._backward_track = MethodType(fake_backward, stack)
    tracks = {"ar": track("ar"), "image": track("image")}
    slices = {"ar": [(0, 1), (1, 2)], "image": [(0, 1), (1, 2)]}

    results = stack._train_one_step(
        tracks,
        slices,
        training_progress=0.0,
        update_index=1,
        profiler=FakeProfiler(),
        anchor_image_host_time_s=7.5,
    )

    assert events == [
        "zero_grad",
        "backward_ar",
        "prepare_image",
        "backward_image",
        "optimizer_step",
        "finish_image_True",
    ]
    assert profile_ranges == [
        "update_1/ar_backward",
        "update_1/image_prepare_reference",
        "update_1/image_ratio_mse_backward",
        "update_1/optimizer",
    ]
    assert set(results["ar"].metrics) == {"ar_backward_host_time_s"}
    assert float(results["ar"].metrics["ar_backward_host_time_s"]) >= 0.0
    assert set(results["image"].metrics) == {
        "anchor_image_host_time_s",
        "image_prepare_reference_host_time_s",
        "image_ratio_mse_backward_host_time_s",
        "pre_optimizer_empty_cache_host_time_s",
        "optimizer_host_time_s",
    }
    assert results["image"].metrics["anchor_image_host_time_s"] == 7.5
    assert float(results["image"].metrics["image_prepare_reference_host_time_s"]) >= 0.0
    assert float(results["image"].metrics["image_ratio_mse_backward_host_time_s"]) >= 0.0
    assert float(results["image"].metrics["pre_optimizer_empty_cache_host_time_s"]) >= 0.0
    assert float(results["image"].metrics["optimizer_host_time_s"]) >= 0.0


def test_train_track_attaches_anchor_timing_to_first_update_only(monkeypatch) -> None:
    monkeypatch.delenv("UNIRL_PROFILE", raising=False)
    prepare_calls: list[str] = []
    update_calls: list[tuple[int, object, object]] = []

    class FakeBackend:
        _device = torch.device("cpu")

        def on_rollout_end(self):
            return None

    def track(prefix: str) -> RolloutTrack:
        return RolloutTrack(
            sample_ids=[f"{prefix}-{i}" for i in range(4)],
            conditions={},
            segment=make_image_segment(latents=torch.zeros(4, 1, 1, 1)),
            advantages=torch.ones(4),
        )

    stack = object.__new__(UnifiedModelTrainStack)
    stack.fsdp_backend = FakeBackend()
    stack.algorithms = {"ar": object(), "image": object()}
    stack.micro_batch_size = 1
    stack.num_updates_per_batch = 2

    def fake_prepare(self, name, resp_track):
        del self, resp_track
        prepare_calls.append(name)

    def fake_train_one_step(
        self,
        tracks,
        slices_by_track,
        *,
        training_progress,
        update_index,
        profiler,
        anchor_image_host_time_s,
        optimizer_state_already_parked,
    ):
        del self, tracks, slices_by_track, training_progress
        assert optimizer_state_already_parked is False
        update_calls.append((update_index, profiler, anchor_image_host_time_s))
        return {
            name: TrainStepResult(
                loss=float(update_index),
                grad_norm=0.5,
                lr=1.0e-6,
                has_backward=True,
                micros=[],
                metrics={"update": float(update_index)},
            )
            for name in ("ar", "image")
        }

    stack.prepare_segment = MethodType(fake_prepare, stack)
    stack._train_one_step = MethodType(fake_train_one_step, stack)

    stack.train_track(track("ar"), track("image"), training_progress=0.0)

    assert prepare_calls == ["ar", "image"]
    assert [update_index for update_index, _, _ in update_calls] == [0, 1]
    assert all(profiler is None for _, profiler, _ in update_calls)
    assert isinstance(update_calls[0][2], float)
    assert float(update_calls[0][2]) >= 0.0
    assert update_calls[1][2] is None


def test_train_track_supplies_full_anchor_plan_and_finalizes_on_failure(monkeypatch) -> None:
    monkeypatch.delenv("UNIRL_PROFILE", raising=False)
    events: list[object] = []

    class FakeBackend:
        _device = torch.device("cpu")

    class ImageAlgorithm:
        prepares_anchor_plan = True

        def prepare_anchor_batch(self, *, updates):
            events.append(("anchor_plan", [[int(segment.batch_size) for _, segment in update] for update in updates]))

        def finish_anchor_batch(self, *, succeeded):
            events.append(("finish_anchor", succeeded))

    def track(prefix: str) -> RolloutTrack:
        return RolloutTrack(
            sample_ids=[f"{prefix}-{i}" for i in range(4)],
            conditions={},
            segment=make_image_segment(latents=torch.zeros(4, 1, 1, 1)),
            advantages=torch.ones(4),
        )

    stack = object.__new__(UnifiedModelTrainStack)
    stack.fsdp_backend = FakeBackend()
    stack.algorithms = {"ar": object(), "image": ImageAlgorithm()}
    stack.micro_batch_size = 1
    stack.num_updates_per_batch = 2

    def fake_prepare(self, name, resp_track):
        del self, resp_track
        events.append(("prepare_segment", name))

    def fail_first_update(self, *args, update_index, **kwargs):
        del self, args, kwargs
        events.append(("update", update_index))
        raise RuntimeError("expected update failure")

    stack.prepare_segment = MethodType(fake_prepare, stack)
    stack._train_one_step = MethodType(fail_first_update, stack)

    with pytest.raises(RuntimeError, match="expected update failure"):
        stack.train_track(track("ar"), track("image"), training_progress=0.0)

    assert events == [
        ("prepare_segment", "ar"),
        ("anchor_plan", [[1, 1], [1, 1]]),
        ("update", 0),
        ("finish_anchor", False),
    ]


def test_legacy_update_preparation_keeps_pair_only_api() -> None:
    captured = []

    class FakeAlgorithm:
        prepares_update_batch = True
        prepares_phased_update_batch = False

        def prepare_update_batch(self, *, micro_batches):
            captured.extend(micro_batches)

    stack = object.__new__(UnifiedModelTrainStack)
    stack.algorithms = {"image": FakeAlgorithm()}
    track = RolloutTrack(
        sample_ids=["image-0", "image-1"],
        conditions={},
        segment=make_image_segment(latents=torch.zeros(2, 1, 1, 1)),
        advantages=torch.ones(2),
    )

    stack._prepare_update_batch("image", track, [(0, 1), (1, 2)], training_progress=0.5)

    assert len(captured) == 2
    assert all(len(micro_batch) == 2 for micro_batch in captured)
    assert all(segment.batch_size == 1 for _, segment in captured)


def test_failed_image_backward_finalizes_prepared_state() -> None:
    events: list[str] = []

    class FakeBackend:
        def zero_grad(self):
            events.append("zero_grad")

    class FakeAlgorithm:
        def __init__(self, name: str, *, prepares_update_batch: bool) -> None:
            self.name = name
            self.prepares_update_batch = prepares_update_batch

        def prepare_update_batch(self, *, micro_batches, training_progress, loss_scale):
            del micro_batches, training_progress, loss_scale
            events.append(f"prepare_{self.name}")

        def finish_update_batch(self, *, succeeded):
            events.append(f"finish_{self.name}_{succeeded}")

    stack = object.__new__(UnifiedModelTrainStack)
    stack.fsdp_backend = FakeBackend()
    stack.algorithms = {
        "ar": FakeAlgorithm("ar", prepares_update_batch=False),
        "image": FakeAlgorithm("image", prepares_update_batch=True),
    }
    stack.empty_cache_after_optimizer = False
    stack.cuda_peak_telemetry = False

    def fake_prepare(self, name, track, micro_slices, *, training_progress, update_index):
        del track, micro_slices, update_index
        self.algorithms[name].prepare_update_batch(
            micro_batches=[], training_progress=training_progress, loss_scale=1.0
        )

    stack._prepare_update_batch = MethodType(fake_prepare, stack)

    def fail_image(self, name, resp_track, micro_slices, *, training_progress):
        del self, resp_track, micro_slices, training_progress
        events.append(f"backward_{name}")
        if name == "image":
            raise RuntimeError("image failed")
        return (
            TrainStepResult(
                loss=1.0,
                grad_norm=0.0,
                lr=0.0,
                has_backward=True,
                micros=[],
                metrics={},
            ),
            True,
        )

    stack._backward_track = MethodType(fail_image, stack)

    try:
        stack._train_one_step(
            {"ar": object(), "image": object()},
            {"ar": [], "image": []},
            training_progress=0.0,
        )
    except RuntimeError as exc:
        assert str(exc) == "image failed"
    else:
        raise AssertionError("expected image failure")

    assert events == [
        "zero_grad",
        "backward_ar",
        "prepare_image",
        "backward_image",
        "finish_image_False",
    ]
