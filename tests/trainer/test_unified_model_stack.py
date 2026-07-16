from __future__ import annotations

from contextlib import contextmanager
from types import MethodType, SimpleNamespace

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
            metrics={"optimizer_host_time_s": 6.0, "ratio_mean": 1.0},
            per_update=(
                {"optimizer_host_time_s": 5.0, "loss": 20.0},
                {"optimizer_host_time_s": 7.0, "loss": 21.0},
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
            metrics={"optimizer_host_time_s": 8.0, "ratio_mean": 9.0},
            per_update=(
                {"optimizer_host_time_s": 6.0, "loss": 92.0},
                {"optimizer_host_time_s": 10.0, "loss": 93.0},
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
        {"optimizer_host_time_s": 6.0, "loss": 20.0},
        {"optimizer_host_time_s": 10.0, "loss": 21.0},
    )
    assert collected["image"].metrics == {"optimizer_host_time_s": 8.0, "ratio_mean": 1.0}


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
            {"image": _train_result(metrics={"anchor_image_host_time_s": 4.0, "ratio_mean": 1.0})},
            {"image": _train_result(metrics={"anchor_image_host_time_s": 9.0, "ratio_mean": 2.0})},
        ],
    )

    assert collected["image"].metrics == {"anchor_image_host_time_s": 9.0, "ratio_mean": 1.0}


def test_unified_train_track_registers_critical_path_collector() -> None:
    config = getattr(UnifiedModelTrainStack.train_track, DISTRIBUTED_CONFIG_ATTR)

    assert config["collect_fn"] is _collect_unified_train_results


def test_update_preparation_runs_immediately_before_its_backward() -> None:
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

        def prepare_update_batch(self, *, micro_batches, training_progress, loss_scale):
            assert len(micro_batches) == 2
            assert all(segment.batch_size == 1 for _, segment, _ in micro_batches)
            assert all(torch.equal(advantages, torch.ones(1)) for _, _, advantages in micro_batches)
            assert training_progress == 0.0
            assert loss_scale == 0.5
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
    stack.num_updates_per_batch = 1
    stack.max_grad_norm = 1.0

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
    ):
        del self, tracks, slices_by_track, training_progress
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

    def fake_prepare(self, name, track, micro_slices, *, training_progress):
        del track, micro_slices
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
