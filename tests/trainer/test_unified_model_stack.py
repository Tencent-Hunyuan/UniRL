from __future__ import annotations

from types import MethodType, SimpleNamespace

import torch

from unirl.train.stack import TrainStepResult
from unirl.train.unified_model_stack import UnifiedModelTrainStack
from unirl.types.rollout_resp import RolloutTrack
from unirl.types.segments import make_image_segment


def test_update_preparation_runs_immediately_before_its_backward() -> None:
    events: list[str] = []

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

        def prepare_update_batch(self, *, micro_batches):
            assert len(micro_batches) == 2
            assert all(segment.batch_size == 1 for _, segment in micro_batches)
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

    stack._train_one_step(tracks, slices, training_progress=0.0)

    assert events == [
        "zero_grad",
        "backward_ar",
        "prepare_image",
        "backward_image",
        "optimizer_step",
        "finish_image_True",
    ]


def test_failed_image_backward_finalizes_prepared_state() -> None:
    events: list[str] = []

    class FakeBackend:
        def zero_grad(self):
            events.append("zero_grad")

    class FakeAlgorithm:
        def __init__(self, name: str, *, prepares_update_batch: bool) -> None:
            self.name = name
            self.prepares_update_batch = prepares_update_batch

        def prepare_update_batch(self, *, micro_batches):
            del micro_batches
            events.append(f"prepare_{self.name}")

        def finish_update_batch(self, *, succeeded):
            events.append(f"finish_{self.name}_{succeeded}")

    stack = object.__new__(UnifiedModelTrainStack)
    stack.fsdp_backend = FakeBackend()
    stack.algorithms = {
        "ar": FakeAlgorithm("ar", prepares_update_batch=False),
        "image": FakeAlgorithm("image", prepares_update_batch=True),
    }

    def fake_prepare(self, name, track, micro_slices):
        del track, micro_slices
        self.algorithms[name].prepare_update_batch(micro_batches=[])

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
