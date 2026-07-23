"""TrainStack must freeze the anchor in eval mode and update in train mode."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from unirl.train.stack.base import TrainStack
from unirl.types.sample import Part


def test_train_track_switches_modes_at_the_anchor_update_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("unirl.utils.profiling.profile_scope", lambda: "")

    model = torch.nn.Linear(1, 1)
    events: list[str] = []
    part = Part.input(["prompt"])
    result = object()
    plans = [[(0, 1)]]

    class Planner:
        def arrange(self, value, *, num_updates, micro_batch_size):
            assert value is part
            assert (num_updates, micro_batch_size) == (1, 1)
            assert not model.training
            events.append("arrange_eval")
            return value, plans

    stack = object.__new__(TrainStack)
    stack.fsdp_backend = SimpleNamespace(model=model)
    stack.micro_planner = Planner()
    stack.num_updates_per_batch = 1
    stack.micro_batch_size = 1

    def align(value):
        assert value is part
        assert not model.training
        events.append("align_eval")

    def prepare(value, *, plans):
        assert value is part
        assert plans == [[(0, 1)]]
        assert not model.training
        events.append("prepare_eval")

    def run_updates(value, *, plans, training_progress):
        assert value is part
        assert plans == [[(0, 1)]]
        assert training_progress == 0.25
        assert model.training
        events.append("updates_train")
        return result

    stack._align_track_inputs = align
    stack.prepare_segment = prepare
    stack._run_updates = run_updates
    stack.on_rollout_end = lambda: events.append("rollout_end")

    assert stack.train_track(part, training_progress=0.25) is result
    assert model.training
    assert events == [
        "align_eval",
        "arrange_eval",
        "prepare_eval",
        "updates_train",
        "rollout_end",
    ]
