"""Regression tests for agentic partial/async drive boundaries and tail cleanup."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import List

import pytest
from omegaconf import OmegaConf

from unirl.trainer.agentic_async import AsyncAgenticTrainer, _GroupAssembler, _GroupBuffer
from unirl.trainer.agentic_partial import AgenticPartialTrainer
from unirl.types.sample import Part, Sample


def _sample(root: str) -> Sample:
    return Sample.request(Part.input([root]))


def test_async_tail_policy_default_is_backward_compatible_carry() -> None:
    parameter = inspect.signature(AsyncAgenticTrainer.__init__).parameters["tail_policy"]
    assert parameter.default == "carry"


@pytest.mark.parametrize(
    ("recipe", "expected"),
    [
        ("examples/alfworld/alfworld_grpo_async.yaml", "drop"),
        ("examples/deep_research/deep_research_calc_mathverify_async.yaml", "carry"),
    ],
)
def test_async_recipes_declare_safe_tail_policy(recipe: str, expected: str) -> None:
    repo = Path(__file__).resolve().parents[2]
    cfg = OmegaConf.load(repo / recipe)
    assert cfg.tail_policy == expected


class _FinalCompletionRollout:
    """A drive whose completion appears only at atomic finalization."""

    def __init__(self, completed: Sample) -> None:
        self.completed = completed
        self.events: List[str] = []

    def poll(self):
        self.events.append("poll")
        return [[]]

    def finalize_if_drained(self):
        self.events.append("finalize")
        return [[self.completed]]

    def submit(self, _tasks) -> None:
        self.events.append("submit")
        raise AssertionError("the final completion must be consumed before a refill")


def _async_boundary_trainer(rollout: _FinalCompletionRollout) -> AsyncAgenticTrainer:
    trainer = object.__new__(AsyncAgenticTrainer)
    trainer.rollout = rollout
    trainer.batch_size = 1
    trainer._assembler = _GroupAssembler(1)
    trainer._buffer = _GroupBuffer()
    trainer._weight_version = 0
    trainer._gen_id = 0
    trainer._buffer_max_staleness = 0
    trainer._pending_carried = []
    return trainer


def _partial_boundary_trainer(rollout: _FinalCompletionRollout) -> AgenticPartialTrainer:
    trainer = object.__new__(AgenticPartialTrainer)
    trainer.rollout = rollout
    trainer._assembler = _GroupAssembler(1)
    trainer._buffer = _GroupBuffer()
    trainer._weight_version = 0
    trainer._gen_id = 0
    return trainer


def test_async_boundary_consumes_final_completion_before_refill() -> None:
    rollout = _FinalCompletionRollout(_sample("async-root"))
    trainer = _async_boundary_trainer(rollout)

    groups = trainer._next_batch(rollout_id=3)

    assert [[traj.parts[0].sample_ids[0] for traj in group] for group in groups] == [["async-root"]]
    assert rollout.events == ["poll", "finalize"]


def test_partial_boundary_consumes_final_completion_before_refill() -> None:
    rollout = _FinalCompletionRollout(_sample("partial-root"))
    trainer = _partial_boundary_trainer(rollout)

    groups = trainer._collect_until(batch_size=1, rollout_id=4, stale=0)

    assert [[traj.parts[0].sample_ids[0] for traj in group] for group in groups] == [["partial-root"]]
    assert rollout.events == ["poll", "finalize"]


def test_group_assembler_discards_only_requested_incomplete_roots() -> None:
    assembler = _GroupAssembler(2)
    assembler.add_completed([_sample("drop"), _sample("keep")])

    assert assembler.discard_roots(["drop", "drop", "missing"]) == 1
    assert assembler.pending_roots() == {"keep"}
    assert assembler.discard_roots(["drop"]) == 0


@pytest.mark.parametrize("trainer_type", [AsyncAgenticTrainer, AgenticPartialTrainer])
def test_stale_buffer_eviction_releases_ground_truth(trainer_type) -> None:
    trainer = object.__new__(trainer_type)
    trainer._assembler = _GroupAssembler(1)
    trainer._buffer = _GroupBuffer()
    trainer._buffer.put([_sample("stale")], weight_version=0, gen_id=0)
    trainer._weight_version = 2
    trainer._gt_by_root = {"stale": "answer"}

    assert trainer._drain_buffer(1, max_staleness=0) is None
    assert trainer._gt_by_root == {}
    assert trainer._buffer.pop_evicted_groups() == []


@pytest.mark.parametrize("trainer_type", [AsyncAgenticTrainer, AgenticPartialTrainer])
def test_drop_tail_purges_completed_siblings_and_ground_truth(trainer_type) -> None:
    trainer = object.__new__(trainer_type)
    trainer._tail_policy = "drop"
    trainer._assembler = _GroupAssembler(2)
    trainer._assembler.add_completed([_sample("drop"), _sample("keep")])
    trainer._gt_by_root = {"drop": "answer", "keep": "other"}

    if trainer_type is AsyncAgenticTrainer:
        trainer._dropped_tail_trajectories = 0
        trainer._dropped_tail_roots = 0
        trainer._discarded_completed_trajectories = 0
        assert trainer._apply_tail_policy([_sample("drop")], rollout_id=5) == []
        assert trainer._dropped_tail_trajectories == 1
        assert trainer._dropped_tail_roots == 1
        assert trainer._discarded_completed_trajectories == 1
    else:
        trainer._carried = []
        trainer._apply_tail_policy([_sample("drop")], rollout_id=5)
        assert trainer._carried == []
        assert trainer._last_dropped_trajectories == 1
        assert trainer._last_dropped_roots == 1
        assert trainer._last_discarded_completed_trajectories == 1

    assert trainer._assembler.pending_roots() == {"keep"}
    assert trainer._gt_by_root == {"keep": "other"}


@pytest.mark.parametrize("trainer_type", [AsyncAgenticTrainer, AgenticPartialTrainer])
def test_carry_tail_preserves_partial_group_state(trainer_type) -> None:
    trainer = object.__new__(trainer_type)
    trainer._tail_policy = "carry"
    trainer._assembler = _GroupAssembler(2)
    trainer._assembler.add_completed([_sample("carry")])
    trainer._gt_by_root = {"carry": "answer"}
    carried = [_sample("carry")]

    if trainer_type is AsyncAgenticTrainer:
        trainer._carried_tail_trajectories = 0
        assert trainer._apply_tail_policy(carried, rollout_id=6) is carried
        assert trainer._carried_tail_trajectories == 1
    else:
        trainer._carried = []
        trainer._apply_tail_policy(carried, rollout_id=6)
        assert trainer._carried is carried

    assert trainer._assembler.pending_roots() == {"carry"}
    assert trainer._gt_by_root == {"carry": "answer"}


class _TrainLoopRollout:
    def __init__(self, first_tail: list[Sample] | None = None) -> None:
        self.first_tail = list(first_tail or [])
        self.abort_calls = 0

    def abort(self):
        self.abort_calls += 1
        return [self.first_tail if self.abort_calls == 1 else []]


class _TrainLoopLogger:
    def __init__(self) -> None:
        self.rollout_metrics: list[tuple[int, dict]] = []

    def log_progress(self, *_args, **_kwargs) -> None:
        pass

    def log_rollout(self, step: int, metrics: dict) -> None:
        self.rollout_metrics.append((step, dict(metrics)))


def _async_train_loop_shell(*, tail: list[Sample]) -> tuple[AsyncAgenticTrainer, list[tuple[list[Sample], int]]]:
    trainer = object.__new__(AsyncAgenticTrainer)
    trainer._buffer_max_staleness = 0
    trainer._oversample = 1
    trainer._n = 2
    trainer.adv_normalization_scope = "group"
    trainer._train_fraction = 0.5
    trainer._tail_policy = "drop"
    trainer._weight_version = 0
    trainer._carried_tail_trajectories = 0
    trainer._dropped_tail_trajectories = 0
    trainer._dropped_tail_roots = 0
    trainer._discarded_completed_trajectories = 0
    trainer._gt_by_root = {"drop": "answer"}
    trainer.data_source = SimpleNamespace(get_samples=lambda _n: [])
    trainer.weight_sync = SimpleNamespace(sync=lambda: None)
    trainer.rollout = _TrainLoopRollout(tail)
    trainer.wandb_logger = _TrainLoopLogger()
    trainer.maybe_load_checkpoint = lambda _load_dir, *, num_rollouts: 0
    trainer._init_wandb = lambda **_kwargs: None
    trainer._next_batch = lambda _rollout_id: [[_sample("trained"), _sample("trained")]]
    trainer._train_on_groups = lambda *_args, **_kwargs: (SimpleNamespace(), 0.0)
    trainer.maybe_save_checkpoint = lambda *_args, **_kwargs: None
    trainer._finish_wandb = lambda: None
    submits: list[tuple[list[Sample], int]] = []
    trainer._submit_drive = lambda carried, rollout_id: submits.append((list(carried), rollout_id))
    return trainer, submits


def test_async_final_step_drops_tail_after_final_poll_without_starting_another_drive() -> None:
    trainer, submits = _async_train_loop_shell(tail=[_sample("drop")])
    pump_calls = 0

    def pump() -> int:
        nonlocal pump_calls
        pump_calls += 1
        if pump_calls == 1:
            trainer._assembler.add_completed([_sample("drop")])
            return 1
        return 0

    trainer._pump = pump

    trainer.train(num_rollouts=1, weight_sync_interval=1)

    assert submits == [([], 0)]
    assert trainer._assembler.pending_roots() == set()
    assert trainer._gt_by_root == {}
    assert trainer._dropped_tail_trajectories == 1
    assert trainer._discarded_completed_trajectories == 1
    assert trainer.wandb_logger.rollout_metrics[-1][0] == 1
    assert trainer.wandb_logger.rollout_metrics[-1][1]["async/dropped_tail_trajectories"] == 1


def test_async_complete_resume_does_not_prime_a_new_drive() -> None:
    trainer, submits = _async_train_loop_shell(tail=[])
    trainer.maybe_load_checkpoint = lambda _load_dir, *, num_rollouts: num_rollouts
    trainer._pump = lambda: 0

    trainer.train(num_rollouts=2, load_dir="checkpoint")

    assert submits == []


def test_async_cleanup_failure_does_not_mask_training_failure() -> None:
    trainer, _ = _async_train_loop_shell(tail=[])
    finished = False

    def fail_training(_rollout_id: int):
        raise ValueError("training failed")

    def fail_abort():
        raise RuntimeError("cleanup failed")

    def finish() -> None:
        nonlocal finished
        finished = True

    trainer._next_batch = fail_training
    trainer.rollout.abort = fail_abort
    trainer._finish_wandb = finish

    with pytest.raises(ValueError, match="training failed"):
        trainer.train(num_rollouts=1)
    assert finished
