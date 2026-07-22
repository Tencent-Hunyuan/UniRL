from dataclasses import fields

import pytest

from unirl.rollout.async_runtime import (
    AsyncRolloutScheduler,
    BufferedRolloutGroup,
    InflightGeneration,
    VersionedGroupBuffer,
)
from unirl.types.sample import Part, Sample


def _sample(sample_id: str) -> Sample:
    return Sample.request(Part.input([sample_id]))


class _FakeDispatcher:
    def __init__(self) -> None:
        self.launched: list[Sample] = []
        self.completed: dict[int, Sample] = {}

    def launch(
        self,
        sample: Sample,
        *,
        gen_id: int,
        weight_version: int,
    ) -> InflightGeneration:
        self.launched.append(sample)
        self.completed[gen_id] = sample
        return InflightGeneration(
            refs=[gen_id],
            worker_local=False,
            gen_id=gen_id,
            weight_version=weight_version,
        )

    def is_ready(self, job: InflightGeneration) -> bool:
        return True

    def wait(self, job: InflightGeneration) -> None:
        raise AssertionError(f"ready job {job.gen_id} should not be waited on")

    def collect(self, job: InflightGeneration) -> Sample:
        return self.completed[job.gen_id]


def test_inflight_generation_does_not_retain_request_sample() -> None:
    assert {field.name for field in fields(InflightGeneration)} == {
        "refs",
        "worker_local",
        "gen_id",
        "weight_version",
    }


def test_buffer_underflow_does_not_consume_fresh_sample_groups() -> None:
    older = BufferedRolloutGroup(sample=_sample("older"), weight_version=1, gen_id=1)
    newer = BufferedRolloutGroup(sample=_sample("newer"), weight_version=1, gen_id=2)
    buffer = VersionedGroupBuffer()
    buffer.put_all([older, newer])

    assert buffer.drain_freshest(3, current_version=1, max_staleness=0) is None
    assert buffer.drain_freshest(2, current_version=1, max_staleness=0) == [newer, older]


def test_failed_completion_is_retryable_without_partial_buffer_insertion() -> None:
    dispatcher = _FakeDispatcher()
    scheduler = AsyncRolloutScheduler(dispatcher, groups_per_step=1)
    scheduler._launch_one(build_sample=lambda gen_id: _sample(f"p{gen_id}"), weight_version=3)
    attempts = 0

    def complete(job: InflightGeneration, completed: Sample) -> list[Sample]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("score failed")
        return [completed]

    with pytest.raises(RuntimeError, match="score failed"):
        scheduler.reap_ready(complete)

    assert scheduler._buffer.drain_freshest(1) is None
    scheduler.reap_ready(complete)
    (group,) = scheduler._buffer.drain_freshest(1) or []
    assert group.sample.parts[0].sample_ids == ["p0"]
    assert group.weight_version == 3
    assert attempts == 2


def test_next_step_uses_sample_builder_and_returns_sample_groups() -> None:
    dispatcher = _FakeDispatcher()
    scheduler = AsyncRolloutScheduler(dispatcher, groups_per_step=2)

    groups = scheduler.next_step(
        rollout_id=0,
        sync_interval=2,
        max_inflight=2,
        max_staleness=0,
        num_rollouts=2,
        current_version=4,
        build_sample=lambda gen_id: _sample(f"p{gen_id}"),
        on_complete=lambda _job, completed: [completed],
    )

    assert [sample.parts[0].sample_ids for sample in dispatcher.launched] == [["p0"], ["p1"]]
    assert [group.sample.parts[0].sample_ids for group in groups] == [["p1"], ["p0"]]
    assert all(group.weight_version == 4 for group in groups)
