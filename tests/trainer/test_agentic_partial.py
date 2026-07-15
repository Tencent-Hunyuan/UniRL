"""CPU contract tests for the colocate partial-rollout trainer's commit-N logic (LIN-531).

Exercises `AgenticPartialTrainer._pump` / `_collect_until` (poll → bucket into complete GRPO
groups → drain the freshest N, refilling on a drained-but-short buffer) with a fake rollout
handle — no GPU / Ray / real engine. The wake/sync/abort/sleep choreography + carry/drop are
covered by the GPU speed comparison.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List

import pytest

pytest.importorskip("torch")  # the unirl types import torch at module load

from unirl.trainer.agentic_async import _GroupAssembler, _GroupBuffer  # noqa: E402
from unirl.trainer.agentic_partial import AgenticPartialTrainer  # noqa: E402
from unirl.types.primitives import Texts  # noqa: E402
from unirl.types.sample import Part, Sample  # noqa: E402


def _traj(root: str) -> Sample:
    """A minimal trajectory Sample carrying just its root id."""
    return Sample.request(Part.input([root]))


class _FakeRollout:
    """Stubs the agentic engine Handle: `poll`/`drained`/`abort` return `[value]` (the
    BROADCAST+RANK_ZERO unwrap the trainer does with `[0]`). `poll` replays a script."""

    def __init__(self, poll_script: List[List[Sample]], drained_script: List[bool]) -> None:
        self._poll = list(poll_script)
        self._drained = list(drained_script)
        self.submitted: List[list] = []

    def poll(self):
        return [self._poll.pop(0) if self._poll else []]

    def drained(self):
        return [self._drained.pop(0) if self._drained else True]

    def submit(self, tasks):
        self.submitted.append(tasks)
        self._drained.insert(0, False)  # a fresh drive is active → not drained until it delivers

    def abort(self):
        return [[]]

    def wake_up(self):
        pass

    def sleep(self):
        pass


class _FakeDataSource:
    """Emits RolloutInputs-shaped batches so `_build_tasks` → `_build_request_sample` works."""

    def get_samples(self, n: int):
        return SimpleNamespace(
            sample_ids=[f"s{i}" for i in range(n)],
            primitives={"text": Texts(texts=[f"prompt-{i}" for i in range(n)])},
            metadata=[{"answer": str(i)} for i in range(n)],
        )


def _mk_partial(*, n: int = 2, batch_size: int = 2, tail_policy: str = "carry") -> AgenticPartialTrainer:
    """Build a partial trainer WITHOUT its heavy GPU __init__ (mirrors the SimpleNamespace-stub
    style of test_agentic_assembly.py)."""
    t = object.__new__(AgenticPartialTrainer)
    t._n = n
    t.batch_size = batch_size
    t._weight_version = 0
    t._gen_id = 0
    t._buffer_max_staleness = 0
    t._tail_policy = tail_policy
    t._oversample = batch_size
    t._assembler = _GroupAssembler(n)
    t._buffer = _GroupBuffer()
    t._gt_by_root = {}
    t._carried = []
    t._stop = ["</tool_call>"]
    return t


def test_pump_promotes_only_complete_groups():
    """`_pump` buckets polled trajectories by root and promotes only roots with all n siblings."""
    t = _mk_partial(n=2)
    t.rollout = _FakeRollout(poll_script=[[_traj("a"), _traj("a"), _traj("b")]], drained_script=[])
    added = t._pump()
    assert added == 3
    assert t._buffer.size() == 1  # 'a' complete (2 siblings); 'b' still pending (1)
    assert t._assembler.pending_roots() == {"b"}


def test_collect_until_returns_freshest_n_and_carries_rest():
    """`_collect_until` pumps until ≥N complete groups then drains the freshest N; the older
    complete groups stay buffered for the next round."""
    t = _mk_partial(n=2, batch_size=2)
    t.rollout = _FakeRollout(
        poll_script=[[_traj("a"), _traj("a")], [_traj("b"), _traj("b"), _traj("c"), _traj("c")]],
        drained_script=[False],  # after poll 1 the buffer has only 1 group → not drained, keep polling
    )
    picked = t._collect_until(2, rollout_id=0, stale=0)
    assert [_GroupAssembler.root_of(g[0]) for g in picked] == ["c", "b"]  # freshest gen_id first
    assert t._buffer.size() == 1  # 'a' (oldest) carried in the buffer


def test_collect_until_refills_when_drive_drains_short():
    """If the drive drains before the buffer fills, `_collect_until` submits a fresh refill drive
    (rather than deadlock) and completes once it arrives."""
    t = _mk_partial(n=2, batch_size=2)
    t.data_source = _FakeDataSource()
    t.rollout = _FakeRollout(
        poll_script=[[_traj("a"), _traj("a")], [], [_traj("b"), _traj("b")]],
        drained_script=[True],  # after the empty poll the drive is drained with only 1 group → refill
    )
    picked = t._collect_until(2, rollout_id=0, stale=0)
    assert len(picked) == 2
    assert len(t.rollout.submitted) == 1  # exactly one refill submit fired
