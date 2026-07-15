"""CPU contract tests for AgenticRolloutEngine (LIN-522).

Exercises the REAL per-worker execution logic — the drain thread pool
(``run_drain`` / ``_drain_worker``), the ``_run_one`` multi-turn trajectory
loop, the pool-size trajectory cap, and the ``next_task`` FIFO queue — with a
CPU ``FakeEngine`` inner (``_fakes.py``) and a re-entrant multi-turn
``FakeEnv``. Only ``_pull`` is monkeypatched (to draw from an in-process queue)
so these run without Ray; the rank-0 coordinator's ``generate``/``Worker.call``
wiring + cross-worker balance are covered by the GPU smoke
(``scripts/agentic_engine_smoke.py``).
"""

from __future__ import annotations

import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pytest

pytest.importorskip("torch")  # the unirl types import torch at module load

from tests.rollout.engine._fakes import FakeEngine  # noqa: E402
from unirl.rollout.engine.agentic.config import AgenticRolloutEngineConfig  # noqa: E402
from unirl.rollout.engine.agentic.engine import AgenticRolloutEngine  # noqa: E402
from unirl.rollout.engine.base import BaseEngineConfig  # noqa: E402
from unirl.types.primitives import Texts  # noqa: E402
from unirl.types.sample import Part, Sample  # noqa: E402
from unirl.types.sampling import ARSamplingParams  # noqa: E402

# --------------------------------------------------------------------------- #
# Fakes: an inner-engine config that builds a FakeEngine, and a re-entrant env
# --------------------------------------------------------------------------- #


@dataclass
class _FakeInnerConfig(BaseEngineConfig):
    """Builds a CPU ``FakeEngine`` as the agentic engine's inner."""

    concurrency: int = 16
    yields: int = 4

    def make_engine(self, **deps: Any) -> FakeEngine:  # deps (device/rank/...) ignored
        return FakeEngine(concurrency=self.concurrency, yields=self.yields)


@dataclass
class _InvalidInnerConfig(BaseEngineConfig):
    def make_engine(self, **deps: Any) -> object:
        return object()


class FakeEnv:
    """Re-entrant multi-turn env: terminate after ``turns_for(root_id)`` turns.

    Stateless — the turn is derived from the sample (``len(gen_parts())``), so one
    instance safely serves many concurrent trajectory threads (the LIN-522
    requirement). Each non-final turn returns a one-row tool-result observation so
    the trajectory grows ``[input, gen, obs, gen, ...]``. Optionally sleeps in
    ``step`` (slow tool) or raises for a set of root ids (failure-isolation).
    """

    def __init__(
        self,
        *,
        turns_by_prompt: Optional[Dict[str, int]] = None,
        default_turns: int = 1,
        step_sleep: float = 0.0,
        fail_roots: Tuple[str, ...] = (),
    ) -> None:
        self._turns = dict(turns_by_prompt or {})
        self._default = int(default_turns)
        self._sleep = float(step_sleep)
        self._fail = set(fail_roots)

    @staticmethod
    def _root(sample: Sample) -> str:
        return sample.parts[0].sample_ids[0]

    def reset(self, request: Sample) -> Sample:
        return request

    def step(self, sample: Sample) -> Tuple[Optional[Texts], bool, dict]:
        if self._sleep:
            time.sleep(self._sleep)
        root = self._root(sample)
        if root in self._fail:
            raise RuntimeError(f"FakeEnv boom for {root}")
        turn = len(sample.gen_parts())
        target = self._turns.get(root, self._default)
        if turn >= target:
            return None, True, {"turn": turn}
        rows = sample.parts[-1].sample_ids
        return Texts(texts=[f"obs::{root}::t{turn}" for _ in rows]), False, {"turn": turn}


def _make_engine(
    *,
    n: int = 1,
    cap: int = 8,
    max_turns: int = 8,
    env: Optional[FakeEnv] = None,
    inner_yields: int = 4,
    concurrency: int = 16,
) -> AgenticRolloutEngine:
    cfg = AgenticRolloutEngineConfig(
        inner=_FakeInnerConfig(concurrency=concurrency, yields=inner_yields),
        env=env if env is not None else FakeEnv(default_turns=1),
        max_turns=max_turns,
        episode_sampling=ARSamplingParams(samples_per_prompt=n),
        per_worker_concurrency=cap,
    )
    return AgenticRolloutEngine(cfg, rank=0)


def _req(root_id: str) -> Sample:
    """A single-trajectory request: ``[input(1)]`` rooted at a slash-free id."""
    return Sample.request(Part.input([root_id], primitives={"text": Texts(texts=[f"prompt-{root_id}"])}))


def _wait_until(predicate, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition not reached in time")


def test_constructor_rejects_inner_without_single_turn_contract():
    cfg = AgenticRolloutEngineConfig(
        inner=_InvalidInnerConfig(),
        env=FakeEnv(),
        max_turns=1,
        episode_sampling=ARSamplingParams(samples_per_prompt=1),
    )
    with pytest.raises(ValueError, match="single-turn engine contract"):
        AgenticRolloutEngine(cfg, rank=0)


def _patch_pull(engine: AgenticRolloutEngine, queue: deque) -> None:
    """Replace ``_pull`` with an in-process pop (no Ray); FIFO, ``None`` sentinel.
    try/pop (not check-then-pop): N drain threads pull concurrently."""

    def _pull(_coordinator: Any, _role: str) -> Optional[Sample]:
        try:
            return queue.popleft()
        except IndexError:
            return None

    engine._pull = _pull  # type: ignore[assignment]


def _drive(engine: AgenticRolloutEngine, queue: deque) -> List[Sample]:
    """Run the background drain to quiescence and return the COMPLETED trajectories.

    The drain writes into the worker-side ``_completed`` / ``_checkpointed``
    buffers (LIN-531) instead of returning a list, so read the completed buffer.
    """
    _patch_pull(engine, queue)
    engine.run_drain(None, "")
    return list(engine._completed)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_run_one_builds_a_multi_turn_trajectory():
    """``_run_one`` forks-generates-observes until the env says done; weight_version
    is stamped on each gen Part. Contract (LIN-531): returns ``(sample, done)``."""
    engine = _make_engine(env=FakeEnv(turns_by_prompt={"p0": 3}))
    traj, done = engine._run_one(_req("p0"))

    assert done is True  # env said done → terminal
    assert traj.parts[0].sample_ids == ["p0"]  # root prompt id preserved
    assert len(traj.gen_parts()) == 3  # 3 turns
    # [input, gen, obs, gen, obs, gen]
    assert len(traj.parts) == 6
    assert [bool(p.sampling_params is not None) for p in traj.parts] == [False, True, False, True, False, True]
    for gp in traj.gen_parts():
        assert gp.weight_version == 0  # FakeEngine stamps version 0
    engine.shutdown()


def test_drain_returns_flat_list_of_n_times_P():
    """The drain returns one trajectory per task; count == n × P."""
    engine = _make_engine(n=2)
    tasks = deque(_req(r) for r in ("p0", "p0", "p1", "p1"))  # 2 prompts × n=2
    out = _drive(engine, tasks)
    assert isinstance(out, list)
    assert len(out) == 4
    assert all(isinstance(s, Sample) for s in out)
    engine.shutdown()


def test_ragged_depth_and_bucket_by_root_id():
    """THE §4.1/§3.1 case: trajectories of DIFFERENT turn counts come back as
    distinct variable-depth Samples (no concat), and bucket-by-root-id recovers
    each prompt's n-group."""
    env = FakeEnv(turns_by_prompt={"p0": 1, "p1": 3})
    engine = _make_engine(n=2, env=env)
    tasks = deque([_req("p0"), _req("p0"), _req("p1"), _req("p1")])
    out = _drive(engine, tasks)

    # group recovered purely by root id (a read over the list)
    by_root: Dict[str, List[Sample]] = {}
    for traj in out:
        by_root.setdefault(traj.parts[0].sample_ids[0], []).append(traj)
    assert Counter({k: len(v) for k, v in by_root.items()}) == Counter({"p0": 2, "p1": 2})

    # ragged depth: p0 trajectories are 1 turn, p1 are 3 — never coerced/concat'd
    assert sorted(len(s.gen_parts()) for s in by_root["p0"]) == [1, 1]
    assert sorted(len(s.gen_parts()) for s in by_root["p1"]) == [3, 3]
    engine.shutdown()


class _Counting(AgenticRolloutEngine):
    """Counts concurrent ``_run_one`` bodies (= trajectories in flight)."""

    def __init__(self, *a: Any, **k: Any) -> None:
        super().__init__(*a, **k)
        self._count_lock = threading.Lock()
        self.inflight = 0
        self.peak = 0

    def _run_one(self, task: Sample) -> Tuple[Sample, bool]:
        with self._count_lock:
            self.inflight += 1
            self.peak = max(self.peak, self.inflight)
        try:
            return super()._run_one(task)
        finally:
            with self._count_lock:
                self.inflight -= 1


def test_cap_bounds_concurrent_trajectories():
    """The per-worker cap (= the drain pool size) is the binding bound on
    concurrent trajectories: the drain saturates it (deterministic via the inner
    hold-gate) and never exceeds it."""
    cap = 4
    cfg = AgenticRolloutEngineConfig(
        inner=_FakeInnerConfig(concurrency=16, yields=4),
        env=FakeEnv(default_turns=1),
        max_turns=4,
        episode_sampling=ARSamplingParams(samples_per_prompt=1),
        per_worker_concurrency=cap,
    )
    engine = _Counting(cfg, rank=0)
    engine._inner._backend.block_until_released = True
    tasks = deque(_req(f"p{i}") for i in range(10))  # 10 > cap
    _patch_pull(engine, tasks)

    drain = threading.Thread(target=lambda: engine.run_drain(None, ""))
    drain.start()
    _wait_until(lambda: engine.inflight == cap)  # saturated: 4 alive, the 5th cannot start
    assert engine.peak == cap

    engine._inner._backend.release.set()  # stays set: the rest flow through
    drain.join(timeout=10)
    assert not drain.is_alive()

    assert len(engine._completed) == 10  # drained fully
    assert engine.peak == cap  # never exceeded: the pool size IS the cap
    engine.shutdown()


def test_two_bounds_are_distinct():
    """Concurrent TRAJECTORIES (cap = pool size) and concurrent REQUESTS (the
    inner backend's semaphore) are different bounds: with concurrency < cap,
    more trajectories are in flight than requests."""
    cap, concurrency = 4, 2
    cfg = AgenticRolloutEngineConfig(
        inner=_FakeInnerConfig(concurrency=concurrency, yields=6),
        env=FakeEnv(default_turns=1),
        max_turns=4,
        episode_sampling=ARSamplingParams(samples_per_prompt=1),
        per_worker_concurrency=cap,
    )
    engine = _Counting(cfg, rank=0)
    engine._inner._backend.block_until_released = True
    tasks = deque(_req(f"p{i}") for i in range(8))
    _patch_pull(engine, tasks)

    drain = threading.Thread(target=lambda: engine.run_drain(None, ""))
    drain.start()
    # 4 trajectories alive at once; only 2 requests admitted by the semaphore.
    _wait_until(lambda: engine.inflight == cap and engine._inner._backend.inflight == concurrency)

    engine._inner._backend.release.set()
    drain.join(timeout=10)
    assert not drain.is_alive()

    assert len(engine._completed) == 8
    assert engine.peak == cap  # 4 trajectories alive at once
    assert engine._inner._backend.peak == concurrency  # but only 2 requests generate at once
    engine.shutdown()


def test_failure_isolation():
    """A trajectory that raises (a failing tool) yields a partial and does NOT sink
    the drain — the other trajectories still complete."""
    env = FakeEnv(default_turns=2, fail_roots=("p1",))
    engine = _make_engine(n=1, env=env)
    tasks = deque(_req(r) for r in ("p0", "p1", "p2"))
    out = _drive(engine, tasks)

    assert len(out) == 3  # all three returned, including the failed one
    by_root = {s.parts[0].sample_ids[0]: s for s in out}
    # p1 failed on its first env.step → partial: it never completed 2 turns
    assert len(by_root["p1"].gen_parts()) < 2
    # the others completed normally (2 turns each)
    assert len(by_root["p0"].gen_parts()) == 2
    assert len(by_root["p2"].gen_parts()) == 2
    engine.shutdown()


def test_next_task_is_fifo_with_none_sentinel():
    """The coordinator's FIFO queue: pop in order, then ``None`` when drained."""
    engine = _make_engine()
    a, b = _req("p0"), _req("p1")
    engine._queue = deque([a, b])
    assert engine.next_task(0) is a
    assert engine.next_task(7) is b  # worker_rank ignored (FIFO)
    assert engine.next_task(0) is None  # drained
    engine.shutdown()


def test_pull_load_balancing_fast_worker_pulls_more():
    """Two workers sharing one FIFO queue, drained on two threads: the fast
    worker (cheaper per-trajectory) pulls strictly more, and no task is lost.
    (The emergent cross-worker balance; the per-worker greedy property is
    test_cap_bounds_concurrent_trajectories.)"""
    fast = _make_engine(cap=1, inner_yields=1, env=FakeEnv(default_turns=1))
    slow = _make_engine(cap=1, inner_yields=40, env=FakeEnv(default_turns=1))

    queue = deque(_req(f"p{i}") for i in range(12))
    qlock = threading.Lock()
    n_fast, n_slow = [0], [0]

    def _mk_pull(counter: List[int]):
        def _pull(_c: Any, _r: str) -> Optional[Sample]:
            with qlock:
                if not queue:
                    return None
                counter[0] += 1
                return queue.popleft()

        return _pull

    fast._pull = _mk_pull(n_fast)  # type: ignore[assignment]
    slow._pull = _mk_pull(n_slow)  # type: ignore[assignment]

    t_fast = threading.Thread(target=lambda: fast.run_drain(None, ""))
    t_slow = threading.Thread(target=lambda: slow.run_drain(None, ""))
    t_fast.start()
    t_slow.start()
    t_fast.join(timeout=10)
    t_slow.join(timeout=10)
    assert not t_fast.is_alive() and not t_slow.is_alive()

    assert n_fast[0] + n_slow[0] == 12  # every task processed exactly once
    assert n_fast[0] > n_slow[0]  # the fast worker pulled more (load balanced by capacity)
    fast.shutdown()
    slow.shutdown()


# --------------------------------------------------------------------------- #
# Partial rollout (LIN-531): turn-boundary checkpoint + resume + buffer routing
# --------------------------------------------------------------------------- #


class _StopAfterEnv(FakeEnv):
    """Flips ``engine._stopping`` True (once) during the step of turn ``stop_after``,
    so the NEXT turn-top check checkpoints — a deterministic mid-trajectory interrupt."""

    def __init__(self, engine: AgenticRolloutEngine, stop_after: int, **kw: Any) -> None:
        super().__init__(**kw)
        self._engine = engine
        self._stop_after = int(stop_after)
        self._fired = False

    def step(self, sample: Sample) -> Tuple[Optional[Texts], bool, dict]:
        out = super().step(sample)
        if not self._fired and len(sample.gen_parts()) == self._stop_after:
            self._engine._stopping = True
            self._fired = True
        return out


def test_run_one_checkpoints_on_stopping_then_resumes_to_completion():
    """THE partial-rollout case: ``_stopping`` flips mid-trajectory → ``_run_one``
    checkpoints at the next turn boundary (``done=False``, k turns, no ``env.step`` on
    the truncated turn); the carried partial resumes from turn k and runs to the env's
    terminal turn (``done=True``), so its turn count spans the checkpoint."""
    engine = _make_engine(n=1, max_turns=8)
    engine._env = _StopAfterEnv(engine, stop_after=2, turns_by_prompt={"p0": 5})

    partial, done = engine._run_one(_req("p0"))
    assert done is False  # checkpointed, not terminal
    assert len(partial.gen_parts()) == 2  # stopped at the turn boundary after turn 2

    engine._stopping = False  # the trainer clears the flag before resubmitting
    resumed, done2 = engine._run_one(partial)
    assert done2 is True
    assert len(resumed.gen_parts()) == 5  # 2 carried + 3 more = the env's terminal turn
    assert resumed.parts[0].sample_ids == ["p0"]  # same trajectory (root id preserved)
    engine.shutdown()


def test_drain_conserves_trajectories_under_mid_drive_stopping():
    """A background drain flipped to stopping mid-drive loses nothing: every submitted
    trajectory ends up completed, checkpointed, or still-queued (conservation), and any
    checkpointed one is a genuine partial (fewer than the env's turns)."""
    engine = _make_engine(n=1, cap=8, env=FakeEnv(default_turns=4, step_sleep=0.005))
    queue = deque(_req(r) for r in ("a", "b", "c", "d"))
    _patch_pull(engine, queue)

    drain = threading.Thread(target=lambda: engine.run_drain(None, ""))
    drain.start()
    time.sleep(0.03)  # let trajectories start + advance a turn or two
    engine._stopping = True
    drain.join(timeout=10)
    assert not drain.is_alive()

    accounted = len(engine._completed) + len(engine._checkpointed) + len(queue)
    assert accounted == 4  # nothing lost or duplicated
    for s in engine._checkpointed:
        assert len(s.gen_parts()) < 4  # a checkpoint is an unfinished (partial) trajectory
    engine.shutdown()


def test_worker_buffers_read_clear_and_reset():
    """``drain_completed`` / ``collect_carried`` read+clear their buffers (so ``poll``
    is incremental); ``reset_round`` clears both + the stop flag for the next drive."""
    engine = _make_engine(n=1, env=FakeEnv(default_turns=1))
    _patch_pull(engine, deque([_req("a"), _req("b")]))
    engine.run_drain(None, "")

    first = engine.drain_completed()
    assert len(first) == 2 and all(isinstance(s, Sample) for s in first)
    assert engine.drain_completed() == []  # cleared — a second poll sees nothing new
    assert engine.collect_carried() == []  # nothing checkpointed on a clean drive

    engine._stopping = True
    engine._checkpointed = [_req("x")]  # sentinel carried
    engine.reset_round()
    assert engine._stopping is False and engine._completed == [] and engine._checkpointed == []
    engine.shutdown()
