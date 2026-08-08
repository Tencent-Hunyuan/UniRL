"""Bounded background dispatch of rollout tasks over per-slot launchers."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Deque, List, Optional, Sequence

if TYPE_CHECKING:
    from unirl.types.sample import Sample


Launch = Callable[["Sample"], Any]


@dataclass(frozen=True)
class _PendingUnit:
    sequence: int
    launcher: int
    task: "Sample"
    pending: Any


class RolloutPool:
    """Background dispatch thread keeping every launcher filled up to its capacity."""

    _PROBE_INTERVAL_S = 0.01

    def __init__(
        self,
        launchers: Sequence[Launch],
        capacities: Sequence[int],
    ) -> None:
        self._launchers = list(launchers)
        self._capacities = [int(capacity) for capacity in capacities]
        if not self._launchers:
            raise ValueError("RolloutPool requires at least one launcher")
        if len(self._launchers) != len(self._capacities):
            raise ValueError(
                f"RolloutPool launcher/capacity count mismatch: {len(self._launchers)} != {len(self._capacities)}"
            )
        if any(capacity <= 0 for capacity in self._capacities):
            raise ValueError(f"RolloutPool capacities must be positive; got {self._capacities}")

        self._queue: Deque[tuple[int, "Sample"]] = deque()
        self._running: List[_PendingUnit] = []
        self._completed: Deque[_PendingUnit] = deque()
        self._reserved = [0] * len(self._launchers)
        self._next_sequence = 0
        self._paused = True
        self._closed = False
        self._failure: Optional[BaseException] = None
        self._condition = threading.Condition()
        self._thread = threading.Thread(target=self._progress, name="rollout-pool", daemon=True)
        self._thread.start()

    def add(self, tasks: List["Sample"]) -> None:
        with self._condition:
            self._raise_if_unavailable()
            for task in tasks:
                self._queue.append((self._next_sequence, task))
                self._next_sequence += 1
            self._paused = False
            self._condition.notify_all()

    def pause(self) -> List["Sample"]:
        with self._condition:
            self._raise_if_failed()
            self._paused = True
            while any(self._reserved):
                self._condition.wait()
                self._raise_if_failed()
            tasks = [task for _, task in self._queue]
            self._queue.clear()
            self._condition.notify_all()
            return tasks

    def take_completed(self, *, block: bool) -> List[_PendingUnit]:
        with self._condition:
            while block and not self._completed and self._has_remote_work() and self._failure is None:
                self._condition.wait()
            self._raise_if_failed()
            completed = list(self._completed)
            self._completed.clear()
            return completed

    def drain(self) -> List[_PendingUnit]:
        with self._condition:
            while (self._running or any(self._reserved)) and self._failure is None:
                self._condition.wait()
            self._raise_if_failed()
            completed = list(self._completed)
            self._completed.clear()
            return completed

    @property
    def live(self) -> bool:
        with self._condition:
            self._raise_if_failed()
            return bool(self._queue or self._running or self._completed or any(self._reserved))

    @property
    def counts(self) -> tuple[int, int]:
        with self._condition:
            self._raise_if_failed()
            inflight = len(self._queue) + len(self._running) + sum(self._reserved)
            return inflight, len(self._completed)

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._paused = True
            self._queue.clear()
            self._condition.notify_all()
        self._thread.join()

    def _has_remote_work(self) -> bool:
        return bool(self._queue or self._running or any(self._reserved))

    def _raise_if_unavailable(self) -> None:
        self._raise_if_failed()
        if self._closed:
            raise RuntimeError("RolloutPool is closed")

    def _raise_if_failed(self) -> None:
        if self._failure is not None:
            raise self._failure

    def _progress(self) -> None:
        while True:
            with self._condition:
                if self._failure is not None:
                    return
                if self._closed and not self._running and not any(self._reserved):
                    return
                plan = self._plan_launches()
                running = list(self._running)
                if not plan and not running:
                    self._condition.wait()
                    continue

            if plan and not self._execute_launches(plan):
                return

            try:
                ready = [unit for unit in running if unit.pending.ready()]
            except BaseException as exc:
                self._record_failure(exc)
                return
            if not ready:
                if not plan:
                    with self._condition:
                        self._condition.wait(timeout=self._PROBE_INTERVAL_S)
                continue

            with self._condition:
                for unit in ready:
                    if unit not in self._running:
                        continue
                    self._running.remove(unit)
                    self._completed.append(unit)
                self._condition.notify_all()

    def _plan_launches(self) -> List[tuple[int, int, "Sample"]]:
        """Reserve launcher slots for queued tasks; caller holds the lock."""
        if self._paused or self._closed:
            return []
        load = list(self._reserved)
        for unit in self._running:
            load[unit.launcher] += 1
        plan = []
        # Most-free launcher first, so tasks spread across slots instead of filling slot 0.
        while self._queue:
            index = max(range(len(self._launchers)), key=lambda i: self._capacities[i] - load[i])
            if load[index] >= self._capacities[index]:
                break
            sequence, task = self._queue.popleft()
            plan.append((sequence, index, task))
            self._reserved[index] += 1
            load[index] += 1
        return plan

    def _execute_launches(self, plan: List[tuple[int, int, "Sample"]]) -> bool:
        """Run reserved launches outside the lock; False when a launch failed (pool poisoned)."""
        launched: List[_PendingUnit] = []
        failure: Optional[BaseException] = None
        for sequence, index, task in plan:
            try:
                pending = self._launchers[index](task)
            except BaseException as exc:
                failure = exc
                break
            launched.append(_PendingUnit(sequence, index, task, pending))

        with self._condition:
            for unit in launched:
                self._reserved[unit.launcher] -= 1
                self._running.append(unit)
            if failure is not None:
                remaining = plan[len(launched) :]
                for _, index, _ in remaining:
                    self._reserved[index] -= 1
                self._queue.extendleft((sequence, task) for sequence, _, task in reversed(remaining))
                if self._failure is None:
                    self._failure = failure
                self._paused = True
            self._condition.notify_all()
        return failure is None

    def _record_failure(self, exc: BaseException) -> None:
        with self._condition:
            if self._failure is None:
                self._failure = exc
            self._paused = True
            self._condition.notify_all()


def resolve_worker_inflight(
    requested: int,
    *,
    worker_max_concurrency: int,
    engine_concurrency: Optional[int],
) -> int:
    """Clamp per-worker rollout concurrency to actor and engine capacity."""
    window = max(1, int(requested))
    worker_limit = 1 if int(worker_max_concurrency) <= 1 else max(1, int(worker_max_concurrency) - 2)
    window = min(window, worker_limit)
    if engine_concurrency is not None:
        window = min(window, max(1, int(engine_concurrency)))
    return window


__all__ = ["RolloutPool", "resolve_worker_inflight"]
