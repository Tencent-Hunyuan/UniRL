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
    """Background dispatch thread keeping every launcher filled up to its capacity.

    Capacity frees when a launch reports ready; resolving results (and surviving a
    failure there) is the caller's job. A launch or probe failure poisons the pool.
    """

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
            while self._running and self._failure is None:
                self._condition.wait()
            self._raise_if_failed()
            completed = list(self._completed)
            self._completed.clear()
            return completed

    @property
    def live(self) -> bool:
        with self._condition:
            self._raise_if_failed()
            return bool(self._queue or self._running or self._completed)

    @property
    def counts(self) -> tuple[int, int]:
        with self._condition:
            self._raise_if_failed()
            return len(self._queue) + len(self._running), len(self._completed)

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
        return bool(self._queue or self._running)

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
                if self._closed and not self._running:
                    return
                if self._failure is not None:
                    return
                try:
                    self._launch_to_capacity()
                except BaseException as exc:
                    self._record_failure(exc)
                    return
                running = list(self._running)
                if not running:
                    self._condition.wait()
                    continue

            try:
                ready = [unit for unit in running if unit.pending.ready()]
            except BaseException as exc:
                self._record_failure(exc)
                return
            if not ready:
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

    def _launch_to_capacity(self) -> None:
        if self._paused or self._closed:
            return
        load = [0] * len(self._launchers)
        for unit in self._running:
            load[unit.launcher] += 1
        # Most-free launcher first, so tasks spread across slots instead of filling slot 0.
        while self._queue:
            index = max(range(len(self._launchers)), key=lambda i: self._capacities[i] - load[i])
            if load[index] >= self._capacities[index]:
                return
            sequence, task = self._queue.popleft()
            pending = self._launchers[index](task)
            self._running.append(_PendingUnit(sequence, index, task, pending))
            load[index] += 1

    def _record_failure(self, exc: BaseException) -> None:
        with self._condition:
            if self._failure is None:
                self._failure = exc
            self._paused = True
            self._condition.notify_all()


__all__ = ["RolloutPool"]
