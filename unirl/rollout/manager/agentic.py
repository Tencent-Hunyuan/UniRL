"""AgenticManager — the agentic rollout task queue, as a rank-0 role (LIN-693).

Owns admission, acceptance and disposal; the engine runs one trajectory and holds no
decision. A background drive thread places and reaps continuously, so the slab keeps
working while the driver is inside an optimizer step.

Task identity is this class's ``uid``: ``Part.fork(1, ...)`` gives a prompt's ``n``
siblings identical ids, so a Sample-derived key would pin a whole GRPO group to one
replica.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Deque, Dict, List, Optional, Sequence, Tuple

import ray

from unirl.distributed.group.dispatch import Dispatch, Execute, distributed
from unirl.distributed.group.remote import Remote
from unirl.rollout.manager.buffers import PendingGroups, VersionedBuffer, root_of

if TYPE_CHECKING:
    from unirl.types.sample import Sample

logger = logging.getLogger(__name__)


@dataclass
class _Running:
    ref: Any
    replica: int
    uid: int


@dataclass
class _Carried:
    uid: int
    sample: "Sample"


@dataclass
class _Counters:
    carried: int = 0
    dropped: int = 0
    dropped_roots: int = 0
    discarded_siblings: int = 0
    placed: int = 0
    completed: int = 0


class AgenticManager(Remote):
    """Rank-0 owner of the agentic rollout task queue and its placement."""

    _component_name = "agentic_manager"

    def __init__(self, *, group_size: int, per_engine_inflight: int = 8) -> None:
        self._group_size = int(group_size)
        self._depth = int(per_engine_inflight)
        if self._depth < 1:
            raise ValueError(f"per_engine_inflight must be >= 1; got {per_engine_inflight}")

        self._engines: List[Any] = []
        self._engine_role: str = ""
        self._replicas: List[int] = []

        self._queue: Deque[Tuple[int, "Sample"]] = deque()
        self._running: List[_Running] = []
        self._carried: List[_Carried] = []
        self._affinity: Dict[int, int] = {}
        self._next_uid = 0

        self._pending = PendingGroups(self._group_size)
        self._buffer: VersionedBuffer[List["Sample"]] = VersionedBuffer()
        self._gen_id = 0
        self._weight_version = 0

        self._lock = threading.RLock()
        self._progress = threading.Condition(self._lock)
        self._drive: Optional[threading.Thread] = None
        self._stop_drive = threading.Event()
        self._counters = _Counters()

    @distributed(dispatch_mode=Dispatch.BROADCAST, execute_mode=Execute.RANK_ZERO)
    def set_engines(self, engine_handles: Sequence[Any], role_name: str, *, start_gen_id: int = 0) -> None:
        """Cache the Worker actor handles of the engine replicas to place work on.

        ``engine_handles`` must be the ENGINE REPLICAS — ``tp_rank == 0 and pp_rank == 0``.
        A non-tp-zero rank is a shell whose ``generate`` returns ``None``, and
        least-outstanding placement would preferentially feed it because failing instantly
        looks cheap. The driver filters with ``Handle.rank_infos`` before calling this.
        """
        self._engines = list(engine_handles)
        if not self._engines:
            raise ValueError("AgenticManager.set_engines: no engine replicas given")
        self._engine_role = str(role_name)
        self._replicas = list(range(len(self._engines)))
        self._gen_id = int(start_gen_id)

    @distributed(dispatch_mode=Dispatch.BROADCAST, execute_mode=Execute.RANK_ZERO)
    def submit(self, fresh: Sequence["Sample"], *, resume_carried: bool = True) -> None:
        """Enqueue single-trajectory tasks, resuming carried tails under their own uids."""
        with self._lock:
            if resume_carried and self._carried:
                self._queue.extend((c.uid, c.sample) for c in self._carried)
                self._carried = []
            for sample in fresh:
                self._next_uid += 1
                self._queue.append((self._next_uid, sample))
            self._progress.notify_all()
        self._ensure_drive()

    @distributed(dispatch_mode=Dispatch.BROADCAST, execute_mode=Execute.RANK_ZERO)
    def collect(
        self,
        n: int,
        *,
        max_staleness: Optional[int] = None,
        timeout_s: Optional[float] = None,
    ) -> List[List["Sample"]]:
        """Block until ``n`` complete GRPO groups are available within the staleness bound.

        ``max_staleness=None`` disables eviction, which is what the barrier path wants.
        """
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        with self._lock:
            while True:
                picked = self._buffer.drain_freshest(
                    n, current_version=self._weight_version, max_staleness=max_staleness
                )
                if picked is not None:
                    return picked
                if not self._running and not self._queue:
                    raise RuntimeError(
                        f"AgenticManager.collect: need {n} complete groups, have "
                        f"{self._buffer.size()}, and nothing is queued or in flight"
                    )
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError(f"AgenticManager.collect timed out waiting for {n} groups")
                self._progress.wait(timeout=1.0 if remaining is None else remaining)

    @distributed(dispatch_mode=Dispatch.BROADCAST, execute_mode=Execute.RANK_ZERO)
    def quiesce(self, *, tail_policy: str = "carry") -> List[str]:
        """Turn-boundary stop; returns the abandoned root ids, empty under ``carry``.

        Trajectories that become terminal during the quiesce are ingested rather than
        carried. Call before the trainer's weight sync and bump the version after it, so
        those groups keep the version they were generated under.

        The roots come back because ``_gt_by_root`` is trainer-side state; the counters
        are on :meth:`metrics` and the tails themselves stay resident here.
        """
        if tail_policy not in ("carry", "drop"):
            raise ValueError(f"tail_policy must be 'carry' or 'drop'; got {tail_policy!r}")

        self._stop_drive.set()
        drive, self._drive = self._drive, None
        if drive is not None:
            drive.join()

        self._fan_engines("set_stopping", (True,))
        try:
            with self._lock:
                for rec in self._running:
                    self._settle(rec, ray.get(rec.ref))
                self._running = []
                self._carried.extend(_Carried(uid, sample) for uid, sample in self._queue)
                self._queue.clear()
                dropped_roots = self._apply_tail_policy(tail_policy)
                self._progress.notify_all()
                return dropped_roots
        finally:
            self._fan_engines("set_stopping", (False,))
            self._stop_drive.clear()

    @distributed(dispatch_mode=Dispatch.BROADCAST, execute_mode=Execute.RANK_ZERO)
    def bump_weight_version(self) -> int:
        """Advance the version ledger after the trainer pushed weights.

        The push stays driver-side, so keep the two adjacent in the trainer. Refuses while
        anything is in flight: a weight update corrupts a live generation, and the SGLang
        native backend raises rather than waiting.
        """
        with self._lock:
            if self._running:
                raise RuntimeError(
                    f"bump_weight_version with {len(self._running)} trajectories in flight; quiesce() first"
                )
            self._weight_version += 1
            logger.info("AgenticManager: weight_version -> %d", self._weight_version)
            return self._weight_version

    @distributed(dispatch_mode=Dispatch.BROADCAST, execute_mode=Execute.RANK_ZERO)
    def metrics(self) -> Dict[str, float]:
        with self._lock:
            return {
                "buffer_groups": float(self._buffer.size()),
                "assembler_pending_roots": float(self._pending.size()),
                "weight_version": float(self._weight_version),
                "queued": float(len(self._queue)),
                "inflight": float(len(self._running)),
                "carried_trajectories": float(self._counters.carried),
                "dropped_trajectories": float(self._counters.dropped),
                "dropped_roots": float(self._counters.dropped_roots),
                "discarded_completed_trajectories": float(self._counters.discarded_siblings),
                "placed": float(self._counters.placed),
                "completed": float(self._counters.completed),
            }

    @distributed(dispatch_mode=Dispatch.BROADCAST, execute_mode=Execute.RANK_ZERO)
    def pop_evicted_roots(self) -> List[str]:
        """Root ids of groups the staleness bound evicted, for the trainer to forget."""
        with self._lock:
            return [root_of(group[0]) for group in self._buffer.pop_evicted() if group]

    def _ensure_drive(self) -> None:
        if self._drive is not None and self._drive.is_alive():
            return
        self._stop_drive.clear()
        self._drive = threading.Thread(target=self._drive_loop, name="agentic-manager-drive", daemon=True)
        self._drive.start()

    def _drive_loop(self) -> None:
        while not self._stop_drive.is_set():
            try:
                reaped = self._reap()
                with self._lock:
                    self._top_up()
                    if reaped:
                        self._progress.notify_all()
            except Exception:
                logger.exception("AgenticManager drive loop failed")
                with self._lock:
                    self._progress.notify_all()
                raise
            if not reaped:
                self._stop_drive.wait(0.005)

    def _reap(self) -> int:
        with self._lock:
            refs = [rec.ref for rec in self._running]
        if not refs:
            return 0
        ready, _ = ray.wait(refs, num_returns=len(refs), timeout=0)
        if not ready:
            return 0
        done = set(ready)
        with self._lock:
            still, settled = [], []
            for rec in self._running:
                (settled if rec.ref in done else still).append(rec)
            self._running = still
            for rec in settled:
                self._settle(rec, ray.get(rec.ref))
            self._counters.completed += len(settled)
            return len(settled)

    def _settle(self, rec: _Running, outcome: Tuple["Sample", bool]) -> None:
        """Route one finished call. A raised ref propagates: task-level faults are already
        terminal and NaN-marked by the engine, so a raise means infrastructure, and turning
        it into a NaN sample would mask worker loss as a bad trajectory."""
        sample, terminal = outcome
        if terminal:
            self._affinity.pop(rec.uid, None)
            self._ingest([sample])
        else:
            self._affinity[rec.uid] = rec.replica
            self._carried.append(_Carried(rec.uid, sample))

    def _top_up(self) -> None:
        """Place queued tasks on replicas with spare capacity; caller holds the lock.

        A carried tail returns to the replica that produced it, where its environment
        session lives and its prefix is cached. Fresh tasks go least-outstanding.
        """
        if not self._engines:
            return
        load = Counter(rec.replica for rec in self._running)
        while self._queue and any(load[i] < self._depth for i in self._replicas):
            uid, task = self._queue.popleft()
            replica = self._affinity.get(uid)
            if replica is None or load[replica] >= self._depth:
                replica = min(self._replicas, key=lambda i: load[i])
            ref = self._engines[replica].call.remote(self._engine_role, "run_trajectory", (task,), {})
            self._running.append(_Running(ref, replica, uid))
            load[replica] += 1
            self._counters.placed += 1
        assert len(self._running) <= self._depth * len(self._replicas)

    def _ingest(self, completed: List["Sample"]) -> None:
        if not completed:
            return
        self._pending.add_completed(completed)
        for group in self._pending.pop_complete_groups():
            self._buffer.put(group, weight_version=self._weight_version, gen_id=self._gen_id)
            self._gen_id += 1

    def _apply_tail_policy(self, tail_policy: str) -> List[str]:
        """Caller holds the lock. Logs the tail-depth histogram, which is what shows
        whether the commit-N actually skipped stragglers or just over-sampled evenly."""
        depths = Counter(len(c.sample.gen_parts()) for c in self._carried)
        logger.info(
            "AgenticManager: %s tail=%d trajectories, turns=%s",
            tail_policy,
            len(self._carried),
            dict(sorted(depths.items())),
        )
        if tail_policy == "carry":
            self._counters.carried += len(self._carried)
            return []

        roots = sorted({root_of(c.sample) for c in self._carried})
        for c in self._carried:
            self._affinity.pop(c.uid, None)
        self._counters.dropped += len(self._carried)
        self._counters.dropped_roots += len(roots)
        self._counters.discarded_siblings += self._pending.discard_roots(roots)
        self._carried = []
        return roots

    def _fan_engines(self, method: str, args: tuple) -> None:
        ray.get([w.call.remote(self._engine_role, method, args, {}) for w in self._engines])


__all__ = ["AgenticManager"]
