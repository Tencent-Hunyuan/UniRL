"""Driver-side async rollout engines and their mechanisms (LIN-631).

The async half of the engine design: ``synchronous.py`` records the worker-side sync
contracts (``BaseRolloutEngine`` / ``SyncRolloutEngine``); this module records
the driver side — everything is single-threaded, lock-free, and ray-free
(non-blocking dispatch is ``Handle.launch_nowait``).

Mechanisms (policy-free — launch ceilings, reap/launch ordering, and step
loops live in the trainers):

- :class:`VersionedBuffer` — payload-agnostic freshness/staleness buffer.
- :class:`InflightPool` — non-blocking pool of distributed ``generate`` calls.

Engines, sharing one consumer surface (``poll`` / ``drain_freshest`` /
``pop_evicted`` / ``quiesce`` + engine-owned ``weight_version``):

- :class:`AsyncBatchRolloutEngine` — batch granularity over a single-turn
  engine slab; one ``submit`` is one non-blocking distributed ``generate``.
  ``(weight_version, gen_id)`` are stamped at LAUNCH.
- :class:`AsyncAgenticRolloutEngine` — trajectory granularity over the
  ``AgenticRolloutEngine`` rank-0 coordinator; ``submit`` fires a task-pool
  drive and completions stream in via ``poll``. ``(weight_version, gen_id)``
  are stamped at COMPLETION; the per-turn version spread inside a carried
  trajectory is corrected per-token by each gen Part's own ``weight_version``.

Submission is deliberately engine-specific (incompatible signatures and
stamping semantics); the consumer verbs above are what the async trainers
program against. The colocate barrier path (``AgenticTrainer``) keeps
calling ``rollout.generate(sample)[0]`` directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
)

if TYPE_CHECKING:
    from unirl.types.sample import Sample

logger = logging.getLogger(__name__)

T = TypeVar("T")


class VersionedBuffer(Generic[T]):
    """Payload-agnostic freshness buffer of ``(payload, weight_version, gen_id)`` items.

    Unifies the batch path's per-``Sample`` buffering and the agentic path's
    per-group (``List[Sample]``) buffering; stamping semantics belong to the
    caller (batch stamps at launch, agentic at completion).
    """

    def __init__(self) -> None:
        self._items: List[Tuple[T, int, int]] = []
        self._evicted: List[T] = []

    def put(self, payload: T, *, weight_version: int, gen_id: int) -> None:
        self._items.append((payload, int(weight_version), int(gen_id)))

    def size(self) -> int:
        return len(self._items)

    def drain_freshest(
        self,
        n: int,
        *,
        current_version: Optional[int] = None,
        max_staleness: Optional[int] = None,
    ) -> Optional[List[T]]:
        """Pop the ``n`` freshest eligible payloads, carrying leftovers forward.

        Over-stale items are evicted first (retrievable via :meth:`pop_evicted`),
        then remaining items are sorted by descending ``gen_id`` (stable — ties
        keep insertion order). Returns ``None`` without consuming anything when
        fewer than ``n`` remain after eviction.
        """
        if max_staleness is not None and current_version is not None:
            kept: List[Tuple[T, int, int]] = []
            for item in self._items:
                if current_version - item[1] <= max_staleness:
                    kept.append(item)
                else:
                    self._evicted.append(item[0])
            self._items = kept
        if len(self._items) < n:
            return None
        self._items.sort(key=lambda item: item[2], reverse=True)
        picked, self._items = self._items[:n], self._items[n:]
        return [payload for payload, _, _ in picked]

    def pop_evicted(self) -> List[T]:
        """Return and clear payloads rejected by the latest staleness checks."""
        evicted, self._evicted = self._evicted, []
        return evicted


Complete = Callable[[int, int, Any], None]


@dataclass(frozen=True)
class _InflightJob:
    gen_id: int
    weight_version: int
    pending: Any


class InflightPool:
    """Non-blocking pool of distributed ``generate`` launches on a rollout Handle.

    Mechanism only: launch ceilings, reap/launch ordering, and step loops are
    caller policy. Jobs are launched via ``Handle.launch_nowait`` and completed
    through ``complete(gen_id, weight_version, payload)`` — all of ``complete``'s
    fallible work must happen before it mutates caller state, because a job
    whose completion raises stays in flight for retry.
    """

    def __init__(self, rollout: Any, *, start_gen_id: int = 0) -> None:
        self._rollout = rollout
        self._next_gen_id = int(start_gen_id)
        self._jobs: List[_InflightJob] = []

    @property
    def next_gen_id(self) -> int:
        return self._next_gen_id

    def __len__(self) -> int:
        return len(self._jobs)

    def launch(self, sample: Any, *, weight_version: int) -> int:
        gen_id = self._next_gen_id
        pending = self._rollout.launch_nowait("generate", sample)
        self._jobs.append(_InflightJob(gen_id, int(weight_version), pending))
        self._next_gen_id += 1
        return gen_id

    def reap_ready(self, complete: Complete) -> int:
        """Complete every ready job; leave unresolved and failed jobs in flight.

        A job whose ``result()``/``complete`` raises stays in flight for retry;
        the first error re-raises after the sweep, later errors are logged.
        KeyboardInterrupt/SystemExit propagate immediately (not deferred behind
        remaining ready jobs). Returns the number completed.
        """
        still: List[_InflightJob] = []
        first_error: Optional[Exception] = None
        completed = 0
        for job in self._jobs:
            if not job.pending.ready():
                still.append(job)
                continue
            try:
                complete(job.gen_id, job.weight_version, job.pending.result())
                completed += 1
            except Exception as exc:
                still.append(job)
                if first_error is None:
                    first_error = exc
                else:
                    logger.error("reap_ready: additional failure for gen_id=%s", job.gen_id, exc_info=exc)
        self._jobs = still
        if first_error is not None:
            raise first_error
        return completed

    def drain_all(self, complete: Complete) -> int:
        """Quiesce: complete every job, blocking as needed. Same error contract as
        :meth:`reap_ready`."""
        jobs, self._jobs = self._jobs, []
        first_error: Optional[Exception] = None
        completed = 0
        for job in jobs:
            try:
                complete(job.gen_id, job.weight_version, job.pending.result())
                completed += 1
            except Exception as exc:
                self._jobs.append(job)
                if first_error is None:
                    first_error = exc
                else:
                    logger.error("drain_all: additional failure for gen_id=%s", job.gen_id, exc_info=exc)
        if first_error is not None:
            raise first_error
        return completed

    def wait_oldest(self) -> None:
        """Block until the oldest in-flight generation resolves, without collecting."""
        if self._jobs:
            self._jobs[0].pending.wait()


class AsyncBatchRolloutEngine:
    """Batch-granular async engine over a ``SyncRolloutEngine`` slab Handle; buffers ``Sample`` groups.

    ``complete(gen_id, completed) -> groups`` runs at reap time — scoring must
    precede training, and on transfer-sensitive backends the next launch. All
    of ``complete``'s fallible work happens before any buffer mutation, so a
    failed job stays in flight for retry without double-inserting groups.

    ``quiesce()`` (drain everything) is MANDATORY before a weight sync, eval, or
    checkpoint: a weight + KV update corrupts an in-flight generation.
    """

    def __init__(
        self,
        rollout: Any,
        *,
        complete: Callable[[int, "Sample"], List["Sample"]],
        start_gen_id: int = 0,
    ) -> None:
        self._complete = complete
        self._pool = InflightPool(rollout, start_gen_id=start_gen_id)
        self._buffer: VersionedBuffer["Sample"] = VersionedBuffer()
        self._weight_version = 0

    @property
    def weight_version(self) -> int:
        return self._weight_version

    def bump_weight_version(self) -> int:
        self._weight_version += 1
        return self._weight_version

    @property
    def next_gen_id(self) -> int:
        """The gen_id the next ``submit`` will get (1:1 with rollout ids)."""
        return self._pool.next_gen_id

    @property
    def inflight(self) -> int:
        return len(self._pool)

    def submit(self, sample: "Sample") -> int:
        """Launch one non-blocking distributed ``generate``; stamps the CURRENT version."""
        return self._pool.launch(sample, weight_version=self._weight_version)

    def poll(self) -> int:
        return self._pool.reap_ready(self._on_complete)

    def drain_freshest(self, n: int, *, max_staleness: int) -> Optional[List["Sample"]]:
        return self._buffer.drain_freshest(n, current_version=self._weight_version, max_staleness=max_staleness)

    def pop_evicted(self) -> List["Sample"]:
        return self._buffer.pop_evicted()

    def quiesce(self) -> List["Sample"]:
        self._pool.drain_all(self._on_complete)
        return []

    def wait_oldest(self) -> None:
        """Block until the oldest in-flight generation resolves (reap via ``poll``)."""
        self._pool.wait_oldest()

    def _on_complete(self, gen_id: int, weight_version: int, completed: "Sample") -> None:
        groups = self._complete(gen_id, completed)  # fallible (scoring) before any buffer put
        for group in groups:
            self._buffer.put(group, weight_version=weight_version, gen_id=gen_id)


def root_of(traj: "Sample") -> str:
    """Root id shared by a prompt's ``n`` sibling trajectories."""
    return traj.parts[0].sample_ids[0]


class PendingGroups:
    """Bucket a flat stream of terminal trajectories into complete GRPO groups.

    Poll returns variable-depth trajectory ``Sample``s; a prompt's ``n``
    siblings share its slash-free root id. A group is complete once all ``n``
    of a root's siblings are terminal. Variable-depth trajectories are NOT
    concatenated — a group stays a ``List[Sample]`` (the trainer flattens their
    gen Parts at train time).
    """

    def __init__(self, n: int) -> None:
        self._n = int(n)
        self._by_root: Dict[str, List["Sample"]] = {}

    def add_completed(self, trajs: List["Sample"]) -> None:
        for t in trajs:
            self._by_root.setdefault(root_of(t), []).append(t)

    def pop_complete_groups(self) -> List[List["Sample"]]:
        ready = [root for root, sibs in self._by_root.items() if len(sibs) >= self._n]
        out: List[List["Sample"]] = []
        for root in ready:
            sibs = self._by_root.pop(root)
            out.append(sibs[: self._n])
        return out

    def discard_roots(self, roots: Iterable[str]) -> int:
        """Drop incomplete buckets for abandoned roots; returns the number of
        terminal sibling trajectories that were being held for them."""
        discarded = 0
        for root in set(roots):
            discarded += len(self._by_root.pop(root, []))
        return discarded

    def size(self) -> int:
        return len(self._by_root)


class AsyncAgenticRolloutEngine:
    """Trajectory-granular async engine over the ``AgenticRolloutEngine``
    rank-0 coordinator Handle; buffers ``List[Sample]`` sibling groups.

    Normalizes the coordinator's BROADCAST+RANK_ZERO returns (every value
    unwraps ``[0]``). Groups are stamped at COMPLETION: ``weight_version`` is
    the engine's counter when a root's last sibling lands, ``gen_id`` a
    monotonic completed-group counter.

    ``submit`` requires the prior drive to be finalized or quiesced — two live
    drains would double-pull the coordinator queue.
    """

    def __init__(self, rollout: Any, *, group_size: int, start_gen_id: int = 0) -> None:
        self._rollout = rollout
        self._pending = PendingGroups(group_size)
        self._buffer: VersionedBuffer[List["Sample"]] = VersionedBuffer()
        self._gen_id = int(start_gen_id)
        self._weight_version = 0
        self._drive_live = False

    @property
    def weight_version(self) -> int:
        return self._weight_version

    def bump_weight_version(self) -> int:
        self._weight_version += 1
        return self._weight_version

    def submit(self, tasks: List["Sample"]) -> None:
        """Fire a background drive over a flat task list (fresh siblings + carried partials).

        Enforced double-pull guard: two live drains would double-pull the
        coordinator queue, so a second ``submit`` before ``finalize_if_drained``
        reported the drive done (or before ``quiesce``) raises instead of
        silently corrupting the drive.
        """
        if self._drive_live:
            raise RuntimeError(
                "AsyncAgenticRolloutEngine.submit: prior drive still live — wait for "
                "finalize_if_drained() to report it done or quiesce() first (a second "
                "drain would double-pull the coordinator queue)."
            )
        # Set before RPC so ambiguous submit failures remain guarded.
        self._drive_live = True
        self._rollout.submit(tasks)

    def poll(self) -> int:
        return self._ingest(self._rollout.poll()[0])

    def finalize_if_drained(self) -> Optional[int]:
        """``None`` while the in-flight drive is still running; otherwise join it
        and ingest its final completions — atomic with the readiness check (a
        separate poll would race the next ``submit``'s worker-buffer reset)."""
        completed = self._rollout.finalize_if_drained()[0]
        if completed is None:
            return None
        self._drive_live = False
        return self._ingest(completed)

    def drain_freshest(self, n: int, *, max_staleness: int) -> Optional[List[List["Sample"]]]:
        return self._buffer.drain_freshest(n, current_version=self._weight_version, max_staleness=max_staleness)

    def pop_evicted(self) -> List[List["Sample"]]:
        return self._buffer.pop_evicted()

    def quiesce(self) -> List["Sample"]:
        """Turn-boundary stop: abort, then one final poll for trajectories that
        completed DURING the quiesce (before the next ``submit`` resets worker
        buffers). Call before ``bump_weight_version`` so those groups carry the
        version they completed under."""
        carried = self._rollout.abort()[0]
        self.poll()
        self._drive_live = False
        return carried

    def discard_roots(self, roots: Iterable[str]) -> int:
        """Drop abandoned roots' incomplete pending buckets (tail-drop policy)."""
        return self._pending.discard_roots(roots)

    def pending_groups(self) -> int:
        """Roots with some-but-not-all siblings terminal (the pending backlog)."""
        return self._pending.size()

    def buffered_groups(self) -> int:
        return self._buffer.size()

    def _ingest(self, completed: List["Sample"]) -> int:
        if completed:
            self._pending.add_completed(completed)
            for group in self._pending.pop_complete_groups():
                self._buffer.put(group, weight_version=self._weight_version, gen_id=self._gen_id)
                self._gen_id += 1
        return len(completed)


def launch_ceiling(rollout_id: int, *, sync_interval: int, max_staleness: int, num_rollouts: int) -> int:
    """The batch trainers' on-policy launch clamp — trainer POLICY, defined once.

    A generation launched now is consumed later, so how far ahead the gen_id
    allocator may run is bounded to ``max_staleness`` weight-sync windows:
    ``max_staleness=0`` ⇒ never launch into a future sync-window ⇒ no
    generation crosses a sync ⇒ ``ratio≈1`` (on-policy).

    OWNERSHIP: this is trainer-side POLICY, not engine surface — its vocabulary
    (``rollout_id`` / ``sync_interval`` / ``num_rollouts``) is the trainers',
    the engine classes never call it, and it must never become an engine
    method. It is hosted in this module only because it is the two batch
    trainers' one shared torch-free home; the step loops that use it stay in
    the trainers as visible statement order.
    """
    return min(num_rollouts, ((rollout_id // sync_interval) + 1 + max_staleness) * sync_interval)


__all__ = [
    "AsyncAgenticRolloutEngine",
    "AsyncBatchRolloutEngine",
    "launch_ceiling",
    "root_of",
]
