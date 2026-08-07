"""Driver-side async rollout engines: ``Handle.launch_nowait`` dispatch; trainer loops own admission and ordering."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Deque,
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
    """Freshness buffer of ``(payload, version, gen_id)`` items."""

    def __init__(self) -> None:
        self._items: List[Tuple[T, int, int]] = []
        self._evicted: List[T] = []

    def put(self, payload: T, *, version: int, gen_id: int) -> None:
        self._items.append((payload, version, gen_id))

    def size(self) -> int:
        return len(self._items)

    def drain_freshest(
        self,
        n: int,
        *,
        current_version: Optional[int] = None,
        staleness_budget: Optional[int] = None,
    ) -> Optional[List[T]]:
        """Pop the ``n`` newest eligible payloads, evicting over-stale ones; ``None`` (no consumption) if short."""
        if staleness_budget is not None and current_version is not None:
            kept: List[Tuple[T, int, int]] = []
            for item in self._items:
                staleness = current_version - item[1]
                if staleness < 0:
                    raise RuntimeError(
                        f"buffer item {item[2]} has future version {item[1]} > current version {current_version}"
                    )
                if staleness <= staleness_budget:
                    kept.append(item)
                else:
                    self._evicted.append(item[0])
            self._items = kept
        if len(self._items) < n:
            return None
        self._items.sort(key=lambda item: (item[1], item[2]), reverse=True)
        picked, self._items = self._items[:n], self._items[n:]
        return [payload for payload, _, _ in picked]

    def pop_evicted(self) -> List[T]:
        """Return and clear payloads rejected by the latest staleness checks."""
        evicted, self._evicted = self._evicted, []
        return evicted


@dataclass(frozen=True)
class RolloutBatch:
    """One atomic train batch produced by one batch-generation launch."""

    groups: List["Sample"]
    output_version: int
    gen_id: int


class RolloutBatchQueue:
    """Completion-order FIFO for single-turn batch generations."""

    def __init__(self) -> None:
        self._items: Deque[RolloutBatch] = deque()

    def put(self, batch: RolloutBatch) -> None:
        self._items.append(batch)

    def __len__(self) -> int:
        return len(self._items)

    def pop_next(
        self,
        *,
        train_version: int,
        staleness_budget: int,
    ) -> Optional[RolloutBatch]:
        if not self._items:
            return None
        item = self._items[0]
        staleness = train_version - item.output_version
        if staleness < 0:
            raise RuntimeError(
                f"generation {item.gen_id} has future output version "
                f"{item.output_version} > train version {train_version}"
            )
        if staleness > staleness_budget:
            raise RuntimeError(
                f"generation {item.gen_id} exceeded its staleness budget: "
                f"staleness={staleness} > budget={staleness_budget} optimizer updates"
            )
        return self._items.popleft()


CompleteGeneration = Callable[[int, int, Any], None]


@dataclass(frozen=True)
class _InflightJob:
    gen_id: int
    output_version: int
    pending: Any


class InflightPool:
    """Non-blocking pool of distributed ``generate`` launches on a rollout Handle.

    Mechanism only: launch admission, reap/launch ordering, and step loops are
    caller policy. Jobs are launched via ``Handle.launch_nowait`` and completed
    through ``complete(gen_id, output_version, payload)`` — all of ``complete``'s
    fallible work must happen before it mutates caller state, because a job
    whose completion raises stays in flight for retry.
    """

    def __init__(self, rollout: Any, *, start_gen_id: int = 0) -> None:
        self._rollout = rollout
        self._next_gen_id = start_gen_id
        self._jobs: List[_InflightJob] = []

    @property
    def next_gen_id(self) -> int:
        return self._next_gen_id

    def __len__(self) -> int:
        return len(self._jobs)

    def launch(self, sample: Any, *, output_version: int) -> int:
        gen_id = self._next_gen_id
        pending = self._rollout.launch_nowait("generate", sample)
        self._jobs.append(_InflightJob(gen_id, output_version, pending))
        self._next_gen_id += 1
        return gen_id

    def reap_ready(self, complete: CompleteGeneration) -> int:
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
                complete(job.gen_id, job.output_version, job.pending.result())
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

    def drain_all(self, complete: CompleteGeneration) -> int:
        """Quiesce: complete every job, blocking as needed. Same error contract as
        :meth:`reap_ready`."""
        jobs, self._jobs = self._jobs, []
        first_error: Optional[Exception] = None
        completed = 0
        for job in jobs:
            try:
                complete(job.gen_id, job.output_version, job.pending.result())
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
    """Batch-granular async engine over a ``SyncRolloutEngine`` slab Handle; one generation is one atomic FIFO batch."""

    def __init__(
        self,
        rollout: Any,
        *,
        process_completion: Callable[[int, "Sample"], List["Sample"]],
        groups_per_batch: int,
        start_gen_id: int = 0,
    ) -> None:
        if groups_per_batch < 1:
            raise ValueError(f"groups_per_batch must be >= 1, got {groups_per_batch}")
        self._process_completion = process_completion
        self._groups_per_batch = groups_per_batch
        self._pool = InflightPool(rollout, start_gen_id=start_gen_id)
        self._ready = RolloutBatchQueue()

    @property
    def next_gen_id(self) -> int:
        """The gen_id the next ``submit`` will get (1:1 with rollout ids)."""
        return self._pool.next_gen_id

    @property
    def inflight_count(self) -> int:
        return len(self._pool)

    @property
    def ready_count(self) -> int:
        return len(self._ready)

    def submit(self, sample: "Sample", *, output_version: int) -> int:
        """Launch one generation under the supplied output policy version."""
        return self._pool.launch(sample, output_version=output_version)

    def poll(self) -> int:
        return self._pool.reap_ready(self._on_complete)

    def pop_next_batch(
        self,
        *,
        train_version: int,
        staleness_budget: int,
    ) -> Optional[RolloutBatch]:
        return self._ready.pop_next(
            train_version=train_version,
            staleness_budget=staleness_budget,
        )

    def quiesce(self) -> None:
        """Drain every in-flight generation; MANDATORY before a weight sync, eval, or checkpoint."""
        self._pool.drain_all(self._on_complete)

    def wait_oldest(self) -> None:
        """Block until the oldest in-flight generation resolves (reap via ``poll``)."""
        self._pool.wait_oldest()

    def _on_complete(self, gen_id: int, output_version: int, completed: "Sample") -> None:
        groups = self._process_completion(gen_id, completed)  # fallible before any queue mutation
        if len(groups) != self._groups_per_batch:
            raise RuntimeError(
                f"generation {gen_id} produced {len(groups)} groups; expected groups_per_batch={self._groups_per_batch}"
            )
        self._ready.put(
            RolloutBatch(
                groups=groups,
                output_version=output_version,
                gen_id=gen_id,
            )
        )


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
        self._n = n
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
    unwraps ``[0]``). Groups are stamped with their oldest output version and a
    monotonic ``gen_id``.

    ``submit`` requires the prior drive to be finalized or quiesced — two live
    drains would double-pull the coordinator queue.
    """

    def __init__(self, rollout: Any, *, group_size: int, start_gen_id: int = 0) -> None:
        self._rollout = rollout
        self._pending = PendingGroups(group_size)
        self._buffer: VersionedBuffer[List["Sample"]] = VersionedBuffer()
        self._gen_id = start_gen_id
        self._version = 0
        self._drive_live = False

    @property
    def version(self) -> int:
        return self._version

    def sync_weights(self, weight_sync: Any, *, train_version: int) -> None:
        """Push train weights and set worker provenance to ``train_version``; raises while a drive is active."""
        if self._drive_live:
            raise RuntimeError("sync_weights with a drive active; finalize or quiesce() first")
        weight_sync.sync()
        self._rollout.set_version(train_version)
        self._version = train_version
        logger.info("sync_weights: pushed train weights; version=%d", self._version)

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

    def drain_freshest(self, n: int, *, staleness_budget: int) -> Optional[List[List["Sample"]]]:
        return self._buffer.drain_freshest(
            n,
            current_version=self._version,
            staleness_budget=staleness_budget,
        )

    def pop_evicted(self) -> List[List["Sample"]]:
        return self._buffer.pop_evicted()

    def quiesce(self) -> List["Sample"]:
        """Turn-boundary stop: abort, then one final poll for trajectories that
        completed DURING the quiesce (before the next ``submit`` resets worker
        buffers). Call before ``sync_weights`` so those groups carry the
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
                gen_parts = [part for traj in group for part in traj.gen_parts()]
                versions = [part.output_version for part in gen_parts]
                if any(version is None for version in versions):
                    raise RuntimeError("completed agentic group is missing output_version provenance")
                version = min(versions) if versions else self._version
                self._buffer.put(group, version=version, gen_id=self._gen_id)
                self._gen_id += 1
        return len(completed)


__all__ = [
    "AsyncAgenticRolloutEngine",
    "AsyncBatchRolloutEngine",
    "RolloutBatch",
    "root_of",
]
