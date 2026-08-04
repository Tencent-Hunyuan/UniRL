"""BatchManager — batch-granular async rollout over a single-turn engine slab.

One ``submit`` is one non-blocking distributed ``generate``; ``(weight_version, gen_id)``
are stamped at LAUNCH.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, List, Optional

from unirl.rollout.manager.buffers import VersionedBuffer

if TYPE_CHECKING:
    from unirl.types.sample import Sample

logger = logging.getLogger(__name__)

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


class BatchManager:
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


__all__ = ["BatchManager", "InflightPool", "launch_ceiling"]
