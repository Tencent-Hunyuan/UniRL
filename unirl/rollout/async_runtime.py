"""Generic driver-side runtime for asynchronous rollout generation.

The runtime is deliberately policy- and trainer-agnostic.  It owns the
non-blocking Ray dispatch seam, in-flight generation bookkeeping, and the
versioned buffer of complete rollout groups.  Callers retain responsibility for
building requests, scoring responses, and training on the selected groups.

Everything here is single-threaded and lock-free.  A generation is always
completed before its groups enter :class:`VersionedGroupBuffer`; partial
trajectory scheduling belongs to a separate, resumable-engine abstraction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Protocol

import ray

from unirl.distributed.group.dispatch import DISPATCH_MODE_REGISTRY, Dispatch
from unirl.distributed.tensor import WorkerLocalTransport
from unirl.distributed.tensor.pytree import infer_batch_size
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BufferedRolloutGroup:
    """One complete rollout group plus the policy version that produced it."""

    resp: RolloutResp
    weight_version: int
    gen_id: int


class VersionedGroupBuffer:
    """Freshness-ordered buffer of complete, tree-preserving rollout groups."""

    def __init__(self) -> None:
        self._items: List[BufferedRolloutGroup] = []

    def put_all(self, items: List[BufferedRolloutGroup]) -> None:
        """Append a prepared batch of groups in one mutation."""

        self._items.extend(items)

    def drain_freshest(
        self,
        n: int,
        *,
        current_version: Optional[int] = None,
        max_staleness: Optional[int] = None,
    ) -> Optional[List[BufferedRolloutGroup]]:
        """Pop the ``n`` freshest eligible groups, carrying leftovers forward.

        Stale groups are evicted first, then remaining groups are sorted by
        descending generation id.

        Returns ``None`` without consuming eligible groups when fewer than ``n``
        remain after eviction.
        """

        if max_staleness is not None and current_version is not None:
            self._items = [item for item in self._items if current_version - item.weight_version <= max_staleness]
        if len(self._items) < n:
            return None
        self._items.sort(key=lambda item: item.gen_id, reverse=True)
        picked, self._items = self._items[:n], self._items[n:]
        return picked


@dataclass(frozen=True)
class InflightGeneration:
    """One non-blocking distributed ``generate`` invocation."""

    refs: List[Any]
    worker_local: bool
    req: RolloutReq
    gen_id: int
    weight_version: int


class GenerationDispatcher(Protocol):
    """Minimal dispatcher contract used by :class:`AsyncRolloutScheduler`."""

    def launch(
        self,
        req: RolloutReq,
        *,
        gen_id: int,
        weight_version: int,
    ) -> InflightGeneration: ...

    def is_ready(self, job: InflightGeneration) -> bool: ...

    def wait(self, job: InflightGeneration) -> None: ...

    def collect(self, job: InflightGeneration) -> RolloutResp: ...


class RayGenerationDispatcher:
    """Non-blocking ``DP_SCATTER`` dispatcher for a rollout ``Handle``.

    This intentionally mirrors the dispatch/localize/execute and
    rebind/collect halves of ``distributed/group/handle.py``'s ``handle_fn``.
    It therefore depends on the Handle's private ``_execute_all`` and
    ``_rebind_tree`` seams; changes to that implementation must update this
    adapter in lockstep.
    """

    def __init__(self, rollout_handle: Any) -> None:
        self._rollout = rollout_handle

    def launch(
        self,
        req: RolloutReq,
        *,
        gen_id: int,
        weight_version: int,
    ) -> InflightGeneration:
        rollout = self._rollout
        dispatch_fn = DISPATCH_MODE_REGISTRY[Dispatch.DP_SCATTER]["dispatch_fn"]
        batch_size = infer_batch_size((req,), {})
        if batch_size is not None and batch_size % rollout.dp_size != 0:
            raise ValueError(f"req batch_size={batch_size} not divisible by rollout dp_size={rollout.dp_size}")
        shards = dispatch_fn(rollout, (req,), {}, batch_size)
        worker_local = issubclass(
            rollout.pool.transport_cls,
            WorkerLocalTransport,
        )
        shards = rollout.pool.transport_cls.localize(
            shards,
            rollout.pool,
            rollout.device_ids,
            rollout.worker_ids,
        )
        refs = rollout._execute_all(
            "generate",
            shards,
            grad_mode=False,
            call_id=None,
        )
        return InflightGeneration(
            refs=refs,
            worker_local=worker_local,
            req=req,
            gen_id=gen_id,
            weight_version=weight_version,
        )

    def is_ready(self, job: InflightGeneration) -> bool:
        ready, _ = ray.wait(
            job.refs,
            num_returns=len(job.refs),
            timeout=0,
        )
        return len(ready) == len(job.refs)

    def wait(self, job: InflightGeneration) -> None:
        ray.get(job.refs)

    def collect(self, job: InflightGeneration) -> RolloutResp:
        rollout = self._rollout
        collect_fn = DISPATCH_MODE_REGISTRY[Dispatch.DP_SCATTER]["collect_fn"]
        results = ray.get(job.refs)
        results = [
            rollout._rebind_tree(
                result,
                rollout.workers[index],
                worker_local=job.worker_local,
            )
            for index, result in enumerate(results)
        ]
        return collect_fn(rollout, results)


BuildRequest = Callable[[int], RolloutReq]
CompleteGeneration = Callable[
    [InflightGeneration, RolloutResp],
    List[RolloutResp],
]


class AsyncRolloutScheduler:
    """Single-threaded scheduler for complete, versioned rollout groups."""

    def __init__(
        self,
        dispatcher: GenerationDispatcher,
        *,
        groups_per_step: int,
    ) -> None:
        if groups_per_step < 1:
            raise ValueError(f"groups_per_step must be >= 1, got {groups_per_step}")
        self._dispatcher = dispatcher
        self._groups_per_step = groups_per_step
        self._buffer = VersionedGroupBuffer()
        self._inflight: List[InflightGeneration] = []
        self._launch_id = 0

    def reset(self, start_id: int = 0) -> None:
        """Reset empty runtime state for a fresh or resumed trainer loop."""

        if self._inflight:
            raise RuntimeError("cannot reset AsyncRolloutScheduler with generations in flight")
        self._buffer = VersionedGroupBuffer()
        self._launch_id = start_id

    def _launch_one(
        self,
        *,
        build_req: BuildRequest,
        weight_version: int,
    ) -> None:
        gen_id = self._launch_id
        req = build_req(gen_id)
        self._inflight.append(
            self._dispatcher.launch(
                req,
                gen_id=gen_id,
                weight_version=weight_version,
            )
        )
        self._launch_id += 1

    def _complete(
        self,
        job: InflightGeneration,
        on_complete: CompleteGeneration,
    ) -> None:
        # Complete-or-nothing: collect + score first, then a single buffer
        # mutation. If either step fails the job stays in-flight for retry
        # without double-inserting groups from a partial put.
        resp = self._dispatcher.collect(job)
        groups = on_complete(job, resp)
        self._buffer.put_all(
            [
                BufferedRolloutGroup(
                    resp=group,
                    weight_version=job.weight_version,
                    gen_id=job.gen_id,
                )
                for group in groups
            ]
        )

    def reap_ready(self, on_complete: CompleteGeneration) -> None:
        """Collect every ready generation; leave unresolved / failed jobs in flight."""

        still: List[InflightGeneration] = []
        first_error: Optional[Exception] = None
        for job in self._inflight:
            if not self._dispatcher.is_ready(job):
                still.append(job)
                continue
            try:
                self._complete(job, on_complete)
            except Exception as exc:
                # Keep the failed job in-flight so finally/drain_all can retry
                # collect+score. KeyboardInterrupt/SystemExit propagate immediately
                # (not deferred behind remaining ready jobs).
                still.append(job)
                if first_error is None:
                    first_error = exc
                else:
                    logger.error(
                        "reap_ready: additional failure for gen_id=%s",
                        job.gen_id,
                        exc_info=exc,
                    )
        self._inflight = still
        if first_error is not None:
            raise first_error

    def drain_all(self, on_complete: CompleteGeneration) -> None:
        """Quiesce every generation and buffer all successfully completed groups."""

        jobs, self._inflight = list(self._inflight), []
        first_error: Optional[Exception] = None
        for job in jobs:
            try:
                self._complete(job, on_complete)
            except Exception as exc:
                self._inflight.append(job)
                if first_error is None:
                    first_error = exc
                else:
                    logger.error(
                        "drain_all: additional failure for gen_id=%s",
                        job.gen_id,
                        exc_info=exc,
                    )
        if first_error is not None:
            raise first_error

    def next_step(
        self,
        *,
        rollout_id: int,
        sync_interval: int,
        max_inflight: int,
        max_staleness: int,
        num_rollouts: int,
        current_version: int,
        build_req: BuildRequest,
        on_complete: CompleteGeneration,
    ) -> List[BufferedRolloutGroup]:
        """Return the freshest full training step, blocking only when needed.

        A step is ``groups_per_step`` complete rollout groups. The launch ceiling
        is the load-bearing on-policy invariant: at ``max_staleness=0`` no
        generation is launched into a future weight-sync window.

        ``sync_interval`` and ``max_inflight`` must already be ``>= 1``; callers
        (e.g. ``AsyncARTrainer``) clamp config before invoking this method.
        """

        if sync_interval < 1:
            raise ValueError(f"sync_interval must be >= 1, got {sync_interval}")
        if max_inflight < 1:
            raise ValueError(f"max_inflight must be >= 1, got {max_inflight}")
        while True:
            staleness_window = ((rollout_id // sync_interval) + 1 + max_staleness) * sync_interval
            ceiling = min(num_rollouts, staleness_window)
            while self._launch_id < ceiling and len(self._inflight) < max_inflight:
                self._launch_one(
                    build_req=build_req,
                    weight_version=current_version,
                )

            self.reap_ready(on_complete)
            picked = self._buffer.drain_freshest(
                self._groups_per_step,
                current_version=current_version,
                max_staleness=max_staleness,
            )
            if picked is not None:
                return picked
            if self._inflight:
                self._dispatcher.wait(self._inflight[0])
            else:
                raise RuntimeError("async rollout buffer underflow with no in-flight generations")


__all__ = [
    "AsyncRolloutScheduler",
    "BufferedRolloutGroup",
    "GenerationDispatcher",
    "InflightGeneration",
    "RayGenerationDispatcher",
    "VersionedGroupBuffer",
]
