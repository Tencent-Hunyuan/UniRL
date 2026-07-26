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
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol

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

    def size(self) -> int:
        return len(self._items)

    def evict_stale(
        self,
        *,
        current_version: Optional[int],
        max_staleness: Optional[int],
    ) -> int:
        """Evict groups older than the configured policy-version budget."""
        if max_staleness is None or current_version is None:
            return 0
        before = len(self._items)
        self._items = [item for item in self._items if int(current_version) - item.weight_version <= int(max_staleness)]
        return before - len(self._items)

    def max_age(self, current_version: int) -> int:
        """Oldest resident group's age in policy versions (0 when empty)."""
        if not self._items:
            return 0
        return max(int(current_version) - item.weight_version for item in self._items)

    def drain_freshest(
        self,
        n: int,
        *,
        current_version: Optional[int] = None,
        max_staleness: Optional[int] = None,
        has_signal: Optional[Callable[[RolloutResp], bool]] = None,
    ) -> Optional[List[BufferedRolloutGroup]]:
        """Pop the ``n`` freshest eligible groups, carrying leftovers forward.

        Eligibility is evaluated in this order: stale groups are evicted, then
        the optional signal predicate is applied, then groups are sorted by
        descending generation id.  ``has_signal`` defaults to ``None`` so this
        extraction does not change existing AR batch selection.

        Returns ``None`` without consuming eligible groups when fewer than ``n``
        remain after eviction/filtering.
        """
        return self._drain_ordered(
            n,
            current_version=current_version,
            max_staleness=max_staleness,
            has_signal=has_signal,
            newest_first=True,
        )

    def drain_oldest(
        self,
        n: int,
        *,
        current_version: Optional[int] = None,
        max_staleness: Optional[int] = None,
        has_signal: Optional[Callable[[RolloutResp], bool]] = None,
    ) -> Optional[List[BufferedRolloutGroup]]:
        """Pop the ``n`` oldest eligible groups to prevent resident starvation."""
        return self._drain_ordered(
            n,
            current_version=current_version,
            max_staleness=max_staleness,
            has_signal=has_signal,
            newest_first=False,
        )

    def _drain_ordered(
        self,
        n: int,
        *,
        current_version: Optional[int],
        max_staleness: Optional[int],
        has_signal: Optional[Callable[[RolloutResp], bool]],
        newest_first: bool,
    ) -> Optional[List[BufferedRolloutGroup]]:
        self.evict_stale(current_version=current_version, max_staleness=max_staleness)
        if has_signal is not None:
            self._items = [item for item in self._items if has_signal(item.resp)]
        if len(self._items) < n:
            return None
        self._items.sort(key=lambda item: item.gen_id, reverse=newest_first)
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
            gen_id=int(gen_id),
            weight_version=int(weight_version),
        )

    @staticmethod
    def is_ready(job: InflightGeneration) -> bool:
        ready, _ = ray.wait(
            job.refs,
            num_returns=len(job.refs),
            timeout=0,
        )
        return len(ready) == len(job.refs)

    @staticmethod
    def wait(job: InflightGeneration) -> None:
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
    """Single-threaded scheduler for complete, versioned rollout groups.

    ``prefetch_batches=None`` preserves the legacy freshness-first admission
    loop.  Bounded mode requires one fixed-yield generation per training batch,
    reserves capacity before request construction, and consumes resident groups
    oldest-first so prefetched work cannot starve until it becomes stale.
    """

    def __init__(
        self,
        dispatcher: GenerationDispatcher,
        *,
        groups_per_batch: int,
        groups_per_generation: Optional[int] = None,
        prefetch_batches: Optional[int] = None,
    ) -> None:
        if int(groups_per_batch) < 1:
            raise ValueError(f"groups_per_batch must be >= 1, got {groups_per_batch}")
        if prefetch_batches is not None and int(prefetch_batches) < 0:
            raise ValueError(f"prefetch_batches must be >= 0 or None, got {prefetch_batches}")
        if groups_per_generation is not None and int(groups_per_generation) < 1:
            raise ValueError(f"groups_per_generation must be >= 1, got {groups_per_generation}")
        if prefetch_batches is not None:
            if groups_per_generation is None:
                raise ValueError("bounded prefetch requires groups_per_generation")
            if int(groups_per_generation) != int(groups_per_batch):
                raise ValueError(
                    "bounded prefetch requires one generation to produce exactly one training batch "
                    "(groups_per_generation == groups_per_batch)"
                )
        self._dispatcher = dispatcher
        self._groups_per_batch = int(groups_per_batch)
        self._groups_per_generation = int(groups_per_generation) if groups_per_generation is not None else None
        self._prefetch_batches = int(prefetch_batches) if prefetch_batches is not None else None
        self._buffer = VersionedGroupBuffer()
        self._inflight: List[InflightGeneration] = []
        self._launch_id = 0
        self._reset_metrics()

    @property
    def inflight_count(self) -> int:
        return len(self._inflight)

    @property
    def launch_id(self) -> int:
        return self._launch_id

    @property
    def buffer_size(self) -> int:
        return self._buffer.size()

    def _reset_metrics(self) -> None:
        self._stat_admitted_jobs = 0
        self._stat_reaped_jobs = 0
        self._stat_buffered_groups = 0
        self._stat_evicted_stale_groups = 0
        self._stat_selected_groups = 0
        self._stat_selected_age_sum = 0
        self._stat_selected_age_max = 0
        self._stat_gen_wait_time_s = 0.0
        self._stat_quiesce_wait_time_s = 0.0

    def reset(self, start_id: int = 0) -> None:
        """Reset empty runtime state for a fresh or resumed trainer loop."""

        if self._inflight:
            raise RuntimeError("cannot reset AsyncRolloutScheduler with generations in flight")
        self._buffer = VersionedGroupBuffer()
        self._launch_id = int(start_id)
        self._reset_metrics()

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
        self._stat_admitted_jobs += 1

    def _complete(
        self,
        job: InflightGeneration,
        on_complete: CompleteGeneration,
    ) -> None:
        # Complete-or-nothing: collect + score + validate first, then mutate the
        # buffer once. A failed job remains in-flight and can be retried without
        # duplicating a prefix of its groups.
        resp = self._dispatcher.collect(job)
        groups = on_complete(job, resp)
        if self._prefetch_batches is not None and len(groups) != self._groups_per_generation:
            raise RuntimeError(
                f"generation {job.gen_id} produced {len(groups)} root group(s); "
                f"bounded prefetch reserved {self._groups_per_generation}"
            )
        prepared = [
            BufferedRolloutGroup(
                resp=group,
                weight_version=job.weight_version,
                gen_id=job.gen_id,
            )
            for group in groups
        ]
        self._buffer.put_all(prepared)
        self._stat_reaped_jobs += 1
        self._stat_buffered_groups += len(groups)

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
                # Keep the failed job so teardown/drain_all can retry it.
                # KeyboardInterrupt/SystemExit still propagate immediately.
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
                wait_start = time.perf_counter()
                self._dispatcher.wait(job)
                self._stat_quiesce_wait_time_s += time.perf_counter() - wait_start
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

    def _evict_stale(self, *, current_version: int, max_staleness: int) -> None:
        self._stat_evicted_stale_groups += self._buffer.evict_stale(
            current_version=current_version,
            max_staleness=max_staleness,
        )

    def _reserved_groups(self) -> int:
        """Completed groups plus fixed-yield reservations for running jobs."""
        if self._groups_per_generation is None:
            return self._buffer.size()
        return self._buffer.size() + self._groups_per_generation * len(self._inflight)

    def _top_up(
        self,
        *,
        ceiling: int,
        inflight_limit: int,
        capacity_groups: Optional[int],
        build_req: BuildRequest,
        weight_version: int,
    ) -> None:
        """Admit work without consuming data when the capacity gate is closed."""
        while self._launch_id < ceiling and len(self._inflight) < inflight_limit:
            if capacity_groups is not None:
                next_groups = self._groups_per_generation
                if next_groups is None or self._reserved_groups() + next_groups > capacity_groups:
                    break
            self._launch_one(build_req=build_req, weight_version=weight_version)

    def drain_metrics(self, *, current_version: int) -> Dict[str, float]:
        """Return and reset cumulative runtime metrics plus current depths."""
        selected_age_mean = (
            self._stat_selected_age_sum / self._stat_selected_groups if self._stat_selected_groups else 0.0
        )
        metrics = {
            "async/policy_version": float(current_version),
            "async/inflight_jobs": float(len(self._inflight)),
            "async/resident_groups": float(self._buffer.size()),
            "async/reserved_groups": float(self._reserved_groups()),
            "async/max_resident_age": float(self._buffer.max_age(current_version)),
            "async/admitted_jobs": float(self._stat_admitted_jobs),
            "async/reaped_jobs": float(self._stat_reaped_jobs),
            "async/buffered_groups": float(self._stat_buffered_groups),
            "async/selected_groups": float(self._stat_selected_groups),
            "async/selected_age_mean": float(selected_age_mean),
            "async/selected_age_max": float(self._stat_selected_age_max),
            "async/evicted_stale_groups": float(self._stat_evicted_stale_groups),
            "async/dropped_groups": 0.0,
            "async/prefetch_batches": float(self._prefetch_batches if self._prefetch_batches is not None else -1),
            "async/gen_wait_time_s": float(self._stat_gen_wait_time_s),
            "async/quiesce_wait_time_s": float(self._stat_quiesce_wait_time_s),
        }
        self._reset_metrics()
        return metrics

    def next_batch(
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
        hard_launch_ceiling: Optional[int] = None,
    ) -> List[BufferedRolloutGroup]:
        """Return the freshest full training batch, blocking only when needed.

        The launch ceiling is the load-bearing on-policy invariant: at
        ``max_staleness=0`` no generation is launched into a future weight-sync
        window.  In bounded mode, ``prefetch_batches`` caps additional
        batch-equivalents independently from ``max_inflight``.
        """

        interval = max(1, int(sync_interval))
        inflight_limit = max(1, int(max_inflight))
        stale = int(max_staleness)
        staleness_window = ((int(rollout_id) // interval) + 1 + stale) * interval
        ceilings = [int(num_rollouts), staleness_window]
        if hard_launch_ceiling is not None:
            ceilings.append(int(hard_launch_ceiling))
        ceiling = min(ceilings)

        while True:
            self._evict_stale(current_version=current_version, max_staleness=stale)
            fill_capacity = (
                self._groups_per_batch * (1 + self._prefetch_batches) if self._prefetch_batches is not None else None
            )
            self._top_up(
                ceiling=ceiling,
                inflight_limit=inflight_limit,
                capacity_groups=fill_capacity,
                build_req=build_req,
                weight_version=current_version,
            )

            self.reap_ready(on_complete)
            drain = self._buffer.drain_oldest if self._prefetch_batches is not None else self._buffer.drain_freshest
            picked = drain(self._groups_per_batch)
            if picked is not None:
                ages = [int(current_version) - item.weight_version for item in picked]
                self._stat_selected_groups += len(picked)
                self._stat_selected_age_sum += sum(ages)
                self._stat_selected_age_max = max(self._stat_selected_age_max, max(ages, default=0))
                if self._prefetch_batches is not None:
                    self._top_up(
                        ceiling=ceiling,
                        inflight_limit=inflight_limit,
                        capacity_groups=self._groups_per_batch * self._prefetch_batches,
                        build_req=build_req,
                        weight_version=current_version,
                    )
                return picked
            if self._inflight:
                wait_start = time.perf_counter()
                self._dispatcher.wait(self._inflight[0])
                self._stat_gen_wait_time_s += time.perf_counter() - wait_start
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
