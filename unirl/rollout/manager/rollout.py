from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Sequence

from unirl.rollout.manager.buffers import CompleteGroups, PendingGroups
from unirl.rollout.manager.dispatch import RolloutPool
from unirl.rollout.manager.filters import RolloutFilter, identity

if TYPE_CHECKING:
    from unirl.distributed.group.handle import Handle
    from unirl.types.sample import Sample

logger = logging.getLogger(__name__)


class RolloutManager:
    def __init__(
        self,
        rollout: "Handle",
        *,
        launchers: Sequence[Callable[["Sample"], Any]],
        capacities: Sequence[int],
        group_size: int,
        worker_max_concurrency: int = 0,
        filter_fn: RolloutFilter = identity,
    ) -> None:
        self._rollout = rollout
        self._pool = RolloutPool(
            launchers,
            capacities,
            worker_max_concurrency=worker_max_concurrency,
        )
        self._group_size = int(group_size)
        self._pending = PendingGroups(group_size)
        self._complete = CompleteGroups()
        self._filter = filter_fn
        self._published_version = 0
        self._closed = False

    def submit(self, tasks: List["Sample"]) -> None:
        self._ensure_open()
        self._pool.add(list(tasks))

    def collect(self, n: int, *, current_version: int) -> List[List["Sample"]]:
        self._ensure_open()
        n = int(n)
        if n <= 0:
            raise ValueError(f"collect count must be positive; got {n}")
        current_version = int(current_version)
        if current_version < 0:
            raise ValueError(f"current_version must be non-negative; got {current_version}")

        while True:
            self._route(self._resolve(self._pool.take_completed(block=False)), allow_suspended=False)
            self._filter_complete(current_version)
            selected = self._complete.take(n)
            if selected is not None:
                return selected
            if not self._pool.live:
                raise RuntimeError(f"needed {n} rollout groups, collected {self._complete.group_count}")
            self._route(self._resolve(self._pool.take_completed(block=True)), allow_suspended=False)

    def quiesce(self, *, current_version: int) -> List["Sample"]:
        self._ensure_open()
        current_version = int(current_version)
        if current_version < 0:
            raise ValueError(f"current_version must be non-negative; got {current_version}")
        undispatched = self._pool.pause()
        self._rollout.set_stopping(True)
        completed = self._resolve(self._pool.drain())
        self._rollout.set_stopping(False)

        suspended = self._route(completed, allow_suspended=True)
        candidates = [*undispatched, *suspended]
        tails_by_root: Dict[str, List["Sample"]] = defaultdict(list)
        carried = []
        for sample in candidates:
            roots = _roots_of(sample)
            if len(roots) == 1:
                tails_by_root[roots[0]].append(sample)
            elif self._keep_root([sample], current_version=current_version):
                carried.append(sample)

        for root, tails in tails_by_root.items():
            known = [*self._pending.get(root), *tails]
            if self._keep_root(known, current_version=current_version):
                carried.extend(tails)
            else:
                discarded = self._pending.discard(root)
                logger.info(
                    "rollout filter discarded incomplete root=%s tails=%d completed=%d",
                    root,
                    len(tails),
                    discarded,
                )
        return carried

    def sync_weights(self, weight_sync: object, *, output_version: int) -> int:
        self._ensure_open()
        self._route(self._resolve(self._pool.take_completed(block=False)), allow_suspended=False)
        inflight_count, ready_count = self.counts
        if inflight_count or ready_count or len(self._pending):
            raise RuntimeError(
                "sync_weights requires no queued, in-flight, completed, or partially grouped rollout work"
            )
        next_version = int(output_version)
        if next_version < self._published_version:
            raise ValueError(
                f"output_version must be monotonic; current={self._published_version}, next={next_version}"
            )
        weight_sync.sync()
        self._rollout.set_version(next_version)
        self._published_version = next_version
        return self._published_version

    @property
    def counts(self) -> tuple[int, int]:
        inflight_count, completed_count = self._pool.counts
        return inflight_count, completed_count + len(self._complete)

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._pool.live:
                self.quiesce(current_version=self._published_version)
        finally:
            self._pool.close()
            self._closed = True

    def _resolve(self, units: List[Any]) -> List[tuple[int, "Sample"]]:
        return [(unit.sequence, unit.pending.result()) for unit in units]

    def _route(self, results: List[tuple[int, "Sample"]], *, allow_suspended: bool) -> List["Sample"]:
        terminal_trajectories = []
        suspended = []
        for _, sample in results:
            status = sample.parts[-1].harness_status if sample.parts else None
            if status == "suspended":
                if not allow_suspended:
                    raise RuntimeError("trajectory suspended outside quiesce")
                suspended.append(sample)
            elif status is None:
                self._complete.add(self._batch_group_count(sample), [sample])
            else:
                self._require_stamped_generated_parts(sample)
                terminal_trajectories.append(sample)
        for group in self._pending.add(terminal_trajectories):
            self._complete.add(1, group)
        return suspended

    def _batch_group_count(self, sample: "Sample") -> int:
        roots = _roots_of(sample)
        if not sample.gen_parts():
            raise RuntimeError("completed batch rollout has no generated Parts")
        self._require_stamped_generated_parts(sample)
        descendants = Counter(sample.root_group_ids(-1))
        malformed = {root: descendants.get(root, 0) for root in roots if descendants.get(root, 0) != self._group_size}
        extra = set(descendants) - set(roots)
        if malformed or extra:
            raise RuntimeError(
                f"batch rollout fan-out does not match group_size={self._group_size}: "
                f"malformed={malformed}, extra_roots={sorted(extra)}"
            )
        return len(roots)

    def _filter_complete(self, current_version: int) -> None:
        self._complete.filter(lambda samples: self._apply_filter(samples, current_version=current_version))

    def _apply_filter(self, samples: List["Sample"], *, current_version: int) -> List["Sample"]:
        candidates = list(samples)
        kept = list(self._filter(list(candidates), current_version))
        candidate_ids = Counter(map(id, candidates))
        kept_ids = Counter(map(id, kept))
        if kept_ids - candidate_ids:
            raise RuntimeError("rollout filter returned a Sample outside its input")
        if any(count != 1 for count in kept_ids.values()):
            raise RuntimeError("rollout filter returned the same Sample more than once")
        return kept

    def _keep_root(self, samples: List["Sample"], *, current_version: int) -> bool:
        candidates = list(samples)
        kept = self._apply_filter(candidates, current_version=current_version)
        if kept and Counter(map(id, kept)) != Counter(map(id, candidates)):
            raise RuntimeError("rollout filter must retain or discard an entire incomplete root")
        return bool(kept)

    @staticmethod
    def _require_stamped_generated_parts(sample: "Sample") -> None:
        unstamped = [index for index, part in enumerate(sample.gen_parts()) if part.output_version is None]
        if unstamped:
            raise RuntimeError(f"completed rollout has unstamped generated Parts at indices {unstamped}")

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("RolloutManager is closed")


def _roots_of(sample: "Sample") -> List[str]:
    if not sample.parts:
        raise ValueError("rollout Sample has no Parts")
    roots = list(dict.fromkeys(sample.root_group_ids(0)))
    if not roots:
        raise ValueError("rollout Sample has no root ids")
    return roots


__all__ = ["RolloutManager"]
