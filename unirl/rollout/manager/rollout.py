from __future__ import annotations

import logging
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Deque, Dict, List, Sequence

from unirl.rollout.manager.buffers import PendingGroups
from unirl.rollout.manager.dispatch import RolloutPool
from unirl.rollout.manager.filters import RolloutFilter, identity

if TYPE_CHECKING:
    from unirl.distributed.group.handle import Handle
    from unirl.types.sample import Sample

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ReadyChunk:
    group_count: int
    samples: List["Sample"]


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
        self._buffer: Deque[_ReadyChunk] = deque()
        self._filter = filter_fn
        self._policy_version = 0
        self._closed = False

    def submit(self, tasks: List["Sample"]) -> None:
        self._ensure_open()
        self._pool.add(list(tasks))

    def collect(self, n: int) -> List[List["Sample"]]:
        self._ensure_open()
        n = int(n)
        if n <= 0:
            raise ValueError(f"collect count must be positive; got {n}")

        selected: List[_ReadyChunk] = []
        selected_groups = 0
        while selected_groups < n:
            self._route(self._resolve(self._pool.take_completed(block=False)), allow_suspended=False)
            self._filter_buffer()
            while self._buffer and selected_groups < n:
                chunk = self._buffer.popleft()
                if selected_groups + chunk.group_count > n:
                    self._split_front(chunk)
                    continue
                selected.append(chunk)
                selected_groups += chunk.group_count
            if selected_groups == n:
                return [chunk.samples for chunk in selected]
            if not self._pool.live:
                raise RuntimeError(f"needed {n} rollout groups, collected {selected_groups}")
            self._route(self._resolve(self._pool.take_completed(block=True)), allow_suspended=False)
        raise AssertionError("unreachable")

    def quiesce(self) -> List["Sample"]:
        self._ensure_open()
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
            elif self._keep_root([sample]):
                carried.append(sample)

        for root, tails in tails_by_root.items():
            known = [*self._pending.get(root), *tails]
            if self._keep_root(known):
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

    def sync_weights(self, weight_sync: object, *, policy_version: int) -> int:
        self._ensure_open()
        self._route(self._resolve(self._pool.take_completed(block=False)), allow_suspended=False)
        if self._pool.live:
            raise RuntimeError("sync_weights requires no queued or in-flight rollout work")
        next_version = int(policy_version)
        if next_version < self._policy_version:
            raise ValueError(f"policy_version must be monotonic; current={self._policy_version}, next={next_version}")
        weight_sync.sync()
        self._rollout.set_policy_version(next_version)
        self._policy_version = next_version
        return self._policy_version

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._pool.live:
                self.quiesce()
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
                self._buffer.append(_ReadyChunk(self._batch_group_count(sample), [sample]))
            else:
                terminal_trajectories.append(sample)
        for group in self._pending.add(terminal_trajectories):
            self._buffer.append(_ReadyChunk(1, group))
        return suspended

    def _batch_group_count(self, sample: "Sample") -> int:
        roots = _roots_of(sample)
        unstamped = [index for index, part in enumerate(sample.gen_parts()) if part.weight_version is None]
        if unstamped:
            raise RuntimeError(f"completed batch rollout has unstamped generated Parts at indices {unstamped}")
        descendants = Counter(sample.root_group_ids(-1))
        malformed = {root: descendants.get(root, 0) for root in roots if descendants.get(root, 0) != self._group_size}
        extra = set(descendants) - set(roots)
        if malformed or extra:
            raise RuntimeError(
                f"batch rollout fan-out does not match group_size={self._group_size}: "
                f"malformed={malformed}, extra_roots={sorted(extra)}"
            )
        return len(roots)

    def _split_front(self, chunk: _ReadyChunk) -> None:
        if len(chunk.samples) != 1:
            raise RuntimeError(
                f"cannot split a {chunk.group_count}-group chunk containing {len(chunk.samples)} Samples"
            )
        groups = chunk.samples[0].split()
        if len(groups) != chunk.group_count:
            raise RuntimeError(
                f"batch chunk reported {chunk.group_count} roots but Sample.split produced {len(groups)}"
            )
        self._buffer.extendleft(_ReadyChunk(1, [sample]) for sample in reversed(groups))

    def _filter_buffer(self) -> None:
        chunks = list(self._buffer)
        if not chunks:
            return
        candidates = [sample for chunk in chunks for sample in chunk.samples]
        kept = self._apply_filter(candidates)
        by_sample = {id(sample): index for index, sample in enumerate(candidates)}
        if len(by_sample) != len(candidates):
            raise RuntimeError("rollout buffer contains the same Sample object more than once")

        chunk_by_sample: Dict[int, int] = {}
        for chunk_index, chunk in enumerate(chunks):
            for sample in chunk.samples:
                chunk_by_sample[id(sample)] = chunk_index

        positions: Dict[int, List[int]] = defaultdict(list)
        kept_by_chunk: Dict[int, List["Sample"]] = defaultdict(list)
        chunk_order = []
        for position, sample in enumerate(kept):
            chunk_index = chunk_by_sample[id(sample)]
            if chunk_index not in kept_by_chunk:
                chunk_order.append(chunk_index)
            positions[chunk_index].append(position)
            kept_by_chunk[chunk_index].append(sample)

        filtered = []
        for chunk_index in chunk_order:
            chunk = chunks[chunk_index]
            chunk_samples = kept_by_chunk[chunk_index]
            if len(chunk_samples) != len(chunk.samples):
                raise RuntimeError("rollout filter must retain or discard an entire logical root")
            chunk_positions = positions[chunk_index]
            if chunk_positions != list(range(chunk_positions[0], chunk_positions[0] + len(chunk_positions))):
                raise RuntimeError("rollout filter cannot interleave Samples from different logical roots")
            filtered.append(_ReadyChunk(chunk.group_count, chunk_samples))
        self._buffer = deque(filtered)

    def _apply_filter(self, samples: List["Sample"]) -> List["Sample"]:
        candidates = list(samples)
        kept = list(self._filter(list(candidates), self._policy_version))
        candidate_ids = Counter(map(id, candidates))
        kept_ids = Counter(map(id, kept))
        if kept_ids - candidate_ids:
            raise RuntimeError("rollout filter returned a Sample outside its input")
        if any(count != 1 for count in kept_ids.values()):
            raise RuntimeError("rollout filter returned the same Sample more than once")
        return kept

    def _keep_root(self, samples: List["Sample"]) -> bool:
        candidates = list(samples)
        kept = self._apply_filter(candidates)
        if kept and Counter(map(id, kept)) != Counter(map(id, candidates)):
            raise RuntimeError("rollout filter must retain or discard an entire incomplete root")
        return bool(kept)

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
