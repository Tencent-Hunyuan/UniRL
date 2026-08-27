"""Grouping buffers for the rollout manager: sibling assembly and completed FIFO chunks."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Deque, Dict, List, Optional

if TYPE_CHECKING:
    from unirl.types.sample import Sample


def roots_of(sample: "Sample") -> List[str]:
    """Ordered unique root ids of ``sample``; raises when it has none."""
    if not sample.parts:
        raise ValueError("rollout Sample has no Parts")
    roots = list(dict.fromkeys(sample.root_group_ids(0)))
    if not roots:
        raise ValueError("rollout Sample has no root ids")
    return roots


class PendingGroups:
    def __init__(self, group_size: int) -> None:
        self._group_size = int(group_size)
        if self._group_size <= 0:
            raise ValueError(f"group_size must be positive; got {group_size}")
        self._by_root: Dict[str, List["Sample"]] = {}

    def add(self, samples: List["Sample"]) -> List[List["Sample"]]:
        complete = []
        for sample in samples:
            roots = roots_of(sample)
            if len(roots) != 1:
                raise RuntimeError(f"terminal trajectory must carry exactly one root; got {roots}")
            siblings = self._by_root.setdefault(roots[0], [])
            siblings.append(sample)
            if len(siblings) > self._group_size:
                raise RuntimeError(f"root {roots[0]!r} has more than {self._group_size} terminal siblings")
            if len(siblings) == self._group_size:
                complete.append(self._by_root.pop(roots[0]))
        return complete

    def get(self, root: str) -> List["Sample"]:
        return list(self._by_root.get(root, []))

    def discard(self, root: str) -> int:
        return len(self._by_root.pop(root, []))

    def __len__(self) -> int:
        return len(self._by_root)


@dataclass(frozen=True)
class _CompleteChunk:
    group_count: int
    samples: List["Sample"]


class CompleteGroups:
    def __init__(self) -> None:
        self._chunks: Deque[_CompleteChunk] = deque()

    def add(self, group_count: int, samples: List["Sample"]) -> None:
        group_count = int(group_count)
        if group_count <= 0:
            raise ValueError(f"group_count must be positive; got {group_count}")
        if not samples:
            raise ValueError("complete rollout chunk must contain at least one Sample")
        self._chunks.append(_CompleteChunk(group_count, list(samples)))

    def filter(self, transform: Callable[[List["Sample"]], List["Sample"]]) -> None:
        """Apply a filter transform chunk-atomically; the caller has already validated the subset contract."""
        chunks = list(self._chunks)
        if not chunks:
            return
        candidates = [sample for chunk in chunks for sample in chunk.samples]
        if any(count != 1 for count in Counter(map(id, candidates)).values()):
            raise RuntimeError("rollout buffer contains the same Sample object more than once")
        kept = list(transform(list(candidates)))

        chunk_by_sample = {
            id(sample): chunk_index for chunk_index, chunk in enumerate(chunks) for sample in chunk.samples
        }
        positions: Dict[int, List[int]] = defaultdict(list)
        kept_by_chunk: Dict[int, List["Sample"]] = defaultdict(list)
        chunk_order = []
        for position, sample in enumerate(kept):
            chunk_index = chunk_by_sample[id(sample)]
            if chunk_index not in kept_by_chunk:
                chunk_order.append(chunk_index)
            positions[chunk_index].append(position)
            kept_by_chunk[chunk_index].append(sample)

        filtered: Deque[_CompleteChunk] = deque()
        for chunk_index in chunk_order:
            chunk = chunks[chunk_index]
            chunk_samples = kept_by_chunk[chunk_index]
            if len(chunk_samples) != len(chunk.samples):
                raise RuntimeError("rollout filter must retain or discard an entire logical root")
            chunk_positions = positions[chunk_index]
            if chunk_positions != list(range(chunk_positions[0], chunk_positions[0] + len(chunk_positions))):
                raise RuntimeError("rollout filter cannot interleave Samples from different logical roots")
            filtered.append(_CompleteChunk(chunk.group_count, chunk_samples))
        self._chunks = filtered

    def take(self, group_count: int) -> Optional[List[List["Sample"]]]:
        group_count = int(group_count)
        if group_count <= 0:
            raise ValueError(f"group_count must be positive; got {group_count}")
        if self.group_count < group_count:
            return None

        selected: List[List["Sample"]] = []
        selected_groups = 0
        while selected_groups < group_count:
            chunk = self._chunks.popleft()
            if selected_groups + chunk.group_count > group_count:
                self._split_front(chunk)
                continue
            selected.append(chunk.samples)
            selected_groups += chunk.group_count
        return selected

    @property
    def group_count(self) -> int:
        return sum(chunk.group_count for chunk in self._chunks)

    def __len__(self) -> int:
        return len(self._chunks)

    def _split_front(self, chunk: _CompleteChunk) -> None:
        if len(chunk.samples) != 1:
            raise RuntimeError(
                f"cannot split a {chunk.group_count}-group chunk containing {len(chunk.samples)} Samples"
            )
        groups = chunk.samples[0].split()
        if len(groups) != chunk.group_count:
            raise RuntimeError(
                f"batch chunk reported {chunk.group_count} roots but Sample.split produced {len(groups)}"
            )
        self._chunks.extendleft(_CompleteChunk(1, [sample]) for sample in reversed(groups))


__all__ = ["CompleteGroups", "PendingGroups", "roots_of"]
