"""Buffering mechanisms shared by the rollout managers: freshness/staleness and GRPO
group assembly. Policy-free."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Generic, Iterable, List, Optional, Tuple, TypeVar

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


__all__ = ["PendingGroups", "VersionedBuffer", "root_of"]
