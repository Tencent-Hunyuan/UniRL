"""AgenticManager — trajectory-granular rollout manager over the rank-0 coordinator.

Groups are stamped at COMPLETION; the per-turn version spread inside a carried
trajectory is corrected per-token by each gen Part's own ``weight_version``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Iterable, List, Optional

from unirl.rollout.manager.buffers import PendingGroups, VersionedBuffer

if TYPE_CHECKING:
    from unirl.types.sample import Sample

logger = logging.getLogger(__name__)


class AgenticManager:
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

    def sync_weights(self, weight_sync: Any) -> int:
        """Push train weights via *weight_sync* and advance the version ledger.

        The only sanctioned weight-push path — pairing the push with the bump
        is what keeps the ledger truthful. Raises while a drive is active (a
        weight push must be decode-idle); a joined ``finalize_if_drained`` or
        ``quiesce`` ends the drive.
        """
        if self._drive_live:
            raise RuntimeError("sync_weights with a drive active; finalize or quiesce() first")
        weight_sync.sync()
        self._weight_version += 1
        logger.info("sync_weights: pushed train weights; weight_version -> %d", self._weight_version)
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
                "AgenticManager.submit: prior drive still live — wait for "
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
                self._buffer.put(group, weight_version=self._weight_version, gen_id=self._gen_id)
                self._gen_id += 1
        return len(completed)


__all__ = ["AgenticManager"]
