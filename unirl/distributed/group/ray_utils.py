"""Controller-side helpers for collecting Ray actor task results."""

from __future__ import annotations

import logging
import time
from typing import Any, List, Optional, Sequence

import ray
from ray.exceptions import GetTimeoutError

logger = logging.getLogger(__name__)


def _poison_and_cancel(
    *,
    pool: Any,
    rank: int,
    role_name: str,
    method_name: str,
    error: BaseException,
    pending: Sequence[Any],
) -> None:
    try:
        pool.mark_poisoned(rank=rank, role_name=role_name, method_name=method_name, error=error)
    except Exception:
        logger.exception("Failed to mark DevicePool poisoned after actor task failure")

    for pending_ref in pending:
        try:
            # Actor tasks cannot be force-cancelled safely. This is
            # best-effort: a running task may continue until it exits.
            ray.cancel(pending_ref, force=False)
        except Exception:
            logger.debug("Failed to cancel pending actor task", exc_info=True)


def inspect_ready_actor_results(
    refs: Sequence[Any],
    *,
    pool: Any,
    role_name: str,
    method_name: str,
) -> bool:
    """True iff every rank succeeded; raise on a ready failure without waiting for peers."""
    ordered_refs = list(refs)
    if not ordered_refs:
        return True

    ready, pending = ray.wait(ordered_refs, num_returns=len(ordered_refs), timeout=0)
    if not ready:
        return False

    index_by_ref = {ref: index for index, ref in enumerate(ordered_refs)}
    for ready_ref in ready:
        try:
            ray.get(ready_ref)
        except Exception as error:
            _poison_and_cancel(
                pool=pool,
                rank=index_by_ref[ready_ref],
                role_name=role_name,
                method_name=method_name,
                error=error,
                pending=pending,
            )
            raise
    return not pending


def get_actor_results(
    refs: Sequence[Any],
    *,
    pool: Any,
    role_name: str,
    method_name: str,
    timeout: Optional[float] = None,
) -> List[Any]:
    """Collect refs in completion order, returning rank order; poison the pool on the first failure."""
    ordered_refs = list(refs)
    if not ordered_refs:
        return []

    index_by_ref = {ref: index for index, ref in enumerate(ordered_refs)}
    results: List[Any] = [None] * len(ordered_refs)
    pending = ordered_refs
    deadline = None if timeout is None else time.monotonic() + timeout

    while pending:
        remaining = None if deadline is None else max(deadline - time.monotonic(), 0.0)
        ready, pending = ray.wait(pending, num_returns=1, timeout=remaining)
        if not ready:
            raise GetTimeoutError(
                f"Timed out after {timeout}s waiting for {len(pending)} of "
                f"{len(ordered_refs)} {role_name}.{method_name} actor tasks"
            )
        ready_ref = ready[0]
        rank = index_by_ref[ready_ref]
        try:
            results[rank] = ray.get(ready_ref)
        except Exception as error:
            _poison_and_cancel(
                pool=pool,
                rank=rank,
                role_name=role_name,
                method_name=method_name,
                error=error,
                pending=pending,
            )
            raise

    return results
