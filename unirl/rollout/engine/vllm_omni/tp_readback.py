"""Envelope for returning one result per TP rank from a worker RPC.

``collective_rpc`` runs on every rank but only rank 0's reply survives, so
per-rank state must be gathered worker-side. Imports nothing: the worker and
driver processes cannot import each other's modules at module scope.
"""

from __future__ import annotations

from typing import Any

TP_RANK_RESULTS_KEY = "__diffrl_tp_rank_results__"

# ``AsyncOmniEngine`` and the diffusion multiprocess executor each wrap a
# stage's rank-0 reply in a singleton list, so the envelope arrives nested a
# couple of layers deep. Bounded so a genuinely single-element payload cannot
# be unwrapped past itself.
_MAX_TRANSPORT_NESTING = 4


def unwrap_tp_rank_readbacks(results: Any) -> Any:
    """Strip RPC transport wrappers and return the per-TP-rank list.

    Checks for the envelope before each unwrap step, so a genuine one-element
    payload is not unwrapped past itself. A payload with no envelope passes
    through, leaving ``_assert_loaded`` to report the shape.
    """
    value = results
    for _ in range(_MAX_TRANSPORT_NESTING):
        if isinstance(value, dict) and TP_RANK_RESULTS_KEY in value:
            per_rank = value[TP_RANK_RESULTS_KEY]
            if not isinstance(per_rank, (list, tuple)):
                raise RuntimeError(
                    f"Malformed TP-rank readback envelope: expected a list under "
                    f"{TP_RANK_RESULTS_KEY!r}, got {type(per_rank).__name__}."
                )
            return list(per_rank)
        if isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
            continue
        break
    return value


__all__ = ["TP_RANK_RESULTS_KEY", "unwrap_tp_rank_readbacks"]
