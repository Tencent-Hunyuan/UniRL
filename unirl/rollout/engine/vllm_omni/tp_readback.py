"""Envelope for returning one result per TP rank from a worker RPC.

vLLM-Omni's ``collective_rpc`` executes the method on every rank, but only
rank 0 owns a result queue — the other ranks' return values are dropped. A
worker method that needs to report per-rank state therefore has to gather it
itself and hand the whole set back inside rank 0's single reply.

The key lives here, rather than in either side, so it cannot drift between the
worker process that writes the envelope and the driver-side backend that reads
it. This module deliberately imports nothing: the worker side already pulls in
torch and vllm, while ``backends/native.py`` defers its vllm-omni import to keep
the driver process light, so neither can import the other at module scope.
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

    Checks for the envelope *before* each unwrap step, so a reply that is
    itself a one-element list is returned intact rather than being unwrapped
    into its only element.

    A payload with no envelope passes through unchanged rather than raising:
    it then reaches ``_assert_loaded``, whose existing shape check reports it
    precisely ("expected N TP rank readbacks, got ..."). Failing here instead
    would replace that message with a less specific one.
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
