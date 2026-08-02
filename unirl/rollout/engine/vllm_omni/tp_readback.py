"""Envelope for returning one result per TP rank from a worker RPC.

vLLM-Omni's ``collective_rpc`` executes the method on every rank, but only
rank 0 owns a result queue — the other ranks' return values are dropped. A
worker method that needs to report per-rank state therefore has to gather it
itself and hand the whole set back inside rank 0's single reply.

Both halves of that envelope live here so the key cannot drift between the
worker process that writes it and the driver-side backend that reads it. This
module deliberately imports nothing: the worker side already pulls in torch and
vllm, while ``backends/native.py`` defers its vllm-omni import to keep the
driver process light, so neither can import the other at module scope.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

TP_RANK_RESULTS_KEY = "__diffrl_tp_rank_results__"

# ``AsyncOmniEngine`` and the diffusion multiprocess executor each wrap a
# stage's rank-0 reply in a singleton list, so the envelope arrives nested a
# couple of layers deep. Bounded so a genuinely single-element payload cannot
# be unwrapped past itself.
_MAX_TRANSPORT_NESTING = 4


def wrap_tp_rank_results(per_rank: Sequence[Any]) -> Dict[str, List[Any]]:
    """Package one entry per TP rank for the trip back through rank 0."""
    return {TP_RANK_RESULTS_KEY: list(per_rank)}


def unwrap_tp_rank_readbacks(results: Any) -> Any:
    """Strip RPC transport wrappers and return the per-TP-rank list.

    Checks for the envelope *before* each unwrap step, so a reply that is
    itself a one-element list is returned intact rather than being unwrapped
    into its only element. Payloads without the envelope (a worker build
    predating it, or a backend that never gathers) pass through unchanged.
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


__all__ = ["TP_RANK_RESULTS_KEY", "unwrap_tp_rank_readbacks", "wrap_tp_rank_results"]
