"""UniRL workload envelope → upstream replay input, with no outcome leakage (see the package README)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence, Tuple

from experimental.aisim.schema import (
    SCHEMA_VERSION,
    Manifest,
    WorkloadRequest,
    validate_workload,
)

# Upstream-facing field names. Kept in one place because the pinned simulator revision owns
# their meaning; this adapter never renames them locally.
REQUEST_ID = "request_id"
INPUT_TOKENS = "input_tokens"
OUTPUT_TOKENS = "output_tokens"
ARRIVAL_MS = "arrival_ms"
WAIT_FOR = "wait_for"
TOOL_WAIT_MS = "tool_wait_ms"


@dataclass(frozen=True)
class ConvertedWorkload:
    """One converted window: predictor input plus the UniRL-only sidecar."""

    contract: str
    requests: Tuple[Dict[str, Any], ...]
    sidecar: Dict[str, Any] = field(default_factory=dict)
    notes: Tuple[str, ...] = ()

    def to_json(self) -> str:
        """Serialise the converted workload and its sidecar."""
        return json.dumps(
            {
                "contract": self.contract,
                "requests": list(self.requests),
                "sidecar": self.sidecar,
                "notes": list(self.notes),
            },
            indent=1,
            sort_keys=True,
        )


def normalize_observed_offsets(requests: Sequence[WorkloadRequest]) -> Tuple[Tuple[float, ...], float]:
    """Return observed offsets rebased so the first submission is 0, plus the origin."""
    observed = [request for request in requests if request.arrival.kind == "observed"]
    if not observed:
        raise ValueError("normalize_observed_offsets: the window has no observed arrival to anchor on")
    origin = min(float(request.arrival.offset_ms) for request in observed)
    return tuple(float(request.arrival.offset_ms) - origin for request in observed), origin


def to_open_loop(manifest: Manifest, requests: Sequence[WorkloadRequest]) -> ConvertedWorkload:
    """Convert observed-arrival requests into the open-loop contract, adding no session/wait field."""
    manifest.validate()
    validate_workload(requests)
    if manifest.replay_contract != "open_loop_observed_arrival":
        raise ValueError(
            f"to_open_loop: manifest.replay_contract is {manifest.replay_contract!r}; "
            "an observed-arrival conversion must not be run under a dependency contract"
        )
    dependency = [request.request_id for request in requests if request.arrival.kind != "observed"]
    if dependency:
        raise ValueError(
            f"to_open_loop: request(s) {dependency[:5]} carry dependency arrivals; convert them with to_dependency"
        )
    if manifest.capture_status == "incomplete":
        raise ValueError(
            "to_open_loop: capture_status='incomplete' — an incomplete trace cannot produce a fidelity run"
        )

    offsets, origin = normalize_observed_offsets(requests)
    rows = []
    for request, offset in zip([r for r in requests if r.arrival.kind == "observed"], offsets):
        rows.append(
            {
                REQUEST_ID: request.request_id,
                INPUT_TOKENS: int(request.input_tokens),
                OUTPUT_TOKENS: int(request.output_tokens),
                ARRIVAL_MS: float(offset),
            }
        )
    sidecar = {
        "unirl_schema_version": SCHEMA_VERSION,
        "run_id": manifest.run_id,
        "arrival_origin_ms": origin,
        "requests": [
            {
                "request_id": request.request_id,
                "rollout_batch_id": request.rollout_batch_id,
                "group_id": request.group_id,
                "trajectory_id": request.trajectory_id,
                "turn_index": request.turn_index,
                "attempt_id": request.attempt_id,
                "engine_id": request.engine_id,
                "worker_id": request.worker_id,
                "policy_identity": request.policy_identity,
            }
            for request in requests
        ],
    }
    return ConvertedWorkload(
        contract="open_loop_observed_arrival",
        requests=tuple(rows),
        sidecar=sidecar,
        notes=("observed arrivals are preserved verbatim; no dependency or session field was added",),
    )


def to_dependency(manifest: Manifest, requests: Sequence[WorkloadRequest]) -> ConvertedWorkload:
    """Convert predecessor-driven requests, keeping measured service time out of the predictor input."""
    manifest.validate()
    validate_workload(requests)
    observed = [request.request_id for request in requests if request.arrival.kind == "observed"]
    if observed:
        raise ValueError(
            f"to_dependency: request(s) {observed[:5]} still carry observed arrivals; a dependency workload must not keep "
            "an arrival the simulator would also honour"
        )
    rows = []
    for request in requests:
        row: Dict[str, Any] = {
            REQUEST_ID: request.request_id,
            INPUT_TOKENS: int(request.input_tokens),
            OUTPUT_TOKENS: int(request.output_tokens),
        }
        if request.arrival.wait_for:
            row[WAIT_FOR] = list(request.arrival.wait_for)
            row[TOOL_WAIT_MS] = float(request.arrival.external_delay_ms)
        else:
            # First turn: the window releases it at an absolute offset instead of via a predecessor.
            row[ARRIVAL_MS] = float(request.arrival.offset_ms)
        rows.append(row)
    sidecar = {
        "unirl_schema_version": SCHEMA_VERSION,
        "run_id": manifest.run_id,
        "slot_semantics": "not_modelled",
        "barrier_semantics": "not_modelled",
    }
    return ConvertedWorkload(
        contract="dependency_arrival",
        requests=tuple(rows),
        sidecar=sidecar,
        notes=(
            "external_delay_ms is a measured external wait, never a difference of neighbouring submit timestamps",
            "trajectory slots and collector barriers are not represented in this contract",
        ),
    )


def result_sidecar(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Wrap measured outcomes for the report only; never merge this into a converter input."""
    return {
        "purpose": "report_only",
        "measured": [dict(row) for row in results],
    }


def assert_no_outcome_leakage(converted: ConvertedWorkload) -> None:
    """Fail if any measuring-outcome key reached the predictor-facing rows."""
    forbidden = {"submit_ms", "complete_ms", "first_token_ms", "latency_ms", "service_time_ms", "status"}
    for row in converted.requests:
        leaked = sorted(forbidden & set(row))
        if leaked:
            raise ValueError(f"predictor input carries measured-outcome key(s) {leaked} for {row.get(REQUEST_ID)!r}")


def describe(converted: ConvertedWorkload) -> str:
    """Short human summary for logs and PR evidence."""
    return f"{converted.contract}: {len(converted.requests)} request(s); notes={list(converted.notes)}"


__all__ = [
    "ConvertedWorkload",
    "assert_no_outcome_leakage",
    "describe",
    "normalize_observed_offsets",
    "result_sidecar",
    "to_dependency",
    "to_open_loop",
]
