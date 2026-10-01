"""Versioned UniRL rollout trace contract for offline replay; stdlib only (see the package README)."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SCHEMA_VERSION = "unirl.rollout.trace.v1"

ARRIVAL_KINDS = ("observed", "after_dependencies")
TERMINATIONS = ("completed",)
UNSUPPORTED_TERMINATIONS = ("failed", "aborted", "partial", "retry", "cancelled")
CAPTURE_STATUS = ("complete", "partial", "incomplete")
REPLAY_CONTRACTS = ("open_loop_observed_arrival", "dependency_arrival")
CLOCK_UNITS = ("ms",)


@dataclass(frozen=True)
class Arrival:
    """When a request becomes submittable: an observed offset, or predecessor completion plus a wait."""

    kind: str
    offset_ms: Optional[float] = None
    wait_for: Tuple[str, ...] = ()
    external_delay_ms: Optional[float] = None

    def validate(self, *, request_id: str) -> None:
        """Reject an arrival that mixes the two kinds or hides a missing wait as zero."""
        if self.kind not in ARRIVAL_KINDS:
            raise ValueError(f"{request_id}: arrival.kind must be one of {list(ARRIVAL_KINDS)}, got {self.kind!r}")
        if self.kind == "observed":
            if self.offset_ms is None:
                raise ValueError(f"{request_id}: an observed arrival requires offset_ms")
            _finite_non_negative(f"{request_id}.arrival.offset_ms", self.offset_ms)
            if self.wait_for:
                raise ValueError(f"{request_id}: an observed arrival must not also declare wait_for")
        else:
            if not self.wait_for:
                # A first-turn request is released by the window itself, not by a predecessor.
                if self.offset_ms is None:
                    raise ValueError(
                        f"{request_id}: an after_dependencies arrival without a predecessor needs offset_ms "
                        "(the external release time)"
                    )
                _finite_non_negative(f"{request_id}.arrival.offset_ms", self.offset_ms)
                if self.external_delay_ms is not None:
                    raise ValueError(
                        f"{request_id}: a release offset already dates this request; do not also set external_delay_ms"
                    )
                return
            if self.external_delay_ms is None:
                # A missing wait is unknown, not zero: refuse to invent the delay.
                raise ValueError(
                    f"{request_id}: after_dependencies requires external_delay_ms (use 0.0 for a real zero)"
                )
            _finite_non_negative(f"{request_id}.arrival.external_delay_ms", self.external_delay_ms)


@dataclass(frozen=True)
class WorkloadRequest:
    """One backend request the simulator may schedule."""

    request_id: str
    rollout_batch_id: str
    group_id: str
    trajectory_id: str
    turn_index: int
    engine_id: str
    worker_id: str
    policy_identity: str
    input_tokens: int
    output_tokens: int
    arrival: Arrival
    termination: str = "completed"
    attempt_id: int = 0
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        """Check identity, lengths, status and arrival for one request."""
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"{self.request_id}: schema_version {self.schema_version!r} != {SCHEMA_VERSION!r}")
        for name in ("request_id", "rollout_batch_id", "group_id", "trajectory_id", "engine_id", "worker_id"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{self.request_id}: {name} must be a non-empty string")
        if not str(self.policy_identity).strip():
            raise ValueError(f"{self.request_id}: policy_identity is required (a frozen revision, not an update count)")
        for name in ("turn_index", "attempt_id"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{self.request_id}: {name} must be a non-negative int, got {value!r}")
        if isinstance(self.input_tokens, bool) or not isinstance(self.input_tokens, int) or self.input_tokens < 1:
            raise ValueError(f"{self.request_id}: input_tokens must be a positive int, got {self.input_tokens!r}")
        if isinstance(self.output_tokens, bool) or not isinstance(self.output_tokens, int) or self.output_tokens < 0:
            raise ValueError(f"{self.request_id}: output_tokens must be a non-negative int, got {self.output_tokens!r}")
        if self.termination in UNSUPPORTED_TERMINATIONS:
            raise ValueError(
                f"{self.request_id}: termination {self.termination!r} is not representable in {SCHEMA_VERSION}; "
                "a window with retries/aborts must be reported as unsupported instead of being trimmed"
            )
        if self.termination not in TERMINATIONS:
            raise ValueError(f"{self.request_id}: unknown termination {self.termination!r}")
        self.arrival.validate(request_id=self.request_id)


@dataclass(frozen=True)
class MeasuredResult:
    """Observed timing for one request — never an input to a predictor."""

    request_id: str
    clock_domain: str
    boundary: str
    submit_ms: float
    complete_ms: float
    status: str
    first_token_ms: Optional[float] = None

    def validate(self) -> None:
        """Check the clock domain, boundary and monotonicity of one measurement."""
        if not str(self.clock_domain).strip() or not str(self.boundary).strip():
            raise ValueError(f"{self.request_id}: measured results require a clock_domain and a boundary")
        _finite_non_negative(f"{self.request_id}.submit_ms", self.submit_ms)
        _finite_non_negative(f"{self.request_id}.complete_ms", self.complete_ms)
        if self.complete_ms < self.submit_ms:
            raise ValueError(f"{self.request_id}: complete_ms {self.complete_ms} precedes submit_ms {self.submit_ms}")
        if self.first_token_ms is not None:
            _finite_non_negative(f"{self.request_id}.first_token_ms", self.first_token_ms)
            if not self.submit_ms <= self.first_token_ms <= self.complete_ms:
                raise ValueError(f"{self.request_id}: first_token_ms is outside [submit_ms, complete_ms]")
        if self.status not in TERMINATIONS:
            raise ValueError(f"{self.request_id}: measured status {self.status!r} is not representable")


@dataclass(frozen=True)
class Manifest:
    """Configuration and provenance for one capture window."""

    run_id: str
    capture_status: str
    unirl_commit: str
    simulator_commit_or_version: str
    replay_contract: str
    backend_name: str
    model_revision: str
    model_dtype: str
    sampling_config_digest: str
    hardware_model: str
    topology_digest: str
    engine_count: int
    tp_size: int
    policy_identity: str
    measurement_boundary: str
    timestamp_unit: str
    privacy_mode: str
    redaction_policy: str
    capture_overhead_method: str
    schema_version: str = SCHEMA_VERSION
    profile_id: Optional[str] = None
    profile_coverage: Optional[str] = None
    extra: Mapping[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        """Reject an incomplete manifest or an unsupported contract before anything is simulated."""
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"manifest schema_version {self.schema_version!r} != {SCHEMA_VERSION!r}")
        required = (
            "run_id",
            "unirl_commit",
            "simulator_commit_or_version",
            "backend_name",
            "model_revision",
            "model_dtype",
            "sampling_config_digest",
            "hardware_model",
            "topology_digest",
            "policy_identity",
            "measurement_boundary",
            "privacy_mode",
            "redaction_policy",
            "capture_overhead_method",
        )
        missing = [name for name in required if not str(getattr(self, name)).strip()]
        if missing:
            raise ValueError(f"manifest is missing required field(s): {missing}")
        if self.capture_status not in CAPTURE_STATUS:
            raise ValueError(f"capture_status must be one of {list(CAPTURE_STATUS)}, got {self.capture_status!r}")
        if self.replay_contract not in REPLAY_CONTRACTS:
            raise ValueError(f"replay_contract must be one of {list(REPLAY_CONTRACTS)}, got {self.replay_contract!r}")
        if self.timestamp_unit not in CLOCK_UNITS:
            raise ValueError(f"timestamp_unit must be one of {list(CLOCK_UNITS)}, got {self.timestamp_unit!r}")
        for name in ("engine_count", "tp_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"manifest.{name} must be a positive int, got {value!r}")
        if self.profile_coverage == "full" and not self.profile_id:
            raise ValueError("profile_coverage='full' requires a profile_id")

    def predictor_view(self) -> Dict[str, Any]:
        """The manifest subset a predictor may read."""
        return asdict(self)


def validate_workload(requests: Sequence[WorkloadRequest]) -> None:
    """Validate one window: unique ids, known predecessors, acyclic dependencies."""
    requests = list(requests)
    seen: Dict[str, WorkloadRequest] = {}
    for request in requests:
        request.validate()
        if request.request_id in seen:
            raise ValueError(f"duplicate request_id {request.request_id!r}")
        seen[request.request_id] = request
    for request in requests:
        for predecessor in request.arrival.wait_for:
            if predecessor not in seen:
                raise ValueError(f"{request.request_id}: predecessor {predecessor!r} is not in this window")
            if predecessor == request.request_id:
                raise ValueError(f"{request.request_id}: self-referential wait_for")
    _assert_acyclic(seen)


def validate_measured(results: Sequence[MeasuredResult], requests: Sequence[WorkloadRequest]) -> None:
    """Require exactly-once measured coverage of the window."""
    wanted = {request.request_id for request in requests}
    got: Dict[str, int] = {}
    for result in results:
        result.validate()
        got[result.request_id] = got.get(result.request_id, 0) + 1
    duplicates = sorted(name for name, count in got.items() if count > 1)
    if duplicates:
        raise ValueError(f"measured results repeated for request(s) {duplicates[:5]}")
    unknown = sorted(set(got) - wanted)
    if unknown:
        # Report the unknown id first: it names the actual mismatch instead of a derived gap.
        raise ValueError(f"measured results reference unknown request(s): {unknown[:5]}")
    missing = sorted(wanted - set(got))
    if missing:
        raise ValueError(f"measured results missing for {len(missing)} request(s); first: {missing[:5]}")


def workload_to_jsonl(requests: Sequence[WorkloadRequest]) -> str:
    """Serialise the workload as JSON lines with tuple fields as lists."""
    return _to_jsonl(
        {
            **asdict(request),
            "arrival": {
                **asdict(request.arrival),
                "wait_for": list(request.arrival.wait_for),
            },
        }
        for request in requests
    )


def workload_from_jsonl(text: str) -> List[WorkloadRequest]:
    """Parse a workload JSONL payload back into requests."""
    out: List[WorkloadRequest] = []
    for line in _iter_lines(text):
        payload = json.loads(line)
        arrival = payload.pop("arrival")
        arrival["wait_for"] = tuple(arrival.get("wait_for") or ())
        out.append(WorkloadRequest(arrival=Arrival(**arrival), **payload))
    return out


def measured_to_jsonl(results: Sequence[MeasuredResult]) -> str:
    """Serialise measured outcomes as JSON lines."""
    return _to_jsonl(asdict(result) for result in results)


def assertion_tie_break(requests: Sequence[WorkloadRequest]) -> List[str]:
    """Deterministic submission order under equal arrival times (stable by request_id)."""
    indexed = list(enumerate(requests))
    indexed.sort(key=lambda pair: (pair[1].arrival.offset_ms if pair[1].arrival.kind == "observed" else 0.0, pair[0]))
    return [request.request_id for _, request in indexed]


def _iter_lines(text: str) -> Iterable[str]:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            yield stripped


def _to_jsonl(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows)


def _assert_acyclic(seen: Mapping[str, WorkloadRequest]) -> None:
    state: Dict[str, int] = {}

    def visit(node: str) -> None:
        if state.get(node) == 1:
            raise ValueError(f"dependency cycle through {node!r}")
        if state.get(node) == 2:
            return
        state[node] = 1
        for predecessor in seen[node].arrival.wait_for:
            visit(predecessor)
        state[node] = 2

    for name in seen:
        visit(name)


def _finite_non_negative(name: str, value: Any) -> float:
    """Reject bool/str/NaN/inf/negative numbers."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if number < 0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return number


__all__ = [
    "ARRIVAL_KINDS",
    "CAPTURE_STATUS",
    "CLOCK_UNITS",
    "REPLAY_CONTRACTS",
    "SCHEMA_VERSION",
    "TERMINATIONS",
    "UNSUPPORTED_TERMINATIONS",
    "Arrival",
    "Manifest",
    "MeasuredResult",
    "WorkloadRequest",
    "assertion_tie_break",
    "measured_to_jsonl",
    "validate_measured",
    "validate_workload",
    "workload_from_jsonl",
    "workload_to_jsonl",
]
