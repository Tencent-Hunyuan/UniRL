"""Quantitative resident/streamed block placement planner; pure data in, plan out (see README)."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SCHEMA_VERSION = "unirl.offload.placement.v1"
PHASES = ("rollout", "replay_backward", "optimizer", "transition")
EXECUTOR_KINDS = ("native_fsdp", "two_slot_streaming")
RECORD_KINDS = ("shard", "full", "cast")


def _finite_non_negative(name: str, value: Any) -> float:
    """Reject NaN/inf/negative measurements instead of letting them poison a plan."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if number < 0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return number


def _non_negative_int(name: str, value: Any) -> int:
    """Reject bool/float/negative byte counts."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return int(value)


@dataclass(frozen=True)
class BlockSpec:
    """One streamable model block and the bytes it moves in each storage role."""

    block_id: str
    owner: str
    order: int
    shard_bytes: int  # local FSDP shard actually copied H2D for this block
    full_bytes: int  # all-gathered device bytes for this block
    cast_bytes: int = 0  # dtype-cast / packing buffer for this block
    phases: Tuple[str, ...] = PHASES
    alias_group: Optional[str] = None  # tied storage: members share one placement
    fixed: bool = False  # never streamed (root-owned leftover parameters)

    def validate(self) -> None:
        """Check one block's own field ranges."""
        if not str(self.block_id):
            raise ValueError("BlockSpec.block_id must be non-empty")
        if not str(self.owner):
            raise ValueError(f"BlockSpec[{self.block_id}].owner must be non-empty")
        _non_negative_int(f"BlockSpec[{self.block_id}].order", self.order)
        for name in ("shard_bytes", "full_bytes", "cast_bytes"):
            _non_negative_int(f"BlockSpec[{self.block_id}].{name}", getattr(self, name))
        unknown = sorted(set(self.phases) - set(PHASES))
        if unknown:
            raise ValueError(f"BlockSpec[{self.block_id}].phases has unknown phase(s) {unknown}; known: {list(PHASES)}")
        if len(set(self.phases)) != len(self.phases):
            # A repeated phase would count the block twice in that phase's transfer/compute accounting.
            raise ValueError(f"BlockSpec[{self.block_id}].phases repeats a phase: {list(self.phases)}")
        if not self.fixed and self.shard_bytes == 0 and self.full_bytes == 0:
            raise ValueError(f"BlockSpec[{self.block_id}] is streamable but declares no bytes")

    def staging_bytes(self) -> int:
        """Transient device bytes one live copy of this block occupies."""
        return int(self.shard_bytes) + int(self.full_bytes) + int(self.cast_bytes)


@dataclass(frozen=True)
class BlockInventory:
    """The auditable block list: identity, FSDP owner, invocation order, fixed/alias constraints."""

    blocks: Tuple[BlockSpec, ...]
    root_bytes: int = 0  # leftover non-block parameters pinned on device

    def validate(self) -> None:
        """Reject duplicate ids/orders and alias groups that cannot move as one unit."""
        _non_negative_int("BlockInventory.root_bytes", self.root_bytes)
        seen: Dict[str, BlockSpec] = {}
        orders: Dict[int, str] = {}
        for block in self.blocks:
            block.validate()
            if block.block_id in seen:
                raise ValueError(f"BlockInventory has duplicate block_id {block.block_id!r}")
            if block.order in orders:
                raise ValueError(
                    f"BlockInventory invocation order {block.order} is claimed by both "
                    f"{orders[block.order]!r} and {block.block_id!r}"
                )
            seen[block.block_id] = block
            orders[block.order] = block.block_id
        groups: Dict[str, List[BlockSpec]] = {}
        for block in self.blocks:
            if block.alias_group:
                groups.setdefault(block.alias_group, []).append(block)
        for name, members in groups.items():
            if len(members) < 2:
                raise ValueError(f"alias group {name!r} has one member; drop the group or name its tie")
            if any(member.fixed for member in members) and not all(member.fixed for member in members):
                raise ValueError(f"alias group {name!r} mixes fixed and streamable members")
            if len({(m.shard_bytes, m.full_bytes, m.cast_bytes) for m in members}) != 1:
                raise ValueError(f"alias group {name!r} members declare different byte footprints")

    def streamable(self) -> Tuple[BlockSpec, ...]:
        """Blocks eligible to be offloaded, in invocation order."""
        return tuple(sorted((b for b in self.blocks if not b.fixed), key=lambda b: b.order))

    def fixed_blocks(self) -> Tuple[BlockSpec, ...]:
        """Blocks that always stay resident."""
        return tuple(sorted((b for b in self.blocks if b.fixed), key=lambda b: b.order))

    def fingerprint(self) -> str:
        """Stable digest of the inventory, for the plan's provenance block."""
        payload = json.dumps(
            {
                "root_bytes": self.root_bytes,
                "blocks": [
                    dataclasses.asdict(block) for block in sorted(self.blocks, key=lambda b: (b.order, b.block_id))
                ],
            },
            sort_keys=True,
        )
        return f"{SCHEMA_VERSION}:{len(self.blocks)}:{hashlib.sha256(payload.encode()).hexdigest()[:12]}"


@dataclass(frozen=True)
class PhaseCost:
    """Measured per-rank cost of one phase with every block resident on device."""

    compute_ms: Mapping[str, float]  # block_id -> kernel time excluding parameter movement
    transfer_ms: Mapping[str, float]  # block_id -> measured H2D + all-gather + cast time for this shard
    resident_gpu_bytes: int  # per-rank high-water EXCLUDING all block parameter storage
    host_bytes: int = 0  # per-node host footprint excluding offloaded parameters
    startup_ms: float = 0.0  # first streamed block's cold preparation in this phase
    collective_ms: float = 0.0  # exposed collective time already not accounted in compute_ms
    measured: Tuple[str, ...] = ()  # block ids actually measured in this phase

    def validate(self, *, phase: str, blocks: set) -> None:
        """Reject unknown blocks and non-finite measurements."""
        for block_id, value in self.compute_ms.items():
            if block_id not in blocks:
                raise ValueError(f"PhaseProfile[{phase}].compute_ms has unknown block {block_id!r}")
            _finite_non_negative(f"PhaseProfile[{phase}].compute_ms[{block_id}]", value)
        for block_id, value in self.transfer_ms.items():
            if block_id not in blocks:
                raise ValueError(f"PhaseProfile[{phase}].transfer_ms has unknown block {block_id!r}")
            _finite_non_negative(f"PhaseProfile[{phase}].transfer_ms[{block_id}]", value)
        for name in ("resident_gpu_bytes", "host_bytes"):
            _non_negative_int(f"PhaseProfile[{phase}].{name}", getattr(self, name))
        for name in ("startup_ms", "collective_ms"):
            _finite_non_negative(f"PhaseProfile[{phase}].{name}", getattr(self, name))


@dataclass(frozen=True)
class PhaseProfile:
    """Calibrated measurements, per rank, with the cache-key material that produced them."""

    fingerprint: Mapping[str, str]
    per_rank: Mapping[str, Mapping[str, PhaseCost]]  # rank id -> phase -> cost
    node_of_rank: Mapping[str, str]
    gpu_budget_bytes: Mapping[str, int]  # rank -> per-rank device budget
    host_budget_bytes: Mapping[str, int]  # node -> pinned host budget
    bandwidth_record_kind: str = "shard"  # which storage role transfer_ms was measured on

    def validate(self, *, blocks: set) -> None:
        """Reject unknown phases/ranks and contradictory budget rows."""
        if self.bandwidth_record_kind not in RECORD_KINDS:
            raise ValueError(f"bandwidth_record_kind must be one of {list(RECORD_KINDS)}")
        if not self.per_rank:
            raise ValueError("PhaseProfile.per_rank is empty")
        for rank, phases in self.per_rank.items():
            if rank not in self.node_of_rank:
                raise ValueError(f"PhaseProfile.node_of_rank is missing rank {rank!r}")
            if rank not in self.gpu_budget_bytes:
                raise ValueError(f"PhaseProfile.gpu_budget_bytes is missing rank {rank!r}")
            unknown = sorted(set(phases) - set(PHASES))
            if unknown:
                raise ValueError(f"PhaseProfile[{rank}] has unknown phase(s) {unknown}; known: {list(PHASES)}")
            for phase, cost in phases.items():
                cost.validate(phase=f"{rank}/{phase}", blocks=blocks)
        for node, budget in self.host_budget_bytes.items():
            _non_negative_int(f"PhaseProfile.host_budget_bytes[{node}]", budget)
        for node in set(self.node_of_rank.values()):
            if node not in self.host_budget_bytes:
                raise ValueError(f"PhaseProfile.host_budget_bytes is missing node {node!r}")

    def missing_phases(self) -> Tuple[str, ...]:
        """Phases no rank measured; the planner treats these as missing evidence."""
        measured = {phase for phases in self.per_rank.values() for phase in phases}
        return tuple(phase for phase in PHASES if phase not in measured)


@dataclass(frozen=True)
class ExecutorCapability:
    """What the executor can actually guarantee, so the memory model matches it."""

    kind: str
    live_streamed_bound: Optional[int]  # max materialised streamed blocks, None = not guaranteed
    requires_contiguous_suffix: bool = True
    supports_phase_switch: bool = False
    double_buffer: bool = False
    notes: str = ""

    def validate(self) -> None:
        """Reject an executor whose guarantees are not a modelled kind."""
        if self.kind not in EXECUTOR_KINDS:
            raise ValueError(f"ExecutorCapability.kind must be one of {list(EXECUTOR_KINDS)}")
        if self.live_streamed_bound is not None:
            _non_negative_int("ExecutorCapability.live_streamed_bound", self.live_streamed_bound)
            if self.live_streamed_bound < 1:
                raise ValueError("ExecutorCapability.live_streamed_bound must be >= 1 when declared")


@dataclass(frozen=True)
class CandidateEvaluation:
    """One resident-prefix candidate, evaluated across every rank and phase."""

    k: int
    feasible: bool
    rejections: Tuple[str, ...]
    predicted_peak_gpu_bytes: Mapping[str, int]
    predicted_peak_host_bytes: Mapping[str, int]
    predicted_exposed_stall_ms: float
    predicted_total_ms: float
    transfer_busy_ms: float
    no_compute_transfer_ms: float


@dataclass(frozen=True)
class PlacementPlan:
    """The selected plan plus the audit trail of every candidate and rejection reason."""

    schema_version: str
    inventory_fingerprint: str
    profile_fingerprint: Mapping[str, str]
    executor_kind: str
    objective: str
    status: str  # FEASIBLE | INFEASIBLE
    resident_block_ids: Tuple[str, ...] = ()
    streamed_block_ids: Tuple[str, ...] = ()
    fixed_block_ids: Tuple[str, ...] = ()
    gpu_budget_bytes_by_rank: Mapping[str, int] = field(default_factory=dict)
    host_budget_bytes_by_node: Mapping[str, int] = field(default_factory=dict)
    predicted_peak_bytes: Mapping[str, int] = field(default_factory=dict)
    predicted_exposed_stall_ms: float = 0.0
    predicted_total_ms: float = 0.0
    transfer_busy_ms: float = 0.0
    no_compute_transfer_ms: float = 0.0
    latency_optimal_k: Optional[int] = None
    latency_optimal_total_ms: float = 0.0
    candidates: Tuple[Mapping[str, Any], ...] = ()
    rejected_candidates_with_reasons: Tuple[Mapping[str, Any], ...] = ()
    missing_evidence: Tuple[str, ...] = ()

    def to_json(self) -> str:
        """Serialise the plan for the evidence directory."""
        return json.dumps(dataclasses.asdict(self), indent=1, sort_keys=True)


def plan_placement(
    inventory: BlockInventory,
    profile: PhaseProfile,
    capability: ExecutorCapability,
    *,
    exposure_limit_ms: Optional[float] = None,
    headroom_bytes_by_rank: Optional[Mapping[str, int]] = None,
) -> PlacementPlan:
    """Enumerate resident prefixes and return the largest feasible offloaded suffix."""
    inventory.validate()
    profile.validate(blocks={block.block_id for block in inventory.blocks})
    capability.validate()
    if exposure_limit_ms is not None:
        _finite_non_negative("exposure_limit_ms", exposure_limit_ms)
    headroom = {
        rank: _non_negative_int(f"headroom[{rank}]", value) for rank, value in (headroom_bytes_by_rank or {}).items()
    }
    streamable = inventory.streamable()
    fixed = inventory.fixed_blocks()
    fixed_ids = tuple(block.block_id for block in fixed)
    fixed_bytes = int(inventory.root_bytes) + sum(block.staging_bytes() for block in fixed)
    missing = profile.missing_phases()

    evaluations: List[CandidateEvaluation] = []
    for k in range(len(streamable) + 1):
        resident = streamable[:k]
        streamed = streamable[k:]
        alias_split = _alias_violations(resident, streamed)
        order_violation = capability.requires_contiguous_suffix and any(
            block.order < max((b.order for b in resident), default=-1) for block in streamed
        )
        per_rank_peak: Dict[str, int] = {}
        per_node_host: Dict[str, int] = {}
        stalls: List[float] = []
        totals: List[float] = []
        busy_union: List[float] = []
        no_compute: List[float] = []
        rejections: List[str] = []
        if alias_split:
            rejections.append("alias group split across the resident/streamed boundary: " + ", ".join(alias_split))
        if order_violation:
            rejections.append("executor requires a contiguous invocation-order suffix")
        for rank, phases in profile.per_rank.items():
            rank_peak = 0
            rank_host = 0
            node = profile.node_of_rank[rank]
            for phase, cost in phases.items():
                invoked = tuple(block for block in streamed if phase in block.phases)
                phase_resident = tuple(block for block in resident if phase in block.phases)
                phase_stage = _live_staging_bytes(invoked, capability)
                peak = (
                    int(cost.resident_gpu_bytes)
                    + fixed_bytes
                    + _params_bytes(phase_resident)
                    + phase_stage
                    + headroom.get(rank, 0)
                )
                rank_peak = max(rank_peak, peak)
                rank_host = max(rank_host, int(cost.host_bytes))
                stall, total, busy, gap = _simulate_phase(
                    invocations=_invocation_order(phase_resident, invoked),
                    resident_ids={block.block_id for block in phase_resident},
                    cost=cost,
                    capability=capability,
                )
                stalls.append(stall)
                totals.append(total + float(cost.collective_ms))
                busy_union.append(busy)
                no_compute.append(gap)
            per_rank_peak[rank] = rank_peak
            per_node_host[node] = per_node_host.get(node, 0) + rank_host + _host_params_bytes(streamed, capability)
            budget = profile.gpu_budget_bytes[rank]
            if rank_peak > budget:
                rejections.append(f"rank {rank} predicted peak {rank_peak} > budget {budget}")
        for node, budget in profile.host_budget_bytes.items():
            predicted = per_node_host.get(node, 0)
            per_node_host[node] = predicted
            if predicted > budget:
                rejections.append(f"node {node} predicted pinned host {predicted} > budget {budget}")
        exposed = max(stalls) if stalls else 0.0
        if exposure_limit_ms is not None and exposed > exposure_limit_ms:
            rejections.append(f"predicted exposed stall {exposed:.1f} ms > limit {exposure_limit_ms:.1f} ms")
        if missing:
            rejections.append(f"profile is missing phase(s) {list(missing)}")
        evaluations.append(
            CandidateEvaluation(
                k=k,
                feasible=not rejections,
                rejections=tuple(rejections),
                predicted_peak_gpu_bytes=per_rank_peak,
                predicted_peak_host_bytes=per_node_host,
                predicted_exposed_stall_ms=exposed,
                predicted_total_ms=max(totals) if totals else 0.0,
                transfer_busy_ms=max(busy_union) if busy_union else 0.0,
                no_compute_transfer_ms=max(no_compute) if no_compute else 0.0,
            )
        )

    feasible = [candidate for candidate in evaluations if candidate.feasible]
    latency_optimal = min(evaluations, key=lambda c: c.predicted_total_ms) if evaluations else None
    latency_optimal_ms = latency_optimal.predicted_total_ms if latency_optimal else 0.0
    if not feasible:
        return PlacementPlan(
            schema_version=SCHEMA_VERSION,
            inventory_fingerprint=inventory.fingerprint(),
            profile_fingerprint=dict(profile.fingerprint),
            executor_kind=capability.kind,
            objective="max_offloaded_suffix",
            status="INFEASIBLE",
            fixed_block_ids=fixed_ids,
            gpu_budget_bytes_by_rank=dict(profile.gpu_budget_bytes),
            host_budget_bytes_by_node=dict(profile.host_budget_bytes),
            latency_optimal_k=latency_optimal.k if latency_optimal else None,
            latency_optimal_total_ms=latency_optimal_ms,
            candidates=_candidate_rows(evaluations),
            rejected_candidates_with_reasons=tuple(
                {"k": c.k, "reasons": list(c.rejections)} for c in evaluations if c.rejections
            ),
            missing_evidence=missing,
        )

    chosen = feasible[0]
    resident = streamable[: chosen.k]
    streamed = streamable[chosen.k :]
    return PlacementPlan(
        schema_version=SCHEMA_VERSION,
        inventory_fingerprint=inventory.fingerprint(),
        profile_fingerprint=dict(profile.fingerprint),
        executor_kind=capability.kind,
        objective="max_offloaded_suffix",
        status="FEASIBLE",
        resident_block_ids=tuple(block.block_id for block in resident),
        streamed_block_ids=tuple(block.block_id for block in streamed),
        fixed_block_ids=fixed_ids,
        gpu_budget_bytes_by_rank=dict(profile.gpu_budget_bytes),
        host_budget_bytes_by_node=dict(profile.host_budget_bytes),
        predicted_peak_bytes=dict(chosen.predicted_peak_gpu_bytes),
        predicted_exposed_stall_ms=chosen.predicted_exposed_stall_ms,
        predicted_total_ms=chosen.predicted_total_ms,
        transfer_busy_ms=chosen.transfer_busy_ms,
        no_compute_transfer_ms=chosen.no_compute_transfer_ms,
        latency_optimal_k=latency_optimal.k if latency_optimal else None,
        latency_optimal_total_ms=latency_optimal_ms,
        candidates=_candidate_rows(evaluations),
        rejected_candidates_with_reasons=tuple(
            {"k": c.k, "reasons": list(c.rejections)} for c in evaluations if c.rejections
        ),
        missing_evidence=missing,
    )


def _candidate_rows(evaluations: Sequence[CandidateEvaluation]) -> Tuple[Mapping[str, Any], ...]:
    """Compact audit row per candidate, so every rejection and prediction is inspectable."""
    return tuple(
        {
            "k": candidate.k,
            "feasible": candidate.feasible,
            "resident_prefix": candidate.k,
            "predicted_peak_gpu_bytes": dict(candidate.predicted_peak_gpu_bytes),
            "predicted_peak_host_bytes": dict(candidate.predicted_peak_host_bytes),
            "predicted_exposed_stall_ms": candidate.predicted_exposed_stall_ms,
            "predicted_total_ms": candidate.predicted_total_ms,
            "transfer_busy_ms": candidate.transfer_busy_ms,
            "no_compute_transfer_ms": candidate.no_compute_transfer_ms,
            "reasons": list(candidate.rejections),
        }
        for candidate in evaluations
    )


def _alias_violations(resident: Sequence[BlockSpec], streamed: Sequence[BlockSpec]) -> Tuple[str, ...]:
    """Alias groups whose members ended up on both sides of the boundary."""
    resident_groups = {block.alias_group for block in resident if block.alias_group}
    streamed_groups = {block.alias_group for block in streamed if block.alias_group}
    return tuple(sorted(resident_groups & streamed_groups))


def _params_bytes(blocks: Sequence[BlockSpec]) -> int:
    """Device parameter bytes of resident blocks (their local shards stay materialised)."""
    return sum(block.staging_bytes() for block in blocks)


def _host_params_bytes(blocks: Sequence[BlockSpec], capability: ExecutorCapability) -> int:
    """Pinned host bytes for the offloaded blocks, including the double-buffer copy."""
    factor = 2 if capability.double_buffer else 1
    return factor * sum(int(block.shard_bytes) for block in blocks)


def _live_staging_bytes(blocks: Sequence[BlockSpec], capability: ExecutorCapability) -> int:
    """Transient device bytes the executor may hold at once for these streamed blocks."""
    if not blocks:
        return 0
    bound = capability.live_streamed_bound
    if bound is None:
        # No measured residency guarantee: charge every streamed block, the conservative
        # upper bound. Never substitute a two-slot model for an unproven lifecycle.
        return sum(block.staging_bytes() for block in blocks)
    ordered = sorted(blocks, key=lambda block: block.staging_bytes(), reverse=True)
    return sum(block.staging_bytes() for block in ordered[:bound])


def _invocation_order(resident: Sequence[BlockSpec], streamed: Sequence[BlockSpec]) -> Tuple[BlockSpec, ...]:
    """Blocks of one phase in invocation order."""
    return tuple(sorted([*resident, *streamed], key=lambda block: block.order))


def _simulate_phase(
    *,
    invocations: Sequence[BlockSpec],
    resident_ids: set,
    cost: PhaseCost,
    capability: ExecutorCapability,
) -> Tuple[float, float, float, float]:
    """Serial copy-lane/compute-lane simulation; returns (stall, total, busy, no-compute gap)."""
    copy_free = 0.0
    compute_free = float(cost.startup_ms)
    stall = 0.0
    first_copy = None
    last_copy = 0.0
    compute_spans: List[Tuple[float, float]] = []
    for block in invocations:
        compute_ms = float(cost.compute_ms.get(block.block_id, 0.0))
        if block.block_id in resident_ids:
            ready = 0.0
        else:
            transfer_ms = float(cost.transfer_ms.get(block.block_id, 0.0))
            ready = copy_free + transfer_ms
            copy_free = ready
            first_copy = copy_free - transfer_ms if first_copy is None else first_copy
            last_copy = max(last_copy, copy_free)
        stall += max(0.0, ready - compute_free)
        start = max(compute_free, ready)
        compute_spans.append((start, start + compute_ms))
        compute_free = start + compute_ms
    busy = compute_free - float(cost.startup_ms)
    gap = 0.0
    if first_copy is not None:
        gap = _uncovered_intervals(compute_spans, first_copy, last_copy)
    return stall, compute_free, busy, gap


def _uncovered_intervals(spans: Sequence[Tuple[float, float]], start: float, end: float) -> float:
    """Length inside [start, end) not covered by any compute span."""
    if end <= start:
        return 0.0
    covered = sorted(spans)
    cursor = start
    total = 0.0
    for span_start, span_end in covered:
        if span_end <= cursor:
            continue
        if span_start >= end:
            break
        if span_start > cursor:
            total += min(span_start, end) - cursor
        cursor = max(cursor, min(span_end, end))
        if cursor >= end:
            break
    if cursor < end:
        total += end - cursor
    return total


__all__ = [
    "PHASES",
    "SCHEMA_VERSION",
    "BlockInventory",
    "BlockSpec",
    "CandidateEvaluation",
    "ExecutorCapability",
    "PhaseCost",
    "PhaseProfile",
    "PlacementPlan",
    "plan_placement",
]
