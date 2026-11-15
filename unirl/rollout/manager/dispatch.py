"""Bounded background dispatch of rollout tasks over per-slot launchers."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Deque, Dict, List, Optional, Sequence

from unirl.rollout.manager.buffers import roots_of

if TYPE_CHECKING:
    from unirl.types.sample import Sample

_WORKER_CONTROL_CONCURRENCY = 2
AFFINITY_POLICIES = ("least_loaded", "group_packing", "cache_aware_scored")
KV_REPLICATION_MODES = ("none", "hicache_l2", "mooncake", "p2p_mirror")


Launch = Callable[["Sample"], Any]


def _single_root_id(task: "Sample") -> Optional[str]:
    """Return ``task``'s single root id when available, else None."""
    if not getattr(task, "parts", None):
        return None
    roots = roots_of(task)
    return roots[0] if len(roots) == 1 else None


def _estimate_task_tokens(task: "Sample") -> int:
    """Estimate prompt token count from ``task`` primitives for active-token budgeting."""
    parts = getattr(task, "parts", None)
    if not parts:
        return 1
    primitives = getattr(parts[0], "primitives", None) or {}
    text_prim = primitives.get("text")
    texts = getattr(text_prim, "texts", None)
    if texts:
        return max(1, sum(len(t) for t in texts) // 4)
    tokens = getattr(parts[0], "tokens", None)
    if tokens is not None and hasattr(tokens, "numel"):
        return max(1, int(tokens.numel()))
    return 1


@dataclass(frozen=True)
class _PendingUnit:
    sequence: int
    launcher: int
    task: "Sample"
    pending: Any
    root_id: Optional[str] = None
    est_tokens: int = 1


class RolloutPool:
    """Background dispatch thread keeping every launcher filled up to its capacity."""

    _PROBE_INTERVAL_S = 0.01

    def __init__(
        self,
        launchers: Sequence[Launch],
        capacities: Sequence[int],
        *,
        affinity_policy: str = "least_loaded",
        stagger_leader_prefill: bool = False,
        leader_stagger_ms: float = 100.0,
        balance_rel_threshold: Optional[float] = None,
        balance_abs_threshold: int = 4,
        max_inflight_tokens: Optional[int] = None,
        worker_node_ids: Optional[Sequence[int]] = None,
        prefix_affinity_weight: float = 1.0,
        load_weight: float = 0.5,
        active_tokens_weight: float = 0.3,
        kv_replication: str = "none",
        enable_work_stealing: Optional[bool] = None,
        kv_usage_fn: Optional[Callable[[], Sequence[float]]] = None,
        kv_replicator: Optional[Callable[[int, int, str], Any]] = None,
    ) -> None:
        if affinity_policy not in AFFINITY_POLICIES:
            raise ValueError(
                f"RolloutPool affinity_policy must be one of {AFFINITY_POLICIES}; got {affinity_policy!r}"
            )
        if kv_replication not in KV_REPLICATION_MODES:
            raise ValueError(
                f"RolloutPool kv_replication must be one of {KV_REPLICATION_MODES}; got {kv_replication!r}"
            )
        if leader_stagger_ms < 0:
            raise ValueError(f"leader_stagger_ms must be non-negative; got {leader_stagger_ms}")
        if balance_rel_threshold is not None and balance_rel_threshold < 1.0:
            raise ValueError(f"balance_rel_threshold must be >= 1.0 or None; got {balance_rel_threshold}")
        if balance_abs_threshold < 1:
            raise ValueError(f"balance_abs_threshold must be >= 1; got {balance_abs_threshold}")
        if max_inflight_tokens is not None and max_inflight_tokens <= 0:
            raise ValueError(f"max_inflight_tokens must be positive or None; got {max_inflight_tokens}")
        self._affinity_policy = affinity_policy
        self._stagger_leader_prefill = bool(stagger_leader_prefill)
        self._leader_stagger_s = float(leader_stagger_ms) / 1000.0
        self._balance_rel_threshold = float(balance_rel_threshold) if balance_rel_threshold is not None else None
        self._balance_abs_threshold = int(balance_abs_threshold)
        self._max_inflight_tokens = int(max_inflight_tokens) if max_inflight_tokens is not None else None
        self._prefix_affinity_weight = float(prefix_affinity_weight)
        self._load_weight = float(load_weight)
        self._active_tokens_weight = float(active_tokens_weight)
        self._kv_replication = kv_replication
        self._enable_work_stealing = (kv_replication != "none") if enable_work_stealing is None else bool(enable_work_stealing)
        self._kv_usage_fn = kv_usage_fn
        self._kv_replicator = kv_replicator
        self._launchers = list(launchers)
        self._capacities = [int(capacity) for capacity in capacities]
        if not self._launchers:
            raise ValueError("RolloutPool requires at least one launcher")
        if len(self._launchers) != len(self._capacities):
            raise ValueError(
                f"RolloutPool launcher/capacity count mismatch: {len(self._launchers)} != {len(self._capacities)}"
            )
        if any(capacity <= 0 for capacity in self._capacities):
            raise ValueError(f"RolloutPool capacities must be positive; got {self._capacities}")
        if worker_node_ids is not None:
            if len(worker_node_ids) != len(self._launchers):
                raise ValueError(
                    f"worker_node_ids length ({len(worker_node_ids)}) != launchers ({len(self._launchers)})"
                )
            self._worker_node_ids: Optional[List[int]] = [int(n) for n in worker_node_ids]
        else:
            self._worker_node_ids = None

        self._queue: Deque[tuple[int, "Sample"]] = deque()
        self._running: List[_PendingUnit] = []
        self._completed: Deque[_PendingUnit] = deque()
        self._reserved = [0] * len(self._launchers)
        self._reserved_tokens = [0] * len(self._launchers)
        self._running_tokens = [0] * len(self._launchers)
        self._next_sequence = 0
        self._paused = True
        self._closed = False
        self._failure: Optional[BaseException] = None
        self._dispatched_tasks = 0
        self._stagger_holds = 0
        self._kv_replication_hits = 0
        self._work_steals = 0
        self._root_launchers: Dict[str, set[int]] = {}
        self._root_task_counts: Dict[str, int] = {}
        self._leader_dispatched_at: Dict[str, float] = {}
        self._leader_primary_worker: Dict[str, int] = {}
        self._leader_completed: set[str] = set()
        self._replicated_roots: Dict[str, int] = {}
        self._condition = threading.Condition()
        self._thread = threading.Thread(target=self._progress, name="rollout-pool", daemon=True)
        self._thread.start()

    def add(self, tasks: List["Sample"]) -> None:
        with self._condition:
            self._raise_if_unavailable()
            for task in tasks:
                self._queue.append((self._next_sequence, task))
                self._next_sequence += 1
            self._paused = False
            self._condition.notify_all()

    def pause(self) -> List["Sample"]:
        with self._condition:
            self._raise_if_failed()
            self._paused = True
            while any(self._reserved):
                self._condition.wait()
                self._raise_if_failed()
            tasks = [task for _, task in self._queue]
            self._queue.clear()
            self._condition.notify_all()
            return tasks

    def take_completed(self, *, block: bool) -> List[_PendingUnit]:
        with self._condition:
            while block and not self._completed and self._has_remote_work() and self._failure is None:
                self._condition.wait()
            self._raise_if_failed()
            completed = list(self._completed)
            self._completed.clear()
            return completed

    def drain(self) -> List[_PendingUnit]:
        with self._condition:
            while (self._running or any(self._reserved)) and self._failure is None:
                self._condition.wait()
            self._raise_if_failed()
            completed = list(self._completed)
            self._completed.clear()
            return completed

    def run_to_completion(self, tasks: List["Sample"]) -> List[_PendingUnit]:
        """Run an isolated task prefix to completion while the pool is otherwise idle."""
        with self._condition:
            self._raise_if_unavailable()
            if self._queue or self._running or self._completed or any(self._reserved):
                raise RuntimeError("run_to_completion requires an idle RolloutPool")
            for task in tasks:
                self._queue.append((self._next_sequence, task))
                self._next_sequence += 1
            self._paused = False
            self._condition.notify_all()
            while (self._queue or self._running or any(self._reserved)) and self._failure is None:
                self._condition.wait()
            self._raise_if_failed()
            self._paused = True
            completed = list(self._completed)
            self._completed.clear()
            return completed

    @property
    def live(self) -> bool:
        with self._condition:
            self._raise_if_failed()
            return bool(self._queue or self._running or self._completed or any(self._reserved))

    @property
    def counts(self) -> tuple[int, int]:
        with self._condition:
            self._raise_if_failed()
            inflight = len(self._queue) + len(self._running) + sum(self._reserved)
            return inflight, len(self._completed)

    def dispatch_stats(self, *, reset: bool = False) -> Dict[str, float]:
        """Return per-root dispersion & sibling co-location stats for dispatched tasks."""
        with self._condition:
            self._raise_if_failed()
            groups_seen = len(self._root_launchers)
            if groups_seen > 0:
                mean_workers = sum(len(workers) for workers in self._root_launchers.values()) / groups_seen
                single_worker_rate = (
                    sum(1 for workers in self._root_launchers.values() if len(workers) == 1) / groups_seen
                )
                extra_tasks = sum(max(0, count - 1) for count in self._root_task_counts.values())
                co_located_extra = sum(
                    max(0, count - len(self._root_launchers[root]))
                    for root, count in self._root_task_counts.items()
                )
                co_located_rate = (co_located_extra / extra_tasks) if extra_tasks > 0 else 1.0
                if self._worker_node_ids is not None:
                    cross_node_groups = sum(
                        1
                        for workers in self._root_launchers.values()
                        if len({self._worker_node_ids[w] for w in workers}) > 1
                    )
                    cross_node_spill_rate = cross_node_groups / groups_seen
                else:
                    cross_node_spill_rate = 0.0
            else:
                mean_workers = 0.0
                single_worker_rate = 0.0
                co_located_rate = 0.0
                cross_node_spill_rate = 0.0
            stats = {
                "tasks_dispatched": float(self._dispatched_tasks),
                "groups_seen": float(groups_seen),
                "mean_workers_per_group": float(mean_workers),
                "group_single_worker_rate": float(single_worker_rate),
                "sibling_co_located_rate": float(co_located_rate),
                "cross_node_spill_rate": float(cross_node_spill_rate),
                "stagger_holds": float(self._stagger_holds),
                "kv_replication_hits": float(self._kv_replication_hits),
                "work_steals": float(self._work_steals),
            }
            if reset:
                self._dispatched_tasks = 0
                self._stagger_holds = 0
                self._kv_replication_hits = 0
                self._work_steals = 0
                self._root_launchers.clear()
                self._root_task_counts.clear()
                self._leader_dispatched_at.clear()
                self._leader_primary_worker.clear()
                self._leader_completed.clear()
                self._replicated_roots.clear()
            return stats

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._paused = True
            self._queue.clear()
            self._condition.notify_all()
        self._thread.join()
        with self._condition:
            pending = [*self._running, *self._completed]
            self._running.clear()
            self._completed.clear()
        for unit in pending:
            unit.pending.discard_on_completion()

    def _has_remote_work(self) -> bool:
        return bool(self._queue or self._running or any(self._reserved))

    def _raise_if_unavailable(self) -> None:
        self._raise_if_failed()
        if self._closed:
            raise RuntimeError("RolloutPool is closed")

    def _raise_if_failed(self) -> None:
        if self._failure is not None:
            raise self._failure

    def _next_stagger_wait_s(self) -> Optional[float]:
        """Return remaining stagger sleep in seconds when followers are waiting, else None."""
        if not self._stagger_leader_prefill or not self._queue or self._leader_stagger_s <= 0:
            return None
        now = time.monotonic()
        earliest: Optional[float] = None
        for _, task in self._queue:
            root_id = _single_root_id(task)
            if root_id is None or root_id in self._leader_completed:
                continue
            dispatched_at = self._leader_dispatched_at.get(root_id)
            if dispatched_at is None:
                continue
            rem = (dispatched_at + self._leader_stagger_s) - now
            if rem > 0 and (earliest is None or rem < earliest):
                earliest = rem
        return earliest

    def _progress(self) -> None:
        while True:
            with self._condition:
                if self._failure is not None:
                    return
                if self._closed:
                    return
                plan = self._plan_launches()
                running = list(self._running)
                if not plan and not running:
                    wait_s = self._next_stagger_wait_s()
                    self._condition.wait(timeout=wait_s)
                    continue

            if plan and not self._execute_launches(plan):
                return

            try:
                ready = [unit for unit in running if unit.pending.ready()]
            except BaseException as exc:
                self._record_failure(exc)
                return
            if not ready:
                if not plan:
                    with self._condition:
                        wait_s = self._next_stagger_wait_s()
                        timeout = self._PROBE_INTERVAL_S if wait_s is None else min(self._PROBE_INTERVAL_S, wait_s)
                        self._condition.wait(timeout=timeout)
                continue

            with self._condition:
                for unit in ready:
                    if unit not in self._running:
                        continue
                    self._running.remove(unit)
                    self._running_tokens[unit.launcher] = max(0, self._running_tokens[unit.launcher] - unit.est_tokens)
                    if unit.root_id is not None:
                        self._leader_completed.add(unit.root_id)
                        if self._kv_replication != "none":
                            self._replicated_roots.setdefault(unit.root_id, unit.launcher)
                    self._completed.append(unit)
                self._condition.notify_all()

    def _record_dispatch(self, launcher_index: int, task: "Sample", root_id: Optional[str], now: float) -> None:
        self._dispatched_tasks += 1
        if root_id is not None:
            self._root_launchers.setdefault(root_id, set()).add(launcher_index)
            self._root_task_counts[root_id] = self._root_task_counts.get(root_id, 0) + 1
            if root_id not in self._leader_dispatched_at:
                self._leader_dispatched_at[root_id] = now
                self._leader_primary_worker[root_id] = launcher_index

    def _poll_kv_usages(self) -> Optional[List[float]]:
        """Poll optional per-worker KV occupancy in [0.0, 1.0]; return None if unavailable."""
        if self._kv_usage_fn is None:
            return None
        try:
            vals = [max(0.0, min(1.0, float(v))) for v in self._kv_usage_fn()]
        except Exception:
            return None
        return vals if len(vals) == len(self._launchers) else None

    def _is_replicated(self, root_id: Optional[str], now: float) -> bool:
        if self._kv_replication == "none" or root_id is None:
            return False
        if root_id in self._replicated_roots:
            return True
        if self._stagger_leader_prefill and root_id in self._leader_dispatched_at:
            if (now - self._leader_dispatched_at[root_id]) >= self._leader_stagger_s:
                primary = self._leader_primary_worker.get(root_id)
                if primary is not None:
                    self._replicated_roots[root_id] = primary
                    return True
        return False

    def _is_stagger_held(self, root_id: Optional[str], now: float) -> bool:
        if not self._stagger_leader_prefill or root_id is None or root_id in self._leader_completed:
            return False
        dispatched_at = self._leader_dispatched_at.get(root_id)
        if dispatched_at is None:
            return False
        return (now - dispatched_at) < self._leader_stagger_s

    def _select_scored_worker(
        self,
        *,
        root_id: Optional[str],
        est_tokens: int,
        load: List[int],
        tok_load: List[int],
        kv_usages: Optional[List[float]],
        now: float,
    ) -> Optional[int]:
        eligible = [i for i in range(len(self._launchers)) if load[i] < self._capacities[i]]
        if not eligible:
            return None
        if self._max_inflight_tokens is not None:
            within_budget = [i for i in eligible if tok_load[i] + est_tokens <= self._max_inflight_tokens]
            if within_budget:
                eligible = within_budget
        min_load = min(load[i] for i in eligible)
        if self._balance_rel_threshold is not None:
            balanced = [
                i
                for i in eligible
                if (load[i] - min_load < self._balance_abs_threshold)
                or (load[i] <= self._balance_rel_threshold * max(1, min_load))
            ]
            if balanced:
                eligible = balanced

        known_workers = self._root_launchers.get(root_id, set()) if root_id is not None else set()
        known_nodes = (
            {self._worker_node_ids[w] for w in known_workers}
            if (self._worker_node_ids is not None and known_workers)
            else set()
        )
        replicated = self._is_replicated(root_id, now)

        # Keep siblings on their home worker while it remains within balance, token, and KV bounds.
        home_eligible = [
            i for i in eligible if i in known_workers and (kv_usages is None or kv_usages[i] < 0.85)
        ]
        if home_eligible:
            eligible = home_eligible

        def score(i: int) -> tuple[float, int, int]:
            if i in known_workers:
                affinity = 1.0
            elif self._worker_node_ids is not None and self._worker_node_ids[i] in known_nodes:
                affinity = 0.35
            else:
                affinity = 0.0
            load_ratio = kv_usages[i] if kv_usages is not None else (load[i] / self._capacities[i])
            tok_ratio = (tok_load[i] / self._max_inflight_tokens) if self._max_inflight_tokens else 0.0
            val = (
                self._prefix_affinity_weight * affinity
                - self._load_weight * load_ratio
                - self._active_tokens_weight * tok_ratio
            )
            return (val, -load[i], -i)

        chosen = max(eligible, key=score)
        if replicated and root_id is not None and chosen not in known_workers:
            self._kv_replication_hits += 1
            if self._enable_work_stealing and any(load[w] > load[chosen] for w in known_workers):
                self._work_steals += 1
            if self._kv_replicator is not None:
                src = self._replicated_roots.get(root_id)
                if src is not None:
                    self._kv_replicator(src, chosen, root_id)
        return chosen

    def _plan_launches(self) -> List[tuple[int, int, "Sample", Optional[str], int]]:
        """Reserve launcher slots for queued tasks; caller holds the lock."""
        if self._paused or self._closed:
            return []
        load = list(self._reserved)
        tok_load = list(self._reserved_tokens)
        for unit in self._running:
            load[unit.launcher] += 1
        for i, rt in enumerate(self._running_tokens):
            tok_load[i] += rt
        plan: List[tuple[int, int, "Sample", Optional[str], int]] = []
        now = time.monotonic()

        if self._affinity_policy == "cache_aware_scored":
            kv_usages = self._poll_kv_usages()
            deferred: Deque[tuple[int, "Sample"]] = deque()
            while self._queue:
                if all(load[i] >= self._capacities[i] for i in range(len(self._launchers))):
                    break
                sequence, task = self._queue.popleft()
                root_id = _single_root_id(task)
                if self._is_stagger_held(root_id, now):
                    self._stagger_holds += 1
                    deferred.append((sequence, task))
                    continue
                est_tokens = _estimate_task_tokens(task)
                index = self._select_scored_worker(
                    root_id=root_id,
                    est_tokens=est_tokens,
                    load=load,
                    tok_load=tok_load,
                    kv_usages=kv_usages,
                    now=now,
                )
                if index is None:
                    deferred.append((sequence, task))
                    break
                plan.append((sequence, index, task, root_id, est_tokens))
                self._reserved[index] += 1
                self._reserved_tokens[index] += est_tokens
                load[index] += 1
                tok_load[index] += est_tokens
                self._record_dispatch(index, task, root_id, now)
            if deferred:
                self._queue.extendleft(reversed(deferred))
            return plan

        # least_loaded / group_packing
        deferred_gp: Deque[tuple[int, "Sample"]] = deque()
        while self._queue:
            index = max(range(len(self._launchers)), key=lambda i: self._capacities[i] - load[i])
            avail = self._capacities[index] - load[index]
            if avail <= 0:
                break
            sequence, task = self._queue.popleft()
            root_id = _single_root_id(task) if self._affinity_policy == "group_packing" else None
            if self._affinity_policy == "group_packing" and self._is_stagger_held(root_id, now):
                self._stagger_holds += 1
                deferred_gp.append((sequence, task))
                continue
            est_tokens = _estimate_task_tokens(task)
            plan.append((sequence, index, task, root_id, est_tokens))
            self._reserved[index] += 1
            self._reserved_tokens[index] += est_tokens
            load[index] += 1
            tok_load[index] += est_tokens
            avail -= 1
            self._record_dispatch(index, task, root_id, now)
            while (
                avail > 0
                and self._queue
                and root_id is not None
                and _single_root_id(self._queue[0][1]) == root_id
                and not self._is_stagger_held(root_id, now)
            ):
                if self._balance_rel_threshold is not None:
                    min_load = min(load)
                    if (
                        load[index] - min_load >= self._balance_abs_threshold
                        and load[index] > self._balance_rel_threshold * max(1, min_load)
                    ):
                        break
                sequence, task = self._queue.popleft()
                est_tokens = _estimate_task_tokens(task)
                plan.append((sequence, index, task, root_id, est_tokens))
                self._reserved[index] += 1
                self._reserved_tokens[index] += est_tokens
                load[index] += 1
                tok_load[index] += est_tokens
                avail -= 1
                self._record_dispatch(index, task, root_id, now)
        if deferred_gp:
            self._queue.extendleft(reversed(deferred_gp))
        return plan

    def _execute_launches(self, plan: List[tuple[int, int, "Sample", Optional[str], int]]) -> bool:
        """Run reserved launches outside the lock; False when a launch failed (pool poisoned)."""
        launched: List[_PendingUnit] = []
        failure: Optional[BaseException] = None
        for sequence, index, task, root_id, est_tokens in plan:
            try:
                pending = self._launchers[index](task)
            except BaseException as exc:
                failure = exc
                break
            launched.append(_PendingUnit(sequence, index, task, pending, root_id=root_id, est_tokens=est_tokens))

        with self._condition:
            for unit in launched:
                self._reserved[unit.launcher] -= 1
                self._reserved_tokens[unit.launcher] = max(0, self._reserved_tokens[unit.launcher] - unit.est_tokens)
                self._running.append(unit)
                self._running_tokens[unit.launcher] += unit.est_tokens
            if failure is not None:
                remaining = plan[len(launched) :]
                for _, index, _, _, est_tokens in remaining:
                    self._reserved[index] -= 1
                    self._reserved_tokens[index] = max(0, self._reserved_tokens[index] - est_tokens)
                self._queue.extendleft((sequence, task) for sequence, _, task, _, _ in reversed(remaining))
                if self._failure is None:
                    self._failure = failure
                self._paused = True
            self._condition.notify_all()
        return failure is None

    def _record_failure(self, exc: BaseException) -> None:
        with self._condition:
            if self._failure is None:
                self._failure = exc
            self._paused = True
            self._condition.notify_all()


def required_worker_concurrency(per_worker_inflight: int) -> int:
    """Return actor concurrency that leaves room for rollout control calls."""
    return per_worker_inflight + _WORKER_CONTROL_CONCURRENCY


def validate_worker_inflight(
    per_worker_inflight: int,
    *,
    worker_max_concurrency: int,
    engine_concurrency: Optional[int],
) -> None:
    """Validate per-worker rollout concurrency against actor and engine capacity."""
    if per_worker_inflight <= 0:
        raise ValueError(f"per_worker_inflight must be positive; got {per_worker_inflight}")

    # Reserve control-call threads for set_stopping, sleep, and weight sync
    # while rollout calls occupy their own slots.
    required_concurrency = required_worker_concurrency(per_worker_inflight)
    if worker_max_concurrency < required_concurrency:
        raise ValueError(
            f"worker_max_concurrency ({worker_max_concurrency}) must be >= per_worker_inflight "
            f"+ {_WORKER_CONTROL_CONCURRENCY} "
            f"({required_concurrency}) so rollout calls cannot starve control calls"
        )

    if engine_concurrency is not None and per_worker_inflight > engine_concurrency:
        raise ValueError(
            f"per_worker_inflight ({per_worker_inflight}) exceeds engine concurrency ({engine_concurrency})"
        )


__all__ = [
    "AFFINITY_POLICIES",
    "KV_REPLICATION_MODES",
    "RolloutPool",
    "required_worker_concurrency",
    "validate_worker_inflight",
]
