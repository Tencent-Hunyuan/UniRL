"""Unified-backbone multi-algorithm train stack (HunyuanImage3).

Wraps ONE :class:`FSDPBackend` (a single shared transformer + optimizer +
scheduler + EMA) and TWO :class:`StageAlgorithm` siblings — an ``ar`` algorithm
over the ``TextSegment`` and an ``image`` algorithm over the ``LatentSegment`` —
into a single training driver.  Both algorithms run forward/backward against the
*same* shared backbone (HunyuanImage3 operates in ``mode="gen_text"`` for AR and
``mode="gen_image"`` for DiT on one set of weights), so their gradients
accumulate into one LoRA adapter and a single optimizer step applies both.

Mirrors :class:`unirl.train.stack.TrainStack` but for the unified-backbone
two-algorithm case.  Sequencing per :meth:`train` call::

    prepare_segment(ar); prepare_segment(image)              # once: freeze both π_old anchors
    for u in range(num_updates_per_batch):                   # PPO-style mini-batches
        backend.zero_grad()
        backend.park_optimizer_state_for_rollout()           # optional after u0: Adam moments -> CPU
        for name in ("ar", "image"):
            for (start, end) in micro_slices(mini_batch_u):
                algorithm[name].compute_loss_and_backward(loss_scale=1/N, ...)  # grads accumulate
        backend.restore_optimizer_state_after_rollout()      # exact slot devices
        backend.optimizer_step(max_grad_norm=...)            # ONE step per mini-batch
    on_rollout_end()
    return {name: TrainStepResult, ...}                      # reduced across updates

``num_updates_per_batch`` (default 1) splits each rollout shard into that many
disjoint mini-batches and runs one optimizer step per mini-batch, with each track's
π_old anchor frozen once across all of them — so the 2nd+ step is off-policy and the
clip / ratio trust region actually engages (the UniGRPO / FlowGRPO PPO schedule).
Mirrors :class:`~unirl.train.stack.TrainStack` but for the two-algorithm backbone.

This is the multi-stage train stack — several stage algorithms share one
optimizer step, in contrast to the single-stage ``TrainStack``.
"""

from __future__ import annotations

import logging
import math
import sys
import time
from contextlib import nullcontext
from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch

from unirl.algorithms import AlgorithmStepResult, StageAlgorithm
from unirl.distributed.group.dispatch import Dispatch, _collect_dp_merge, distributed
from unirl.distributed.group.remote import Remote
from unirl.train.backend.fsdp import FSDPBackend
from unirl.train.stack import TrainStepResult, _build_micro_batch_slices
from unirl.train.stack.base import _aggregate_update_results
from unirl.train.stack.planner.types import _positive_int, _update_ranges
from unirl.types.rollout_resp import RolloutTrack
from unirl.utils.misc import aggregate_numeric_metrics

logger = logging.getLogger(__name__)

_PHASE_HOST_TIME_METRICS = frozenset(
    {
        "anchor_image_host_time_s",
        "ar_backward_host_time_s",
        "image_prepare_reference_host_time_s",
        "image_ratio_mse_backward_host_time_s",
        "image_micro_empty_cache_host_time_s",
        "pre_optimizer_empty_cache_host_time_s",
        "train_optimizer_park_host_time_s",
        "train_optimizer_restore_host_time_s",
        "optimizer_host_time_s",
        "post_optimizer_empty_cache_host_time_s",
    }
)
_CUDA_PEAK_METRICS = frozenset(
    {
        "cuda_peak_allocated_gb",
        "cuda_peak_reserved_gb",
    }
)
_CUDA_TRAIN_WINDOW_PEAK_METRICS = frozenset(
    {
        "cuda_train_window_peak_allocated_gb",
        "cuda_train_window_peak_reserved_gb",
    }
)
_CUDA_PHASE_MEMORY_METRICS = frozenset(
    {
        f"cuda_{phase}_{kind}_gb"
        for phase in ("pre_optimizer_restore", "post_optimizer_restore", "post_optimizer_step")
        for kind in ("allocated", "reserved")
    }
)
_IMAGE_MICRO_RECLAIM_MAX_COUNT_METRICS = frozenset(
    {
        "image_micro_empty_cache_call_count",
        "image_micro_empty_cache_pressure_call_count",
        "image_micro_empty_cache_pressure_check_count",
    }
)
_IMAGE_MICRO_RECLAIM_MIN_COUNT_METRICS = frozenset({"image_micro_empty_cache_skip_count"})
_IMAGE_MICRO_RECLAIM_COUNT_METRICS = _IMAGE_MICRO_RECLAIM_MAX_COUNT_METRICS | _IMAGE_MICRO_RECLAIM_MIN_COUNT_METRICS
_IMAGE_MICRO_RECLAIM_HEADROOM_METRICS = frozenset({"image_micro_empty_cache_min_free_gb"})
_TRAIN_OPTIMIZER_STATE_SUM_METRICS = frozenset(
    {
        "train_optimizer_state_bytes",
        "train_optimizer_state_bytes_parked",
        "train_optimizer_state_bytes_restored",
    }
)
_TRAIN_OPTIMIZER_STATE_MAX_METRICS = frozenset({"train_optimizer_state_restore_slots_pending"})
_DP_MAX_METRICS = (
    _PHASE_HOST_TIME_METRICS
    | _CUDA_PEAK_METRICS
    | _CUDA_TRAIN_WINDOW_PEAK_METRICS
    | _CUDA_PHASE_MEMORY_METRICS
    | _IMAGE_MICRO_RECLAIM_MAX_COUNT_METRICS
)
_DP_MIN_METRICS = _IMAGE_MICRO_RECLAIM_MIN_COUNT_METRICS | _IMAGE_MICRO_RECLAIM_HEADROOM_METRICS
_DP_SUM_METRICS = _TRAIN_OPTIMIZER_STATE_SUM_METRICS


def _max_dp_metrics(base: Mapping[str, object], peers: List[Mapping[str, object]]) -> Dict[str, object]:
    """Copy ``base`` while reducing BAGEL safety/critical-path diagnostics over DP peers.

    Timings, allocator peaks, pending-slot counts, and reclaim counts use the
    worst-case maximum; free-memory headroom uses the worst-case minimum;
    optimizer-state bytes sum the physical shards owned by DP peers.
    """
    reduced = dict(base)
    for metric_name in _DP_MAX_METRICS | _TRAIN_OPTIMIZER_STATE_MAX_METRICS:
        values = [float(metrics[metric_name]) for metrics in peers if metric_name in metrics]
        if values:
            reduced[metric_name] = max(values)
    for metric_name in _DP_MIN_METRICS:
        values = [float(metrics[metric_name]) for metrics in peers if metric_name in metrics]
        if values:
            reduced[metric_name] = min(values)
    for metric_name in _DP_SUM_METRICS:
        values = [float(metrics[metric_name]) for metrics in peers if metric_name in metrics]
        if values:
            reduced[metric_name] = sum(values)
    return reduced


def _train_optimizer_metrics(report: Mapping[str, object]) -> Dict[str, float]:
    """Namespace the existing backend park/restore report for train metrics."""
    names = {
        "optimizer_state_bytes": "train_optimizer_state_bytes",
        "optimizer_state_bytes_parked": "train_optimizer_state_bytes_parked",
        "optimizer_park_host_time_s": "train_optimizer_park_host_time_s",
        "optimizer_state_bytes_restored": "train_optimizer_state_bytes_restored",
        "optimizer_state_restore_slots_pending": "train_optimizer_state_restore_slots_pending",
        "optimizer_restore_host_time_s": "train_optimizer_restore_host_time_s",
    }
    return {names[key]: float(value) for key, value in report.items() if key in names}


def _collect_unified_train_results(wg: Any, results: List[Any]) -> Any:
    """Use DP critical-path maxima for BAGEL timings and CUDA peaks.

    The standard DP collector intentionally selects scalar fields from the first
    DP result. Keep that behavior for losses and algorithm metrics, but reduce the
        diagnostic host timers plus per-update and whole-train allocator peaks
        across DP heads after every worker RPC has returned. This is controller-only
        work: it adds neither a training collective nor a CUDA synchronization.
    """
    collected = _collect_dp_merge(wg, results)
    if collected is None or not isinstance(collected, dict):
        return collected

    dp_results = [
        results[i]
        for i in range(len(results))
        if wg.rank_infos[i].tp_rank == 0
        and wg.rank_infos[i].is_pipeline_last_stage
        and wg.rank_infos[i].sp_rank == 0
        and isinstance(results[i], dict)
    ]
    if len(dp_results) <= 1:
        return collected

    reduced: Dict[str, TrainStepResult] = {}
    for track_name, base_result in collected.items():
        peer_results = [result[track_name] for result in dp_results if track_name in result]
        if not peer_results or not isinstance(base_result, TrainStepResult):
            reduced[track_name] = base_result
            continue

        per_update = tuple(base_result.per_update)
        if per_update:
            reduced_updates = []
            for update_index, base_metrics in enumerate(per_update):
                peer_metrics = [
                    peer.per_update[update_index] for peer in peer_results if update_index < len(peer.per_update)
                ]
                reduced_updates.append(_max_dp_metrics(base_metrics, peer_metrics))

            summary_metrics = dict(base_result.metrics)
            for metric_name in (
                _PHASE_HOST_TIME_METRICS | _IMAGE_MICRO_RECLAIM_COUNT_METRICS | _TRAIN_OPTIMIZER_STATE_SUM_METRICS
            ):
                values = [float(metrics[metric_name]) for metrics in reduced_updates if metric_name in metrics]
                if values:
                    summary_metrics[metric_name] = sum(values) / len(values)
            for metric_name in _CUDA_PEAK_METRICS | _CUDA_PHASE_MEMORY_METRICS | _TRAIN_OPTIMIZER_STATE_MAX_METRICS:
                values = [float(metrics[metric_name]) for metrics in reduced_updates if metric_name in metrics]
                if values:
                    summary_metrics[metric_name] = max(values)
            for metric_name in _CUDA_TRAIN_WINDOW_PEAK_METRICS:
                values = [float(peer.metrics[metric_name]) for peer in peer_results if metric_name in peer.metrics]
                if values:
                    summary_metrics[metric_name] = max(values)
            for metric_name in _IMAGE_MICRO_RECLAIM_HEADROOM_METRICS:
                values = [float(metrics[metric_name]) for metrics in reduced_updates if metric_name in metrics]
                if values:
                    summary_metrics[metric_name] = min(values)
            reduced[track_name] = replace(
                base_result,
                metrics=summary_metrics,
                per_update=tuple(reduced_updates),
            )
        else:
            reduced[track_name] = replace(
                base_result,
                metrics=_max_dp_metrics(base_result.metrics, [peer.metrics for peer in peer_results]),
            )
    return reduced


def _profile_record(profiler: Any, name: str):
    """Return a named profiler scope when whole-train profiling is active."""
    return profiler.record(name) if profiler is not None else nullcontext()


class UnifiedModelTrainStack(Remote):
    """Single-backbone, multi-algorithm train stack.

    Holds one shared :class:`FSDPBackend` and a dict of named
    :class:`StageAlgorithm` siblings (``{"ar": GRPO, "image": FlowGRPO}``).
    Each algorithm trains its own track but backward-accumulates into the same
    shared transformer; one optimizer step applies all algorithms' gradients.

    Created as a sibling ``Remote`` inside a placement block; takes handles to
    its ``FSDPBackend`` and ``StageAlgorithm`` siblings via sibling-handle
    auto-resolve (same pattern as :class:`TrainStack`).
    """

    def __init__(
        self,
        *,
        fsdp_backend: FSDPBackend,
        ar_algorithm: StageAlgorithm,
        image_algorithm: StageAlgorithm,
        micro_batch_size: int,
        max_grad_norm: float,
        num_updates_per_batch: int = 1,
        park_optimizer_state_during_train: bool = False,
        empty_cache_after_image_micro: bool = False,
        image_micro_empty_cache_interval: int = 1,
        image_micro_empty_cache_min_free_gb: float = 0.0,
        empty_cache_after_optimizer: bool = False,
        cuda_peak_telemetry: bool = False,
    ) -> None:
        super().__init__()
        if int(micro_batch_size) < 1:
            raise ValueError(f"UnifiedModelTrainStack.micro_batch_size must be >= 1; got {micro_batch_size}.")
        if float(max_grad_norm) <= 0.0:
            raise ValueError(f"UnifiedModelTrainStack.max_grad_norm must be > 0; got {max_grad_norm}.")
        self.fsdp_backend = fsdp_backend
        # Order matters only for logging; gradients accumulate regardless.
        self.algorithms: Dict[str, StageAlgorithm] = {
            "ar": ar_algorithm,
            "image": image_algorithm,
        }
        self.micro_batch_size = int(micro_batch_size)
        self.max_grad_norm = float(max_grad_norm)
        self.park_optimizer_state_during_train = bool(park_optimizer_state_during_train)
        if self.park_optimizer_state_during_train and bool(getattr(fsdp_backend, "_persistent_cpu_offload", False)):
            raise ValueError(
                "UnifiedModelTrainStack.park_optimizer_state_during_train requires backend.fsdp_cfg.cpu_offload=false."
            )
        self.empty_cache_after_image_micro = bool(empty_cache_after_image_micro)
        self.image_micro_empty_cache_interval = _positive_int(
            name="UnifiedModelTrainStack.image_micro_empty_cache_interval",
            value=image_micro_empty_cache_interval,
        )
        self.image_micro_empty_cache_min_free_gb = float(image_micro_empty_cache_min_free_gb)
        if not math.isfinite(self.image_micro_empty_cache_min_free_gb):
            raise ValueError(
                "UnifiedModelTrainStack.image_micro_empty_cache_min_free_gb must be finite; "
                f"got {image_micro_empty_cache_min_free_gb}."
            )
        if self.image_micro_empty_cache_min_free_gb < 0.0:
            raise ValueError(
                "UnifiedModelTrainStack.image_micro_empty_cache_min_free_gb must be >= 0; "
                f"got {image_micro_empty_cache_min_free_gb}."
            )
        self.empty_cache_after_optimizer = bool(empty_cache_after_optimizer)
        self.cuda_peak_telemetry = bool(cuda_peak_telemetry)
        # PPO-style multi-update: split each rollout shard into this many disjoint
        # mini-batches and run ONE optimizer step per mini-batch, with the π_old
        # anchor frozen once across all of them (prepare_segment). >1 makes the
        # clip / ratio trust region actually engage (the 2nd+ step is off-policy);
        # 1 (default) keeps the prior single-step behavior. BOTH algorithms must
        # keep their anchor frozen across the N steps (supports_multi_update).
        self.num_updates_per_batch = _positive_int(
            name="UnifiedModelTrainStack.num_updates_per_batch", value=num_updates_per_batch
        )
        if self.num_updates_per_batch > 1:
            for name, algo in self.algorithms.items():
                if not getattr(algo, "supports_multi_update", False):
                    raise ValueError(
                        f"num_updates_per_batch={self.num_updates_per_batch} requires every algorithm's "
                        f"π_old anchor to stay frozen across the N optimizer steps, but the {name!r} "
                        f"algorithm ({type(algo).__name__}) sets supports_multi_update=False. Set "
                        f"num_updates_per_batch=1."
                    )

    def _reset_cuda_peak_memory_stats(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def _cuda_phase_memory_metrics(self, phase: str) -> Dict[str, float]:
        if not torch.cuda.is_available():
            return {}
        return {
            f"cuda_{phase}_allocated_gb": torch.cuda.memory_allocated() / 2**30,
            f"cuda_{phase}_reserved_gb": torch.cuda.memory_reserved() / 2**30,
        }

    def _cuda_peak_memory_metrics(self) -> Dict[str, float]:
        if not torch.cuda.is_available():
            return {}
        return {
            "cuda_peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
            "cuda_peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
        }

    def _cuda_train_window_peak_memory_metrics(self) -> Dict[str, float]:
        if not torch.cuda.is_available():
            return {}
        return {
            "cuda_train_window_peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
            "cuda_train_window_peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
        }

    def _optimizer_step_slices(self, total: int) -> List[List[Tuple[int, int]]]:
        """Per-optimizer-step lists of absolute ``(start, end)`` micro-batch slices.

        One inner list per ``num_updates_per_batch`` mini-batch (one optimizer step),
        each split into ``micro_batch_size`` micro-batches. Shared by
        :meth:`prepare_segment` (to freeze the anchor at the exact geometry) and the
        train loop. Mirrors :meth:`unirl.train.stack.TrainStack._optimizer_step_slices`.
        """
        steps: List[List[Tuple[int, int]]] = []
        for mini_start, mini_end in _update_ranges(total_size=total, num_updates=self.num_updates_per_batch):
            steps.append(
                [
                    (mini_start + ms, mini_start + me)
                    for ms, me in _build_micro_batch_slices(
                        total_size=mini_end - mini_start, micro_batch_size=self.micro_batch_size
                    )
                ]
            )
        return steps

    def prepare_segment(self, name: str, resp_track: RolloutTrack) -> None:
        """Freeze one algorithm's π_old anchor once, before the multi-update loop.

        No-op if ``segment`` is None or the algorithm has no ``prepare_segment``. If
        the algorithm recomputes its anchor at train geometry (``recomputes_anchor()``
        — e.g. FlowGRPO under ``old_logp_source='replay'``), the declared
        ``anchor_fields`` are recomputed at the SAME (mini, micro) slices training will
        use, so the on-policy ratio is exactly 1 (mirrors
        :meth:`TrainStack.prepare_segment`). Rollout-anchored algorithms (the BAGEL
        UniGRPO recipe: AR GRPO + image ``old_logp_source='rollout'``) take the
        one-shot path — the anchor is the rollout emission, geometry-independent.
        """
        if resp_track.segment is None:
            return
        algorithm = self.algorithms[name]
        prepare = getattr(algorithm, "prepare_segment", None)
        if prepare is None:
            return
        recomputes = getattr(algorithm, "recomputes_anchor", None)
        if recomputes is None or not recomputes():
            prepare(conditions=resp_track.conditions, segment=resp_track.segment)
            return
        micro_slices = [sl for step in self._optimizer_step_slices(int(resp_track.batch_size)) for sl in step]
        if len(micro_slices) == 1:
            prepare(conditions=resp_track.conditions, segment=resp_track.segment)
            return
        anchor_fields = getattr(algorithm, "anchor_fields", ())
        collected: Dict[str, List[torch.Tensor]] = {field: [] for field in anchor_fields}
        for start, end in micro_slices:
            micro = resp_track.slice(start, end)
            prepare(conditions=micro.conditions, segment=micro.segment)
            for field in collected:
                value = getattr(micro.segment, field, None)
                if value is None:
                    raise RuntimeError(
                        f"UnifiedModelTrainStack.prepare_segment: {type(algorithm).__name__} declares "
                        f"anchor field {field!r} but a micro-slice produced None."
                    )
                collected[field].append(value)
        for field, parts in collected.items():
            setattr(resp_track.segment, field, torch.cat(parts, dim=0))

    def _prepare_anchor_batch(
        self,
        name: str,
        track: RolloutTrack,
        update_slices: List[List[Tuple[int, int]]],
    ) -> None:
        """Supply one algorithm the complete pre-optimizer update partition."""
        algorithm = self.algorithms[name]
        updates = []
        for micro_slices in update_slices:
            update = []
            for start, end in micro_slices:
                micro = track.slice(start, end)
                if micro.segment is None:
                    raise RuntimeError(
                        f"UnifiedModelTrainStack._prepare_anchor_batch: track {name!r} "
                        "produced a micro-batch with segment=None."
                    )
                update.append((micro.conditions, micro.segment))
            updates.append(update)
        algorithm.prepare_anchor_batch(updates=updates)

    def _backward_track(
        self,
        name: str,
        resp_track: RolloutTrack,
        micro_slices: List[Tuple[int, int]],
        *,
        training_progress: float,
    ) -> tuple[TrainStepResult, bool]:
        """Backward one algorithm's track over the given absolute ``micro_slices``
        (no zero_grad / no optimizer step).

        Returns ``(per_algorithm_result, has_backward)``. ``zero_grad`` and the shared
        ``optimizer_step`` are owned by :meth:`_train_one_step` so both algorithms
        accumulate into one step. ``micro_slices`` are absolute ranges into
        ``resp_track`` for ONE optimizer step (one ``num_updates_per_batch``
        mini-batch), produced by :meth:`_optimizer_step_slices`.
        """
        if resp_track.advantages is None:
            raise ValueError(
                f"UnifiedModelTrainStack.train: track {name!r} has advantages=None; "
                "upstream advantage pipeline must populate it before training."
            )
        if not micro_slices:
            raise ValueError(f"UnifiedModelTrainStack.train: empty micro_slices for track {name!r}.")

        bs = int(resp_track.batch_size)
        algorithm = self.algorithms[name]
        loss_scale = 1.0 / len(micro_slices)
        micros: List[AlgorithmStepResult] = []
        total_loss = 0.0
        has_backward = False
        image_micro_empty_cache_host_time_s = 0.0
        image_micro_empty_cache_call_count = 0
        image_micro_empty_cache_skip_count = 0
        image_micro_empty_cache_pressure_call_count = 0
        image_micro_empty_cache_pressure_check_count = 0
        image_micro_empty_cache_min_free_gb: Optional[float] = None

        single_micro = len(micro_slices) == 1 and micro_slices[0] == (0, bs)
        image_reclamation_enabled = name == "image" and self.empty_cache_after_image_micro and torch.cuda.is_available()
        reclaim_interval = int(getattr(self, "image_micro_empty_cache_interval", 1))
        min_free_gb = float(getattr(self, "image_micro_empty_cache_min_free_gb", 0.0))
        for micro_index, (start, end) in enumerate(micro_slices):
            micro_track = resp_track if single_micro else resp_track.slice(start, end)
            result = algorithm.compute_loss_and_backward(
                conditions=micro_track.conditions,
                segment=micro_track.segment,
                advantages=micro_track.advantages,
                training_progress=training_progress,
                loss_scale=loss_scale,
            )
            if image_reclamation_enabled:
                ordinal = micro_index + 1
                cadence_due = ordinal % reclaim_interval == 0
                final_micro = ordinal == len(micro_slices)
                pressure_due = False
                # Query the driver only when cadence would otherwise skip this
                # boundary. A configured floor is an emergency trigger, never a
                # reason to defer the bounded-cadence or final reclamation.
                if not cadence_due and not final_micro and min_free_gb > 0.0:
                    free_bytes, _ = torch.cuda.mem_get_info()
                    free_gb = float(free_bytes) / 2**30
                    image_micro_empty_cache_pressure_check_count += 1
                    image_micro_empty_cache_min_free_gb = (
                        free_gb
                        if image_micro_empty_cache_min_free_gb is None
                        else min(image_micro_empty_cache_min_free_gb, free_gb)
                    )
                    pressure_due = free_gb < min_free_gb

                if cadence_due or final_micro or pressure_due:
                    empty_cache_started = time.perf_counter()
                    torch.cuda.empty_cache()
                    image_micro_empty_cache_host_time_s += time.perf_counter() - empty_cache_started
                    image_micro_empty_cache_call_count += 1
                    image_micro_empty_cache_pressure_call_count += int(pressure_due)
                else:
                    image_micro_empty_cache_skip_count += 1
            micros.append(result)
            total_loss += result.loss
            has_backward = has_backward or result.has_backward

        aggregated: Mapping[str, object] = aggregate_numeric_metrics([r.metrics for r in micros if r.metrics])
        if name == "image" and self.empty_cache_after_image_micro:
            aggregated = {
                **dict(aggregated),
                "image_micro_empty_cache_host_time_s": image_micro_empty_cache_host_time_s,
                "image_micro_empty_cache_call_count": float(image_micro_empty_cache_call_count),
                "image_micro_empty_cache_skip_count": float(image_micro_empty_cache_skip_count),
                "image_micro_empty_cache_pressure_call_count": float(image_micro_empty_cache_pressure_call_count),
                "image_micro_empty_cache_pressure_check_count": float(image_micro_empty_cache_pressure_check_count),
            }
            if image_micro_empty_cache_min_free_gb is not None:
                aggregated = {
                    **dict(aggregated),
                    "image_micro_empty_cache_min_free_gb": image_micro_empty_cache_min_free_gb,
                }
        # grad_norm / lr are filled by ``_train_one_step`` after the shared optimizer step.
        partial = TrainStepResult(
            loss=total_loss,
            grad_norm=0.0,
            lr=0.0,
            has_backward=has_backward,
            micros=micros,
            metrics=aggregated,
        )
        return partial, has_backward

    def _train_one_step(
        self,
        tracks: Dict[str, RolloutTrack],
        slices_by_track: Dict[str, List[Tuple[int, int]]],
        *,
        training_progress: float,
        update_index: int = 0,
        profiler: Any = None,
        anchor_image_host_time_s: Optional[float] = None,
        optimizer_state_already_parked: bool = False,
    ) -> Dict[str, TrainStepResult]:
        """One optimizer step: zero_grad → backward BOTH tracks over their mini-batch
        slices → shared optimizer_step → stamp grad_norm / lr onto each track's result.
        """
        park_optimizer = bool(getattr(self, "park_optimizer_state_during_train", False))
        if optimizer_state_already_parked and not park_optimizer:
            raise ValueError("optimizer_state_already_parked requires park_optimizer_state_during_train=true.")
        cuda_peak_telemetry = bool(getattr(self, "cuda_peak_telemetry", False))
        if cuda_peak_telemetry:
            self._reset_cuda_peak_memory_stats()
        phase_times: Dict[str, float] = {}
        train_optimizer_metrics: Dict[str, float] = {}
        optimizer_park_attempted = bool(optimizer_state_already_parked)
        optimizer_restored = False
        finished_algorithms: set[int] = set()

        def finish_prepared_state(*, succeeded: bool) -> List[Tuple[str, BaseException]]:
            errors: List[Tuple[str, BaseException]] = []
            for algorithm in self.algorithms.values():
                if not algorithm.prepares_update_batch or id(algorithm) in finished_algorithms:
                    continue
                # finish_update_batch must release transient state even if its
                # completeness validation raises, so never call it twice.
                finished_algorithms.add(id(algorithm))
                try:
                    algorithm.finish_update_batch(succeeded=succeeded)
                except BaseException as exc:
                    errors.append((f"{type(algorithm).__name__}.finish_update_batch", exc))
            return errors

        def reclaim_allocator() -> None:
            started = time.perf_counter()
            self.fsdp_backend.reclaim_cuda_allocator()
            phase_times["pre_optimizer_empty_cache_host_time_s"] = phase_times.get(
                "pre_optimizer_empty_cache_host_time_s", 0.0
            ) + (time.perf_counter() - started)

        def restore_optimizer_state() -> None:
            nonlocal optimizer_restored
            if not optimizer_park_attempted or optimizer_restored:
                return
            reclaim_allocator()
            if cuda_peak_telemetry:
                train_optimizer_metrics.update(self._cuda_phase_memory_metrics("pre_optimizer_restore"))
            report = self.fsdp_backend.restore_optimizer_state_after_rollout()
            train_optimizer_metrics.update(_train_optimizer_metrics(report))
            pending = float(report.get("optimizer_state_restore_slots_pending", 0.0))
            if pending:
                raise RuntimeError(f"optimizer restore left {int(pending)} state slot(s) pending.")
            optimizer_restored = True
            if cuda_peak_telemetry:
                train_optimizer_metrics.update(self._cuda_phase_memory_metrics("post_optimizer_restore"))

        succeeded = False
        try:
            self.fsdp_backend.zero_grad()
            if park_optimizer and not optimizer_state_already_parked:
                # Reuse the rollout boundary's exact per-slot device plan. This
                # second zero_grad is intentional: the backend primitive records
                # before transfer and repairs a partially failed park in cleanup.
                optimizer_park_attempted = True
                report = self.fsdp_backend.park_optimizer_state_for_rollout()
                train_optimizer_metrics.update(_train_optimizer_metrics(report))

            results: Dict[str, TrainStepResult] = {}
            any_backward = False
            for name, algorithm in self.algorithms.items():
                # Prepare immediately before this algorithm's forward/backward.
                # BAGEL image reference KVs therefore do not occupy GPU memory
                # during the preceding AR backward, and every reference swap is
                # complete before its own activation-checkpointed graph exists.
                if algorithm.prepares_update_batch:
                    started = time.perf_counter()
                    with _profile_record(profiler, f"update_{update_index}/{name}_prepare_reference"):
                        self._prepare_update_batch(
                            name,
                            tracks[name],
                            slices_by_track[name],
                            training_progress=training_progress,
                            update_index=update_index,
                        )
                    if name == "image":
                        phase_times["image_prepare_reference_host_time_s"] = time.perf_counter() - started
                started = time.perf_counter()
                phase_name = "image_ratio_mse_backward" if name == "image" else f"{name}_backward"
                with _profile_record(profiler, f"update_{update_index}/{phase_name}"):
                    partial, has_backward = self._backward_track(
                        name, tracks[name], slices_by_track[name], training_progress=training_progress
                    )
                phase_times[f"{phase_name}_host_time_s"] = time.perf_counter() - started
                results[name] = partial
                any_backward = any_backward or has_backward

            if any_backward:
                with _profile_record(profiler, f"update_{update_index}/optimizer"):
                    if not park_optimizer:
                        empty_cache_started = time.perf_counter()
                        if self.num_updates_per_batch > 1 and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        phase_times["pre_optimizer_empty_cache_host_time_s"] = time.perf_counter() - empty_cache_started
                    # Adam moments are absent for forward/backward and return to
                    # their exact original per-slot devices only for the step.
                    # Reclamation is part of restore and runs for U=1 too.
                    restore_optimizer_state()
                    optimizer_started = time.perf_counter()
                    grad_norm = float(self.fsdp_backend.optimizer_step(max_grad_norm=float(self.max_grad_norm)))
                    phase_times["optimizer_host_time_s"] = time.perf_counter() - optimizer_started
                    if cuda_peak_telemetry:
                        train_optimizer_metrics.update(self._cuda_phase_memory_metrics("post_optimizer_step"))
                    if self.empty_cache_after_optimizer and torch.cuda.is_available():
                        empty_cache_started = time.perf_counter()
                        torch.cuda.empty_cache()
                        phase_times["post_optimizer_empty_cache_host_time_s"] = (
                            time.perf_counter() - empty_cache_started
                        )
            else:
                if park_optimizer:
                    finish_errors = finish_prepared_state(succeeded=True)
                    if finish_errors:
                        action, exc = finish_errors[0]
                        raise RuntimeError(f"Unified train cleanup failed in {action}.") from exc
                    self.fsdp_backend.zero_grad()
                restore_optimizer_state()
                grad_norm = 0.0
                logger.warning("UnifiedModelTrainStack._train_one_step: no algorithm reported backward; skipping step.")

            cuda_peak_metrics = self._cuda_peak_memory_metrics() if cuda_peak_telemetry else {}

            lr = self._current_lr()
            for name, result in list(results.items()):
                metrics = dict(result.metrics)
                if name == "ar" and "ar_backward_host_time_s" in phase_times:
                    metrics["ar_backward_host_time_s"] = phase_times["ar_backward_host_time_s"]
                if name == "image":
                    for metric_name in (
                        "image_prepare_reference_host_time_s",
                        "image_ratio_mse_backward_host_time_s",
                        "pre_optimizer_empty_cache_host_time_s",
                        "optimizer_host_time_s",
                        "post_optimizer_empty_cache_host_time_s",
                    ):
                        if metric_name in phase_times:
                            metrics[metric_name] = phase_times[metric_name]
                    if anchor_image_host_time_s is not None:
                        metrics["anchor_image_host_time_s"] = float(anchor_image_host_time_s)
                    metrics.update(train_optimizer_metrics)
                    metrics.update(cuda_peak_metrics)
                results[name] = TrainStepResult(
                    loss=result.loss,
                    grad_norm=grad_norm,
                    lr=lr,
                    has_backward=result.has_backward,
                    micros=result.micros,
                    metrics=metrics,
                )
            succeeded = True
            return results
        finally:
            if not optimizer_park_attempted:
                # Preserve the historical default-off cleanup path exactly.
                for algorithm in self.algorithms.values():
                    if algorithm.prepares_update_batch:
                        algorithm.finish_update_batch(succeeded=succeeded)
            else:
                active_error = sys.exc_info()[0] is not None
                cleanup_errors = finish_prepared_state(succeeded=succeeded)

                if not succeeded:
                    try:
                        self.fsdp_backend.zero_grad()
                    except BaseException as exc:
                        cleanup_errors.append(("optimizer gradient cleanup", exc))
                    if optimizer_restored:
                        try:
                            reclaim_allocator()
                        except BaseException as exc:
                            cleanup_errors.append(("CUDA allocator reclaim", exc))
                    else:
                        # Prepared state is gone and failed gradients are cleared
                        # before the retryable restore attempts to consume GPU RAM.
                        try:
                            restore_optimizer_state()
                        except BaseException as exc:
                            cleanup_errors.append(("optimizer state restore", exc))

                if cleanup_errors:
                    if active_error:
                        for action, exc in cleanup_errors:
                            logger.error(
                                "Unified train cleanup failed in %s: %s",
                                action,
                                exc,
                                exc_info=(type(exc), exc, exc.__traceback__),
                            )
                    else:
                        action, exc = cleanup_errors[0]
                        raise RuntimeError(f"Unified train cleanup failed in {action}.") from exc

    def _prepare_update_batch(
        self,
        name: str,
        track: RolloutTrack,
        micro_slices: List[Tuple[int, int]],
        *,
        training_progress: float,
        update_index: int = 0,
    ) -> None:
        """Run one algorithm's detached preparation at update geometry."""
        algorithm = self.algorithms[name]
        phased = bool(getattr(algorithm, "prepares_phased_update_batch", False))
        micro_batches = []
        for start, end in micro_slices:
            micro = track.slice(start, end)
            if micro.segment is None:
                raise RuntimeError(
                    f"UnifiedModelTrainStack._prepare_update_batch: track {name!r} "
                    "produced a micro-batch with segment=None."
                )
            if phased and micro.advantages is None:
                raise RuntimeError(
                    f"UnifiedModelTrainStack._prepare_update_batch: track {name!r} "
                    "produced a micro-batch with advantages=None."
                )
            if phased:
                micro_batches.append((micro.conditions, micro.segment, micro.advantages))
            else:
                micro_batches.append((micro.conditions, micro.segment))
        if phased:
            kwargs = {
                "micro_batches": micro_batches,
                "training_progress": float(training_progress),
                "loss_scale": 1.0 / len(micro_slices),
            }
            if bool(getattr(algorithm, "prepares_indexed_update_batch", False)):
                kwargs["update_index"] = int(update_index)
            algorithm.prepare_update_batch(**kwargs)
        else:
            algorithm.prepare_update_batch(micro_batches=micro_batches)

    def on_rollout_end(self) -> None:
        """Per-rollout-boundary hook — delegates to the FSDPBackend's EMA."""
        self.fsdp_backend.on_rollout_end()

    def _train_step_profiler(self):
        """Lazily build the per-worker train-step profiler (None unless UNIRL_PROFILE)."""
        cached = getattr(self, "_profiler_cache", "unset")
        if cached == "unset":
            from unirl.utils.profiling import maybe_build_train_profiler

            cached = maybe_build_train_profiler(int(getattr(self.fsdp_backend, "_rank", 0)))
            self._profiler_cache = cached
        return cached

    @distributed(dispatch_mode=Dispatch.DP_SCATTER, collect_fn=_collect_unified_train_results)
    def train_track(
        self,
        ar_track: RolloutTrack,
        image_track: RolloutTrack,
        *,
        training_progress: float,
        optimizer_state_already_parked: bool = False,
    ) -> Dict[str, TrainStepResult]:
        """Driver-callable: prepare → backward(ar) + backward(image) → ONE step.

        Both tracks arrive DP_SCATTER-sharded (each DP worker gets its shard of
        both). ``prepare_segment`` freezes each track's π_old anchor ONCE; then the
        shard is split into ``num_updates_per_batch`` disjoint mini-batches and one
        optimizer step runs per mini-batch (each: backward ar + image over its
        mini-batch → one shared step). The 2nd+ step is off-policy, so the clip /
        ratio trust region engages; ``num_updates_per_batch=1`` is the prior
        single-step behavior. Per-track results are reduced across the updates;
        per-shard results merge back via ``pytree_cat`` on collect.
        """
        park_optimizer = bool(getattr(self, "park_optimizer_state_during_train", False))
        if optimizer_state_already_parked and not park_optimizer:
            raise ValueError("optimizer_state_already_parked requires park_optimizer_state_during_train=true.")
        cuda_peak_telemetry = bool(getattr(self, "cuda_peak_telemetry", False))
        train_window_peak_metrics: Dict[str, float] = {}

        def record_train_window_peak() -> None:
            for name, value in self._cuda_train_window_peak_memory_metrics().items():
                train_window_peak_metrics[name] = max(train_window_peak_metrics.get(name, 0.0), value)

        # This reset starts the whole-train window. Each update resets again for
        # the established cuda_peak_* contract; snapshots on both sides of those
        # resets preserve hydration + anchor + all update peaks under distinct names.
        if cuda_peak_telemetry:
            self._reset_cuda_peak_memory_stats()

        # Move both tracks onto this worker's model device before any replay.
        # The HI3 rollout tracks are hydrated to CPU on the driver (the two
        # anchored engines return single transport handles that the driver
        # materializes off-GPU before re-sharding), so segment latents / AR
        # tokens / fused conditions arrive on CPU while the backbone is on cuda.
        # One to_device here covers both algorithms' replays (AR teacher-force +
        # diffusion step) and their conditions — no per-replay device juggling.
        device = self.fsdp_backend._device
        ar_track = ar_track.to_device(device)
        image_track = image_track.to_device(device)

        # Only UNIRL_PROFILE=train applies here (one-update lives in TrainStack._run_updates);
        # warn if one-update was set so it isn't silently ignored.
        from unirl.utils.profiling import profile_scope

        scope = profile_scope()
        if scope == "one-update" and not getattr(self, "_warned_one_update", False):
            self._warned_one_update = True
            logger.warning(
                "UNIRL_PROFILE=one-update is not supported on the unified-model stack "
                "(no _run_updates loop); use UNIRL_PROFILE=train. No trace produced."
            )
        profiler = self._train_step_profiler() if scope == "train" else None
        with profiler.record("train_track") if profiler is not None else nullcontext():
            tracks = {"ar": ar_track, "image": image_track}
            # N optimizer steps over disjoint mini-batches (each track sliced by the same
            # shared _optimizer_step_slices; M=1 keeps ar/image 1:1 and equally sized).
            steps_by_track = {
                name: self._optimizer_step_slices(int(tracks[name].batch_size)) for name in self.algorithms
            }
            anchor_batch_algorithms = [
                algorithm
                for algorithm in self.algorithms.values()
                if bool(getattr(algorithm, "prepares_anchor_plan", False))
            ]
            train_succeeded = False
            inherited_optimizer_state_pending = bool(optimizer_state_already_parked)
            try:
                # Freeze each track's π_old anchor before the first optimizer step.
                # An opt-in algorithm receives the full disjoint-update partition and
                # may derive update 0's anchor lazily from its identical current replay.
                anchor_image_host_time_s: Optional[float] = None
                for name, algorithm in self.algorithms.items():
                    started = time.perf_counter()
                    with _profile_record(profiler, f"anchor_{name}"):
                        if bool(getattr(algorithm, "prepares_anchor_plan", False)):
                            self._prepare_anchor_batch(name, tracks[name], steps_by_track[name])
                        else:
                            self.prepare_segment(name, tracks[name])
                    if name == "image":
                        anchor_image_host_time_s = time.perf_counter() - started

                per_update: List[Dict[str, TrainStepResult]] = []
                for u in range(self.num_updates_per_batch):
                    if cuda_peak_telemetry:
                        record_train_window_peak()
                    slices_by_track = {name: steps_by_track[name][u] for name in self.algorithms}
                    update_inherits_parked_state = bool(inherited_optimizer_state_pending and u == 0)
                    if update_inherits_parked_state:
                        # _train_one_step now owns the restore plan and guarantees
                        # its own exception cleanup, so the anchor-level fallback
                        # must not race or duplicate that ownership.
                        inherited_optimizer_state_pending = False
                    per_update.append(
                        self._train_one_step(
                            tracks,
                            slices_by_track,
                            training_progress=float(training_progress),
                            update_index=u,
                            profiler=profiler,
                            anchor_image_host_time_s=anchor_image_host_time_s if u == 0 else None,
                            optimizer_state_already_parked=update_inherits_parked_state,
                        )
                    )
                if cuda_peak_telemetry:
                    record_train_window_peak()
                train_succeeded = True
            finally:
                if not optimizer_state_already_parked:
                    for algorithm in anchor_batch_algorithms:
                        algorithm.finish_anchor_batch(succeeded=train_succeeded)
                else:
                    active_error = sys.exc_info()[0] is not None
                    cleanup_errors: List[Tuple[str, BaseException]] = []
                    for algorithm in anchor_batch_algorithms:
                        try:
                            algorithm.finish_anchor_batch(succeeded=train_succeeded)
                        except BaseException as exc:
                            cleanup_errors.append((f"{type(algorithm).__name__}.finish_anchor_batch", exc))
                    if inherited_optimizer_state_pending:
                        try:
                            self.fsdp_backend.zero_grad()
                        except BaseException as exc:
                            cleanup_errors.append(("optimizer gradient cleanup", exc))
                        try:
                            self.fsdp_backend.reclaim_cuda_allocator()
                        except BaseException as exc:
                            cleanup_errors.append(("CUDA allocator reclaim", exc))
                        try:
                            report = self.fsdp_backend.restore_optimizer_state_after_rollout()
                            pending = float(report.get("optimizer_state_restore_slots_pending", 0.0))
                            if pending:
                                raise RuntimeError(f"optimizer restore left {int(pending)} state slot(s) pending.")
                        except BaseException as exc:
                            cleanup_errors.append(("optimizer state restore", exc))
                    if cleanup_errors:
                        if active_error:
                            for action, exc in cleanup_errors:
                                logger.error(
                                    "Unified anchor cleanup failed in %s: %s",
                                    action,
                                    exc,
                                    exc_info=(type(exc), exc, exc.__traceback__),
                                )
                        else:
                            action, exc = cleanup_errors[0]
                            raise RuntimeError(f"Unified anchor cleanup failed in {action}.") from exc
        if profiler is not None:
            profiler.step()

        self.on_rollout_end()

        # Reduce each track's per-optimizer-step results into one summary, attaching
        # each optimizer step's own metrics on ``per_update`` so the logger emits ONE
        # wandb point per optimizer update (on-policy update0 vs off-policy update1+
        # stay distinct series instead of being averaged into one misleading
        # ratio_mean). Mirrors TrainStack.train_track; passthrough at num_updates==1.
        results: Dict[str, TrainStepResult] = {}
        for name in self.algorithms:
            updates = [upd[name] for upd in per_update]
            aggregated = _aggregate_update_results(updates)
            if name == "image" and cuda_peak_telemetry:
                summary_metrics = dict(aggregated.metrics)
                for metric_name in _CUDA_PEAK_METRICS:
                    values = [float(update.metrics[metric_name]) for update in updates if metric_name in update.metrics]
                    if values:
                        summary_metrics[metric_name] = max(values)
                summary_metrics.update(train_window_peak_metrics)
                aggregated = replace(aggregated, metrics=summary_metrics)
            if len(updates) > 1:
                update_metrics = [
                    {
                        **dict(result.metrics),
                        "loss": float(result.loss),
                        "grad_norm": float(result.grad_norm),
                        "lr": float(result.lr),
                    }
                    for result in updates
                ]
                if name == "image" and train_window_peak_metrics:
                    # The window is complete only after the final update. Attach
                    # it there so per-update loggers emit one honest point.
                    update_metrics[-1].update(train_window_peak_metrics)
                aggregated = replace(
                    aggregated,
                    per_update=tuple(update_metrics),
                )
            results[name] = aggregated
        return results

    def _current_lr(self) -> float:
        optimizer = self.fsdp_backend.optimizer
        param_groups = getattr(optimizer, "param_groups", None)
        if isinstance(param_groups, list) and param_groups:
            return float(param_groups[0]["lr"])
        scheduler = self.fsdp_backend.scheduler
        if scheduler is not None and hasattr(scheduler, "get_last_lr"):
            last = scheduler.get_last_lr()
            if isinstance(last, list) and last:
                return float(last[0])
        return 0.0


__all__ = ["UnifiedModelTrainStack"]
