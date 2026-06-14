"""Family-agnostic single-stage train stack.

:class:`TrainStack` wraps one :class:`FSDPBackend` (training state: model +
optimizer + scheduler + EMA) and one :class:`StageAlgorithm` (loss + backward
against the bundle's trainable module) into a single-stage training driver. One
stack = one training track.

It owns the entire family-agnostic pipeline — device alignment, the π_old anchor
freeze, the per-update micro-accumulation loop, EMA, metrics — and defers exactly
ONE decision to an injected :class:`~unirl.train.stack.planner.MicroPlanner`
(composition, not inheritance): how each update's samples are grouped into
micro-batches. :class:`~unirl.train.stack.planner.CountPlanner` (the default)
groups by fixed count; :class:`~unirl.train.stack.planner.TokenBudgetPlanner`
packs by token budget. Swapping the strategy is a recipe-level ``micro_planner``
block, no subclass.

Sequencing per :meth:`train_track` call (one rollout)::

    track, plans = micro_planner.arrange(track)  # reorder (if packing) + plan
    prepare_segment(track, plans)                # once: freeze the π_old anchor
    for micros in plans:                         # num_updates_per_batch updates
        _run_update(track, micros=micros)        # one optimizer step each
    on_rollout_end()                             # once: EMA / rollout boundary

**Sort-then-slice.** Variable-length packing wants to group samples of similar
length, which would normally force arbitrary index lists threaded through the
whole pipeline. Instead the planner *reorders the track once up front* (length-sort
within each update, see :meth:`~unirl.train.stack.planner.TokenBudgetPlanner.arrange`)
so every micro is again a **contiguous** ``(start, end)`` range — exactly the
count-based geometry. The stack therefore only ever slices, and the anchor
reassembly is a plain ordered ``cat``; all packing-specific logic lives in the
planner (a no-op for :class:`~unirl.train.stack.planner.CountPlanner`).

``num_updates_per_batch`` partitions the rollout batch into that many disjoint
updates and runs one optimizer step per update — the FlowGRPO / DanceGRPO
schedule. Because ``prepare_segment`` captures the pre-update policy once, every
update shares the same PPO anchor; this is only correct for algorithms with
``supports_multi_update`` (the ctor enforces it). Defaults to 1.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Dict, List, Mapping, Optional, Tuple

import torch

from unirl.algorithms import AlgorithmStepResult, StageAlgorithm
from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.distributed.group.remote import Remote
from unirl.distributed.tensor.batch import _move_value
from unirl.train.backend.fsdp import FSDPBackend
from unirl.train.stack.planner import CountPlanner, MicroPlanner, Plan, UpdatePlan, _positive_int
from unirl.types.rollout_resp import RolloutTrack
from unirl.utils.misc import aggregate_numeric_metrics

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainStepResult:
    """Result of one full optimizer step on this stage."""

    loss: float
    grad_norm: float
    lr: float
    has_backward: bool
    micros: List[AlgorithmStepResult]
    metrics: Mapping[str, object]
    # Per-optimizer-step metrics when num_updates_per_batch > 1 (one Mapping per
    # update, in order); empty for the single-update path. Lets the trainer log one
    # wandb point per optimizer step instead of averaging the updates.
    per_update: Tuple[Mapping[str, object], ...] = ()


def _aggregate_update_results(results: List["TrainStepResult"]) -> "TrainStepResult":
    """Collapse one rollout's per-update results into a single summary.

    Scalars are averaged across the N optimizer steps (``lr`` is the last,
    post-step value), ``micros`` are concatenated, and algorithm metrics are
    averaged via :func:`aggregate_numeric_metrics`. Downstream logging then treats
    the whole rollout as one point, exactly as in the single-update path.
    """
    if len(results) == 1:
        return results[0]
    n = len(results)
    micros: List[AlgorithmStepResult] = [m for r in results for m in r.micros]
    metrics = aggregate_numeric_metrics([dict(r.metrics) for r in results if r.metrics])
    return TrainStepResult(
        loss=sum(r.loss for r in results) / n,
        grad_norm=sum(r.grad_norm for r in results) / n,
        lr=results[-1].lr,
        has_backward=any(r.has_backward for r in results),
        micros=micros,
        metrics=metrics,
    )


def _align_track_to_model(resp_track: RolloutTrack, *, device: torch.device) -> None:
    """Move a track's training inputs onto the model's device — SGLang returns them
    on CPU via Ray IPC. Uses :meth:`Batch.to_device` (recursive; carries
    framework-managed ``_packed_cu_seqlens`` and tensors nested in tuples/dicts) on
    the segment + conditions only, so heavy ``decoded`` / ``media_preview`` payloads
    stay off the GPU. dtype is left to the model, which casts what it feeds the
    network (see SD3DiffusionStep.predict_noise).

    Condition values are moved via ``_move_value`` (the same recursive mover
    ``Batch.to_device`` uses) rather than assuming each value is a ``Batch``: most
    are (e.g. ``TextTokenCondition``), but multimodal stages also carry raw
    per-sample ``FieldKind.CONCAT`` lists of tensors (Qwen2.5-VL's ``pixel_values``
    / ``image_grid_thw``), which have no ``.to_device`` of their own — ``_move_value``
    handles Batch / tensor / list / dict / None uniformly."""
    if resp_track.segment is not None:
        resp_track.segment = resp_track.segment.to_device(device)
    resp_track.conditions = {k: _move_value(v, device) for k, v in resp_track.conditions.items()}
    if resp_track.advantages is not None:
        resp_track.advantages = resp_track.advantages.to(device=device)


class TrainStack(Remote):
    """Single-stage stage-driven train stack — family-agnostic.

    One stage only — no track-name dict, no optional-track semantics, no multi-track
    on_rollout_end fan-out. The ONLY family-varying decision — micro-batch grouping —
    is delegated to an injected ``micro_planner`` (count-based vs token-budget);
    everything else is shared. Defaults to
    :class:`~unirl.train.stack.planner.CountPlanner` (the historical diffusion
    behaviour), so the 60+ count-based configs need no ``micro_planner`` block.

    Created as a sibling ``Remote`` inside a placement block; takes handles to its
    FSDPBackend and StageAlgorithm siblings via sibling-handle auto-resolve.
    """

    def __init__(
        self,
        *,
        fsdp_backend: FSDPBackend,
        algorithm: StageAlgorithm,
        micro_batch_size: int = 1,
        max_grad_norm: float,
        num_updates_per_batch: int = 1,
        micro_planner: Optional[MicroPlanner] = None,
    ) -> None:
        super().__init__()
        cls = type(self).__name__
        if int(micro_batch_size) < 1:
            raise ValueError(f"{cls}.micro_batch_size must be >= 1; got {micro_batch_size}.")
        if float(max_grad_norm) <= 0.0:
            raise ValueError(f"{cls}.max_grad_norm must be > 0; got {max_grad_norm}.")
        self.num_updates_per_batch = _positive_int(name=f"{cls}.num_updates_per_batch", value=num_updates_per_batch)
        if self.num_updates_per_batch > 1 and not getattr(algorithm, "supports_multi_update", False):
            raise ValueError(
                f"num_updates_per_batch={self.num_updates_per_batch} requires an algorithm whose "
                f"old_logp anchor stays frozen across the N optimizer steps "
                f"(FlowGRPO / FlowDPPO / GRPO / DRPO). "
                f"{type(algorithm).__name__} sets supports_multi_update=False, so >1 optimizer "
                f"step would train against a moving anchor. Set num_updates_per_batch=1."
            )
        self.fsdp_backend = fsdp_backend
        self.algorithm = algorithm
        self.micro_batch_size = int(micro_batch_size)
        self.max_grad_norm = float(max_grad_norm)
        # Composition: the micro-batch grouping strategy. None → the historical
        # fixed-count behaviour. The planner also owns the algorithm precondition its
        # grouping requires (e.g. token-budget packing needs a seq-mean loss),
        # checked once here at construction.
        self.micro_planner: MicroPlanner = micro_planner if micro_planner is not None else CountPlanner()
        self.micro_planner.validate(algorithm)

    def prepare_segment(self, resp_track: RolloutTrack, *, plans: Plan) -> None:
        """Freeze the π_old anchor once, before the ``num_updates_per_batch`` loop.

        No-op if ``segment`` is None. If the algorithm does NOT replay the anchor
        (``recomputes_anchor() == False`` — e.g. rollout GRPO), the anchor is the
        rollout engine's own emission, so one full-segment call suffices. If it DOES
        replay (replay GRPO; FlowDPPO always, for ``sde_means``), the recomputed
        ``anchor_fields`` are computed at the SAME micro geometry training will use —
        the contiguous ranges in ``plans`` (already aligned with the reordered track
        from :meth:`~unirl.train.stack.planner.MicroPlanner.arrange`) — so the
        old/new forwards match bf16-element-for-element on those fields. Concretely,
        the on-policy PPO ratio is exactly 1 only where ``sde_logp`` is replayed
        (replay GRPO, or FlowDPPO under ``old_logp_source='replay'``), and the
        on-policy KL is exactly 0 wherever ``sde_means`` is replayed (FlowDPPO
        always). A single micro degenerates to one full-segment call; only the
        algorithm's declared ``anchor_fields`` are re-sliced and reassembled (no
        hardcoded field names). Because every micro is a contiguous range covering
        the shard in order, the per-micro field chunks reassemble with a plain
        ordered ``cat``.
        """
        if resp_track.segment is None:
            return
        algorithm = self.algorithm
        if not algorithm.recomputes_anchor():
            algorithm.prepare_segment(conditions=resp_track.conditions, segment=resp_track.segment)
            return
        micro_slices = [r for update in plans for r in update]
        if len(micro_slices) == 1:
            algorithm.prepare_segment(conditions=resp_track.conditions, segment=resp_track.segment)
            return
        collected: Dict[str, List[torch.Tensor]] = {field: [] for field in algorithm.anchor_fields}
        for start, end in micro_slices:
            micro = resp_track.slice(start, end)
            algorithm.prepare_segment(conditions=micro.conditions, segment=micro.segment)
            for field in collected:
                value = getattr(micro.segment, field, None)
                if value is None:
                    raise RuntimeError(
                        f"{type(self).__name__}.prepare_segment: {type(algorithm).__name__} declares "
                        f"anchor field {field!r} but a micro produced None."
                    )
                collected[field].append(value)
        for field, parts in collected.items():
            setattr(resp_track.segment, field, torch.cat(parts, dim=0))

    def _run_update(
        self,
        resp_track: RolloutTrack,
        *,
        micros: UpdatePlan,
        training_progress: float,
    ) -> TrainStepResult:
        """Run one optimizer step over the contiguous micro ranges of a single update.

        ``micros`` is one update's worth of ``(start, end)`` ranges produced by
        :meth:`~unirl.train.stack.planner.MicroPlanner.arrange` so the forward
        geometry matches the π_old anchor frozen by :meth:`prepare_segment`.
        """
        if resp_track.advantages is None:
            raise ValueError(
                f"{type(self).__name__}._run_update: resp_track.advantages is None; "
                "upstream advantage pipeline must populate it before training."
            )
        if not micros:
            raise ValueError(f"{type(self).__name__}._run_update: empty micros.")

        bs = int(resp_track.batch_size)
        self.fsdp_backend.zero_grad()

        update_total = sum(end - start for start, end in micros)
        micro_results: List[AlgorithmStepResult] = []
        total_loss = 0.0
        has_backward = False

        single_micro = len(micros) == 1 and micros[0] == (0, bs)
        last_micro = len(micros) - 1
        for i, (start, end) in enumerate(micros):
            # Defer the per-block gradient reduce-scatter to the last micro-batch so
            # it runs once per optimizer step instead of once per micro-batch (no-op
            # unless defer_grad_sync + ZeRO-2). Must precede the backward.
            self.fsdp_backend.set_grad_sync(i == last_micro)
            micro_track = resp_track if single_micro else resp_track.slice(start, end)
            # Sample-share weighting: the algorithm's micro loss is a MEAN over the
            # micro's sequences (seq-mean agg modes), so the update gradient equals
            # the whole-update mean only when each micro is weighted by its share of
            # samples. With equal count-based micros this reduces to 1/len(micros);
            # with token-budget packing micros vary in size.
            loss_scale = (end - start) / float(update_total)
            result = self.algorithm.compute_loss_and_backward(
                conditions=micro_track.conditions,
                segment=micro_track.segment,
                advantages=micro_track.advantages,
                training_progress=training_progress,
                loss_scale=loss_scale,
            )
            micro_results.append(result)
            total_loss += result.loss
            has_backward = has_backward or result.has_backward

        aggregated_metrics: Mapping[str, object] = aggregate_numeric_metrics(
            [r.metrics for r in micro_results if r.metrics]
        )

        # Under defer_grad_sync the deferred reduce-scatter only runs inside a
        # backward that executes after set_grad_sync(True) — the last micro's. If
        # that micro skipped backward while earlier ones ran, the accumulated grads
        # were never synced: the optimizer would silently step on empty grads now,
        # and the stale unsharded accumulation (which zero_grad cannot reach) would
        # leak into the NEXT step's reduce-scatter. Fail fast instead — mirrors
        # fsdp_wrap's stray-trainable guard.
        if has_backward and not micro_results[-1].has_backward and self.fsdp_backend.grad_sync_deferred:
            raise RuntimeError(
                f"{type(self).__name__}._run_update: defer_grad_sync deferred the gradient "
                "reduce-scatter to the last micro-batch, but it reported no backward (all-empty "
                "micro?) while earlier micro-batches did — the accumulated grads were never "
                "synced. Disable training.fsdp.defer_grad_sync or investigate the empty micro-batch."
            )

        if has_backward:
            grad_norm = float(self.fsdp_backend.optimizer_step(max_grad_norm=float(self.max_grad_norm)))
        else:
            grad_norm = 0.0
            logger.warning(
                "%s._run_update: no micro reported backward; skipping optimizer step.",
                type(self).__name__,
            )
        if torch.cuda.is_available():
            # CUDA memory footprint per optimizer step (leak diagnosis: tp2 path
            # showed progressive OOM). Surfaces as train/cuda_alloc_gb|cuda_reserved_gb.
            aggregated_metrics = {
                **dict(aggregated_metrics),
                "cuda_alloc_gb": torch.cuda.memory_allocated() / 2**30,
                "cuda_reserved_gb": torch.cuda.memory_reserved() / 2**30,
            }

        return TrainStepResult(
            loss=total_loss,
            grad_norm=grad_norm,
            lr=self._current_lr(),
            has_backward=has_backward,
            micros=micro_results,
            metrics=aggregated_metrics,
        )

    def on_rollout_end(self) -> None:
        """Per-rollout-boundary hook — delegates to the FSDPBackend's EMA."""
        self.fsdp_backend.on_rollout_end()

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def train_track(
        self,
        resp_track: RolloutTrack,
        *,
        training_progress: float,
    ) -> TrainStepResult:
        """Driver-callable: arrange → prepare → run updates (×N) → on_rollout_end.

        Combines the steps so worker-side mutations (``segment.sde_logp`` populated
        by ``prepare_segment``) flow into the subsequent update(s) without
        round-tripping through the driver. Dispatched ``DP_SCATTER`` so each DP
        worker receives its shard of ``resp_track``; per-shard loss/grad_norm/metrics
        merge back via ``pytree_merge``.

        ``arrange`` reorders the shard (if packing) and builds the contiguous plan;
        ``prepare_segment`` then freezes the π_old anchor once at that geometry,
        ``num_updates_per_batch`` optimizer steps run over disjoint updates, and
        ``on_rollout_end`` runs once — see :meth:`_run_updates`.
        """
        self._align_track_inputs(resp_track)
        # Arrange once: reorder the track so packed micros are contiguous (no-op for
        # CountPlanner) and produce the plan. The SAME (track, plans) feed both the
        # anchor freeze and the train loop so both run the exact same geometry.
        resp_track, plans = self.micro_planner.arrange(
            resp_track,
            num_updates=self.num_updates_per_batch,
            micro_batch_size=self.micro_batch_size,
        )
        self.prepare_segment(resp_track, plans=plans)
        result = self._run_updates(resp_track, plans=plans, training_progress=float(training_progress))
        self.on_rollout_end()
        return result

    def _run_updates(
        self,
        resp_track: RolloutTrack,
        *,
        plans: Plan,
        training_progress: float,
    ) -> TrainStepResult:
        """Run ``num_updates_per_batch`` optimizer steps over disjoint updates.

        The update/micro grouping comes from
        :meth:`~unirl.train.stack.planner.MicroPlanner.arrange` — the same source
        :meth:`prepare_segment` froze the π_old anchor at — so every update's
        ``new_logp`` is computed at exactly the anchor's geometry. ``prepare_segment``
        must already have frozen the anchor so all updates train against the same
        pre-update policy. With a single optimizer step the result passes through
        unchanged; otherwise the per-update results are reduced into one summary and
        each update's own metrics are attached on ``per_update`` (see
        :func:`_aggregate_update_results`).
        """
        results = [self._run_update(resp_track, micros=micros, training_progress=training_progress) for micros in plans]
        if len(results) == 1:
            return results[0]
        aggregated = _aggregate_update_results(results)
        # Attach each optimizer step's own metrics (in order) so the trainer can log
        # one wandb point per optimizer step — the on-policy update0 and the
        # off-policy update1 stay distinct series instead of being averaged into one
        # misleading ``ratio_mean``. Structured data on the result object, which the
        # DP collect (``pytree_cat``) returns whole, so it rides along.
        per_update = tuple(
            {**dict(r.metrics), "loss": float(r.loss), "grad_norm": float(r.grad_norm), "lr": float(r.lr)}
            for r in results
        )
        return replace(aggregated, per_update=per_update)

    def _align_track_inputs(self, resp_track: RolloutTrack) -> None:
        """Move the track onto the model's device; see :func:`_align_track_to_model`."""
        device = next(self.fsdp_backend.trainable_module().parameters()).device
        _align_track_to_model(resp_track, device=device)

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
