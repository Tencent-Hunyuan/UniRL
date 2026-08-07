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
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Dict, List, Mapping, Optional, Tuple, Union

import torch

from unirl.algorithms import AlgorithmStepResult, StageAlgorithm
from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.distributed.group.remote import Remote
from unirl.distributed.tensor.batch import _move_value
from unirl.train.backend.fsdp import FSDPBackend
from unirl.train.stack.planner import CountPlanner, MicroPlanner, Plan, UpdatePlan, _positive_int
from unirl.types.sample import Part
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


def _align_track_to_model(part: Part, *, device: torch.device) -> None:
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
    if part.segment is not None:
        part.segment = part.segment.to_device(device)
    part.conditions = {k: _move_value(v, device) for k, v in part.conditions.items()}
    if part.advantages is not None:
        part.advantages = part.advantages.to(device=device)


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
        self.micro_planner: MicroPlanner = micro_planner if micro_planner is not None else CountPlanner()
        self.micro_planner.validate(algorithm)

    def prepare_segment(self, part: Part, *, plans: Plan) -> None:
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
        if part.segment is None:
            return
        algorithm = self.algorithm
        if not algorithm.recomputes_anchor():
            algorithm.prepare_segment(conditions=part.conditions, segment=part.segment)
            return
        micro_slices = [r for update in plans for r in update]
        if len(micro_slices) == 1:
            algorithm.prepare_segment(conditions=part.conditions, segment=part.segment)
            return
        collected: Dict[str, List[torch.Tensor]] = {field: [] for field in algorithm.anchor_fields}
        for start, end in micro_slices:
            micro = part.slice(start, end)
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
            setattr(part.segment, field, torch.cat(parts, dim=0))

    def _run_update(
        self,
        part: Part,
        *,
        micros: UpdatePlan,
        training_progress: float,
        zero_grad: bool = True,
        do_optimizer_step: bool = True,
        loss_weight: float = 1.0,
        prior_backward: bool = False,
    ) -> TrainStepResult:
        """Run the micro ranges of a single update; step unless mid-window.

        ``micros`` is one update's worth of ``(start, end)`` ranges produced by
        :meth:`~unirl.train.stack.planner.MicroPlanner.arrange` so the forward
        geometry matches the π_old anchor frozen by :meth:`prepare_segment`.

        The keyword flags are :meth:`_run_window`'s internal mechanics
        (``prior_backward``: whether an earlier part of the window backwarded);
        the defaults are the plain one-step-per-call contract.
        """
        if part.advantages is None and getattr(self.algorithm, "requires_advantages", True):
            raise ValueError(
                f"{type(self).__name__}._run_update: part.advantages is None; "
                "upstream advantage pipeline must populate it before training. "
                "(Supervised algorithms opt out by declaring requires_advantages=False.)"
            )
        if not micros:
            raise ValueError(f"{type(self).__name__}._run_update: empty micros.")

        bs = int(part.batch_size)
        if zero_grad:
            self.fsdp_backend.zero_grad()

        loss_scales, global_weight = self._resolve_loss_scales(part, micros=micros)
        micro_results: List[AlgorithmStepResult] = []
        total_loss = 0.0
        weighted_loss_sum = 0.0
        has_backward = False

        single_micro = len(micros) == 1 and micros[0] == (0, bs)
        last_micro = len(micros) - 1
        for i, (start, end) in enumerate(micros):
            # Defer gradient reduce-scatter until the stepping rollout's final microbatch.
            self.fsdp_backend.set_grad_sync(do_optimizer_step and i == last_micro)
            micro_part = part if single_micro else part.slice(start, end)
            result = self.algorithm.compute_loss_and_backward(
                conditions=micro_part.conditions,
                segment=micro_part.segment,
                advantages=micro_part.advantages,
                training_progress=training_progress,
                # loss_weight (1/M) scales only the backward; result.loss stays the raw micro mean.
                loss_scale=loss_scales[i] * loss_weight,
            )
            micro_results.append(result)
            if global_weight is None:
                total_loss += result.loss * loss_scales[i]
            else:
                weighted_loss_sum += result.loss * self._micro_loss_weight(part, start, end)
            has_backward = has_backward or result.has_backward

        aggregated_metrics: Mapping[str, object] = aggregate_numeric_metrics(
            [r.metrics for r in micro_results if r.metrics]
        )

        window_backward = prior_backward or has_backward
        # The stepping backward must run when deferred gradient sync is enabled.
        if (
            do_optimizer_step
            and window_backward
            and not micro_results[-1].has_backward
            and self.fsdp_backend.grad_sync_deferred
        ):
            raise RuntimeError(
                f"{type(self).__name__}._run_update: defer_grad_sync deferred the gradient "
                "reduce-scatter to the stepping backward, but the last micro-batch before the "
                "optimizer step reported no backward (all-empty micro?) while earlier backwards "
                "ran — the accumulated grads were never synced. Disable "
                "training.fsdp.defer_grad_sync or investigate the empty micro-batch."
            )

        if not do_optimizer_step:
            # Accumulation-only part: the window's final part steps.
            grad_norm = 0.0
        elif window_backward:
            grad_norm = float(self.fsdp_backend.optimizer_step(max_grad_norm=float(self.max_grad_norm)))
        else:
            grad_norm = 0.0
            logger.warning(
                "%s._run_update: no micro reported backward; skipping optimizer step.",
                type(self).__name__,
            )
        if torch.cuda.is_available():
            # CUDA memory footprint per optimizer step (leak diagnosis: tp2 path showed progressive OOM).
            aggregated_metrics = {
                **dict(aggregated_metrics),
                "cuda_alloc_gb": torch.cuda.memory_allocated() / 2**30,
                "cuda_reserved_gb": torch.cuda.memory_reserved() / 2**30,
            }

        if global_weight is None:
            (global_loss_sum,) = self._all_reduce_sums([total_loss])
            total_loss = global_loss_sum / self._loss_weight_world()
        else:
            (global_loss_sum,) = self._all_reduce_sums([weighted_loss_sum])
            total_loss = global_loss_sum / global_weight
            aggregated_metrics = {**dict(aggregated_metrics), "global_loss_weight": global_weight}

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

    def _resolve_loss_scales(self, part: Part, *, micros: UpdatePlan) -> Tuple[List[float], Optional[float]]:
        """Per-micro ``loss_scale`` factors for one optimizer step.

        ``algorithm.loss_weighting`` selects the convention (see
        :class:`~unirl.algorithms.StageAlgorithm`):

        - ``"sample"``: micro's share of the update's samples. Sums to 1 per
          rank; FSDP's DP-mean gradient reduction then yields the rank-mean of
          shard means (exact global sample-mean, since DP_SCATTER shards are
          equal-sized). Returns ``(scales, None)``.
        - ``"token"``: ``micro_tokens * dp_world / global_tokens`` where
          ``global_tokens`` is the valid-token count of the WHOLE optimizer
          step, all-reduced across the training group. The ``* dp_world``
          cancels FSDP's gradient averaging, so the update gradient equals the
          full-batch token-mean regardless of micro packing or DP layout.
          Returns ``(scales, global_tokens)``.
        """
        weighting = str(getattr(self.algorithm, "loss_weighting", "sample"))
        if weighting == "sample":
            update_total = sum(end - start for start, end in micros)
            return [(end - start) / update_total for start, end in micros], None
        if weighting != "token":
            raise ValueError(
                f"{type(self).__name__}: unknown algorithm.loss_weighting={weighting!r}; expected 'sample' or 'token'."
            )
        rank_info = getattr(self, "rank_info", None)
        if rank_info is not None and rank_info.sp_size > 1:
            # Reject sequence parallelism until loss denominators include the SP dimension.
            raise ValueError(
                f"{type(self).__name__}: loss_weighting='token' is not validated under "
                f"sequence parallelism (sp_size={rank_info.sp_size}); use sp_size=1."
            )
        weights = [self._micro_loss_weight(part, start, end) for start, end in micros]
        local_total = sum(weights)
        (global_total,) = self._all_reduce_sums([local_total])
        if global_total <= 0.0:
            raise ValueError(
                f"{type(self).__name__}: zero valid tokens in this optimizer step "
                "(fully-masked batch?) — the data source must not emit steps with no "
                "supervision (0/0 loss NaNs destroyed checkpoints in verl #785)."
            )
        dp_world = self._loss_weight_world()
        return [w * dp_world / global_total for w in weights], global_total

    def _micro_loss_weight(self, part: Part, start: int, end: int) -> float:
        """Valid-token count of one contiguous micro range (loss_mask-aware)."""
        segment = part.segment
        if segment is None:
            raise ValueError(f"{type(self).__name__}: loss_weighting='token' requires a segment.")
        cu = segment.cu_seqlens
        loss_mask = getattr(segment, "loss_mask", None)
        if loss_mask is not None and cu is not None:
            return float(loss_mask[int(cu[start]) : int(cu[end])].sum().item())
        if segment.lengths is not None:
            return float(segment.lengths[start:end].sum().item())
        raise ValueError(
            f"{type(self).__name__}: loss_weighting='token' requires a packed segment "
            "(cu_seqlens/lengths) — build it via TextSegment.pack(...)."
        )

    def _loss_weight_world(self) -> int:
        """World size whose gradient averaging the token weighting must cancel."""
        return self.fsdp_backend.gradient_average_world_size()

    def _all_reduce_sums(self, values: List[float]) -> List[float]:
        """SUM scalars over the backend's FSDP mesh (no-op single-rank).

        Every rank must call this the same number of times per step — callers keep
        the collective count rank-symmetric (micro plans are; DP_SCATTER shards are
        equal-sized).
        """
        return self.fsdp_backend.all_reduce_loss_sums(values)

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def eval_track(self, part: Part) -> Dict[str, float]:
        """Weighted forward-only loss over this shard; returns GLOBAL metrics.

        Drives ``algorithm.evaluate_loss(conditions=..., segment=...) ->
        (loss_sum, weight)`` micro-by-micro under ``torch.no_grad()`` with the
        trainable module in eval mode (dropout off — the same loss the train
        path optimizes, never a second implementation). Per-rank sums are
        SUM-all-reduced, so every rank returns the identical global weighted
        mean and the driver's collected dict is exact regardless of which
        rank's value survives the merge.
        """
        eval_fn = getattr(self.algorithm, "evaluate_loss", None)
        if not callable(eval_fn):
            raise TypeError(
                f"{type(self).__name__}.eval_track: {type(self.algorithm).__name__} does not "
                "expose evaluate_loss(conditions=..., segment=...) -> (loss_sum, weight)."
            )
        self._align_track_inputs(part)
        model = self.fsdp_backend.trainable_module()
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                bs = part.batch_size
                mbs = self.micro_batch_size
                loss_sum = 0.0
                weight_sum = 0.0
                for start in range(0, bs, mbs):
                    end = min(start + mbs, bs)
                    micro = part.slice(start, end)
                    s, w = eval_fn(
                        conditions=micro.conditions,
                        segment=micro.segment,
                        sample_ids=list(micro.sample_ids) if micro.sample_ids else None,
                    )
                    loss_sum += float(s)
                    weight_sum += float(w)
        finally:
            model.train(was_training)
        global_loss, global_weight = self._all_reduce_sums([loss_sum, weight_sum])
        if global_weight <= 0.0:
            raise ValueError(f"{type(self).__name__}.eval_track: zero eval weight (empty/fully-padded batch?).")
        return {"loss": global_loss / global_weight, "weight": global_weight}

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def train_track(
        self,
        parts: Union[Part, Tuple[Part, ...]],
        *,
        training_progress: float,
    ) -> TrainStepResult:
        """Driver-callable: arrange → prepare → run updates → on_rollout_end.

        One call = ONE optimizer step. A bare ``Part`` is the plain path; a
        TUPLE of Parts is a gradient-accumulation window (the MOPD task cycle):
        every part is prepared and backwarded at ``1/len(parts)`` on the same
        never-stepped weights, and the single step sees the mean gradient of
        the whole window. The window lives inside this one call, so a
        checkpoint between calls can never split it. Tuple, not list — the
        DP_SCATTER dispatcher recurses tuples element-wise (each part is
        row-sharded), while lists are row-sliced as data.

        Combines the steps so worker-side mutations (``segment.sde_logp`` populated
        by ``prepare_segment``) flow into the subsequent update(s) without
        round-tripping through the driver. Dispatched ``DP_SCATTER`` so each DP
        worker receives its shard of every part; per-shard loss/grad_norm/metrics
        merge back via ``pytree_merge``.

        ``arrange`` reorders each shard (if packing) and builds the contiguous
        plan; ``prepare_segment`` freezes the π_old anchor at that geometry;
        ``num_updates_per_batch`` optimizer steps (single-part calls only) run
        over disjoint updates; ``on_rollout_end`` runs once.
        """
        window = parts if isinstance(parts, tuple) else (parts,)
        if not window:
            raise ValueError(f"{type(self).__name__}.train_track: empty accumulation window.")
        if len(window) > 1 and self.num_updates_per_batch > 1:
            raise ValueError(
                f"{type(self).__name__}.train_track: an accumulation window of {len(window)} parts "
                f"requires num_updates_per_batch == 1 (got {self.num_updates_per_batch}) — extra "
                "optimizer steps inside the window would re-step on partial gradients."
            )
        arranged = []
        for part in window:
            self._align_track_inputs(part)
            arranged.append(
                self.micro_planner.arrange(
                    part,
                    num_updates=self.num_updates_per_batch,
                    micro_batch_size=self.micro_batch_size,
                )
            )
        from unirl.utils.profiling import profile_scope

        profiler = self._train_step_profiler() if profile_scope() == "train" else None
        with profiler.record("train_track") if profiler is not None else nullcontext():
            if len(arranged) == 1:
                part, plans = arranged[0]
                part = self._prepare_for_training(part, plans=plans)
                result = self._run_updates(part, plans=plans, training_progress=float(training_progress))
            else:
                result = self._run_window(arranged, training_progress=float(training_progress))
        if profiler is not None:
            profiler.step()
        self.on_rollout_end()
        return result

    def _prepare_for_training(self, part: Part, *, plans: Plan) -> Part:
        """Freeze this part's anchor in eval mode, then return the model to train mode.

        Eval mode keeps train-time stochasticity (notably dropout) out of the
        anchor forward; the gradient-bearing replay needs train mode so HF
        gradient checkpointing, which is gated on ``self.training``, engages.
        """
        self.fsdp_backend.model.eval()
        self.prepare_segment(part, plans=plans)
        part = self.algorithm.prepare_part(part)
        self.fsdp_backend.model.train()
        return part

    def _run_window(self, arranged: List[Tuple[Part, Plan]], *, training_progress: float) -> TrainStepResult:
        """One optimizer step over an accumulation window of single-update parts.

        Zero once, then per part: prepare its anchor and immediately backward its
        micros at ``1/len(arranged)``. Preparing each part right before its own
        backward keeps any per-part state the algorithm sets in ``prepare_part``
        (DiffusionOPD's active teacher, which names the per-domain loss metric)
        valid for that part's loss. Gradients are identical either way — no step
        happens inside the window, so every part sees the same weights.

        The (ZeRO-2) gradient sync is deferred to the final part's last micro and
        the step runs once. The merged result reports the window-mean loss, the
        stepping call's grad_norm, and the union of per-part metrics — per-domain
        keys land side by side in one point.
        """
        m = len(arranged)
        self.fsdp_backend.zero_grad()
        results: List[TrainStepResult] = []
        prior_backward = False
        for w, (part, plans) in enumerate(arranged):
            part = self._prepare_for_training(part, plans=plans)
            (micros,) = plans  # window parts are single-update (validated in train_track)
            result = self._run_update(
                part,
                micros=micros,
                training_progress=training_progress,
                zero_grad=False,
                do_optimizer_step=(w == m - 1),
                loss_weight=1.0 / m,
                prior_backward=prior_backward,
            )
            prior_backward = prior_backward or result.has_backward
            results.append(result)
        # The window is one optimizer step: its grad_norm is the stepping call's.
        return replace(_aggregate_update_results(results), grad_norm=results[-1].grad_norm)

    def _train_step_profiler(self):
        """Lazily build the per-worker train-step profiler (None unless UNIRL_PROFILE)."""
        cached = getattr(self, "_profiler_cache", "unset")
        if cached == "unset":
            from unirl.utils.profiling import maybe_build_train_profiler

            cached = maybe_build_train_profiler(int(getattr(self.fsdp_backend, "_rank", 0)))
            self._profiler_cache = cached
        return cached

    def _run_updates(
        self,
        part: Part,
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
        from unirl.utils.profiling import maybe_profile_update, profile_scope

        scope_update = profile_scope() == "one-update"
        results = []
        for micros in plans:
            cm = (
                maybe_profile_update(self, int(getattr(self.fsdp_backend, "_rank", 0)))
                if scope_update
                else nullcontext()
            )
            with cm:
                results.append(self._run_update(part, micros=micros, training_progress=training_progress))
        if len(results) == 1:
            return results[0]
        aggregated = _aggregate_update_results(results)
        per_update = tuple(
            {**dict(r.metrics), "loss": float(r.loss), "grad_norm": float(r.grad_norm), "lr": float(r.lr)}
            for r in results
        )
        return replace(aggregated, per_update=per_update)

    def _align_track_inputs(self, part: Part) -> None:
        """Move the track onto the model's device; see :func:`_align_track_to_model`."""
        device = next(self.fsdp_backend.trainable_module().parameters()).device
        _align_track_to_model(part, device=device)

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
