"""Family-agnostic single-stage train stack."""

from __future__ import annotations

import logging
import math
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
from unirl.utils.metrics import aggregate_numeric_metrics

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainStepResult:
    """Result of one ``train_track`` call, possibly spanning multiple optimizer updates."""

    loss: float
    grad_norm: float
    lr: float
    has_backward: bool
    micros: List[AlgorithmStepResult]
    metrics: Mapping[str, object]
    # Steps that committed; a no-backward or non-finite grad norm contributes 0.
    optimizer_updates: int
    per_update: Tuple[Mapping[str, object], ...] = ()


def _aggregate_update_results(results: List["TrainStepResult"]) -> "TrainStepResult":
    """Collapse one rollout's per-update results into a single summary."""
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
        optimizer_updates=sum(r.optimizer_updates for r in results),
    )


def _align_track_to_model(part: Part, *, device: torch.device) -> None:
    """Move a track's training inputs onto the model's device — SGLang returns them"""
    if part.segment is not None:
        part.segment = part.segment.to_device(device)
    part.conditions = {k: _move_value(v, device) for k, v in part.conditions.items()}
    if part.advantages is not None:
        part.advantages = part.advantages.to(device=device)


class TrainStack(Remote):
    """Single-stage stage-driven train stack — family-agnostic."""

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
        """Freeze the π_old anchor once, before the ``num_updates_per_batch`` loop."""
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
        """Run the micro ranges of a single update; step unless mid-window."""
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
            optimizer_updates=1 if has_backward and math.isfinite(grad_norm) else 0,
        )

    def on_rollout_end(self) -> None:
        """Per-rollout-boundary hook — delegates to the FSDPBackend's EMA."""
        self.fsdp_backend.on_rollout_end()

    def _resolve_loss_scales(self, part: Part, *, micros: UpdatePlan) -> Tuple[List[float], Optional[float]]:
        """Per-micro ``loss_scale`` factors for one optimizer step."""
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
        """SUM scalars over the backend's FSDP mesh (no-op single-rank)."""
        return self.fsdp_backend.all_reduce_loss_sums(values)

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def eval_track(self, part: Part) -> Dict[str, float]:
        """Weighted forward-only loss over this shard; returns GLOBAL metrics."""
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
        """Driver-callable: arrange → prepare → run updates → on_rollout_end."""
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
        from unirl.utils.profiling import profile_mode

        profiler = self._train_step_profiler() if profile_mode() == "train" else None
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
        """Freeze this part's anchor in eval mode, then return the model to train mode."""
        self.fsdp_backend.model.eval()
        self.prepare_segment(part, plans=plans)
        part = self.algorithm.prepare_part(part)
        self.fsdp_backend.model.train()
        return part

    def _run_window(self, arranged: List[Tuple[Part, Plan]], *, training_progress: float) -> TrainStepResult:
        """One optimizer step over an accumulation window of single-update parts."""
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
        """Run ``num_updates_per_batch`` optimizer steps over disjoint updates."""
        from unirl.utils.profiling import maybe_profile_update, profile_mode

        scope_update = profile_mode() == "one-update"
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
            {
                **r.metrics,
                "loss": r.loss,
                "grad_norm": r.grad_norm,
                "lr": r.lr,
                "optimizer_updates": r.optimizer_updates,
            }
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
