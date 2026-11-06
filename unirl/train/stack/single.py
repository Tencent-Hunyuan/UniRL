"""Family-agnostic single-stage train stack."""

from __future__ import annotations

import logging
import math
from contextlib import nullcontext
from dataclasses import replace
from typing import Dict, List, Mapping, Optional, Tuple, Union

import torch

from unirl.algorithms.base import AlgorithmStepResult, StageAlgorithm
from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.train.backend.fsdp import FSDPBackend
from unirl.train.stack.anchor import prepare_segment_anchors, validate_anchor_contract
from unirl.train.stack.base import BaseTrainStack, TrainStepResult, aggregate_update_results
from unirl.train.stack.loss_scale import resolve_loss_scales
from unirl.train.stack.planner import (
    Arrangement,
    CountPlanner,
    MicroPlanner,
    Plan,
    UpdatePlan,
    UpdatePlanner,
    arranged_slice,
)
from unirl.types.sample import Part
from unirl.utils.metrics import aggregate_numeric_metrics

logger = logging.getLogger(__name__)


class TrainStack(BaseTrainStack):
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
        shuffle_updates: bool = False,
        shuffle_seed: int = 0,
        first_update_ratio_tol: Optional[float] = None,
    ) -> None:
        super().__init__(
            fsdp_backend=fsdp_backend,
            algorithms=(algorithm,),
            micro_batch_size=micro_batch_size,
            max_grad_norm=max_grad_norm,
            num_updates_per_batch=num_updates_per_batch,
        )
        if first_update_ratio_tol is not None and not (
            math.isfinite(first_update_ratio_tol) and first_update_ratio_tol > 0.0
        ):
            raise ValueError(
                f"{type(self).__name__}.first_update_ratio_tol must be finite and > 0 when set; "
                f"got {first_update_ratio_tol}."
            )
        self.algorithm = algorithm
        self.first_update_ratio_tol = first_update_ratio_tol
        self.update_planner = UpdatePlanner(
            micro_planner if micro_planner is not None else CountPlanner(),
            shuffle_updates=shuffle_updates,
            shuffle_seed=shuffle_seed,
        )
        validate_anchor_contract(algorithm)

    def _run_update(
        self,
        part: Part,
        *,
        micros: UpdatePlan,
        order: Optional[torch.Tensor],
        training_progress: float,
        zero_grad: bool = True,
        do_optimizer_step: bool = True,
        loss_weight: float = 1.0,
        prior_backward: bool = False,
    ) -> TrainStepResult:
        """Run the micro ranges of a single update; step unless mid-window."""
        if part.advantages is None and self.algorithm.requires_advantages:
            raise ValueError(
                f"{type(self).__name__}._run_update: part.advantages is None; "
                "upstream advantage pipeline must populate it before training. "
                "(Supervised algorithms opt out by declaring requires_advantages=False.)"
            )
        if not micros:
            raise ValueError(f"{type(self).__name__}._run_update: empty micros.")

        bs = part.batch_size
        if zero_grad:
            self.fsdp_backend.zero_grad()

        cls = type(self).__name__
        loss_scales, global_weight = resolve_loss_scales(
            self.algorithm,
            part,
            micros=micros,
            order=order,
            backend=self.fsdp_backend,
            rank_info=self.rank_info,
            owner=cls,
        )
        if global_weight is not None and global_weight <= 0.0:
            raise ValueError(
                f"{cls}: zero valid tokens in this optimizer step "
                "(fully-masked batch?) — the data source must not emit steps with no "
                "supervision (0/0 loss NaNs destroyed checkpoints in verl #785)."
            )
        micro_results: List[AlgorithmStepResult] = []
        total_loss = 0.0
        has_backward = False

        single_micro = len(micros) == 1 and micros[0] == (0, bs)
        last_micro = len(micros) - 1
        for i, (start, end) in enumerate(micros):
            # Defer gradient reduce-scatter until the stepping rollout's final microbatch.
            self.fsdp_backend.set_grad_sync(do_optimizer_step and i == last_micro)
            micro_part = part if single_micro else arranged_slice(part, order, start, end)
            result = self.algorithm.compute_loss_and_backward(
                conditions=micro_part.conditions,
                segment=micro_part.segment,
                advantages=micro_part.advantages,
                training_progress=training_progress,
                # loss_weight (1/M) scales only the backward; result.loss stays the raw micro mean.
                loss_scale=loss_scales[i] * loss_weight,
            )
            micro_results.append(result)
            total_loss += result.loss * loss_scales[i]
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
            grad_norm = self.fsdp_backend.optimizer_step(max_grad_norm=self.max_grad_norm)
        else:
            grad_norm = 0.0
            logger.warning(
                "%s._run_update: no micro reported backward; skipping optimizer step.",
                type(self).__name__,
            )
        if torch.cuda.is_available():
            # CUDA memory footprint per optimizer step (leak diagnosis: tp2 path showed progressive OOM).
            aggregated_metrics = {
                **aggregated_metrics,
                "cuda_alloc_gb": torch.cuda.memory_allocated() / 2**30,
                "cuda_reserved_gb": torch.cuda.memory_reserved() / 2**30,
            }

        (global_loss_sum,) = self._all_reduce_sums([total_loss])
        total_loss = global_loss_sum / self.fsdp_backend.gradient_average_world_size()
        if global_weight is not None:
            aggregated_metrics = {**aggregated_metrics, "global_loss_weight": global_weight}

        return TrainStepResult(
            loss=total_loss,
            grad_norm=grad_norm,
            lr=self._current_lr(),
            has_backward=has_backward,
            micros=micro_results,
            metrics=aggregated_metrics,
            optimizer_updates=1 if has_backward and math.isfinite(grad_norm) else 0,
        )

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
                        sample_ids=micro.sample_ids or None,
                    )
                    loss_sum += s
                    weight_sum += w
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
        rollout_id: Optional[int] = None,
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
        for part in window:
            self._align_track_inputs(part)
        arranged = self.update_planner.arrange_many(
            window,
            num_updates=self.num_updates_per_batch,
            micro_batch_size=self.micro_batch_size,
            shuffle_step=rollout_id,
        )
        from unirl.utils.profiling import profile_mode

        profiler = self._train_step_profiler if profile_mode() == "train" else None
        with profiler.record("train_track") if profiler is not None else nullcontext():
            if len(arranged) == 1:
                part, plans, order = arranged[0]
                part = self._prepare_for_training(part, plans=plans, order=order)
                result = self._run_updates(part, plans=plans, order=order, training_progress=training_progress)
            else:
                result = self._run_window(arranged, training_progress=training_progress)
        if profiler is not None:
            profiler.step()
        self.on_rollout_end()
        return result

    def _prepare_for_training(self, part: Part, *, plans: Plan, order: Optional[torch.Tensor]) -> Part:
        """Freeze this part's anchor in eval mode, then return the model to train mode."""
        self.fsdp_backend.model.eval()
        prepare_segment_anchors(self.algorithm, part, plans, order=order)
        part = self.algorithm.prepare_part(part)
        self.fsdp_backend.model.train()
        return part

    def _run_window(self, arranged: Tuple[Arrangement, ...], *, training_progress: float) -> TrainStepResult:
        """One optimizer step over an accumulation window of single-update parts."""
        m = len(arranged)
        self.fsdp_backend.zero_grad()
        results: List[TrainStepResult] = []
        prior_backward = False
        for w, (part, plans, order) in enumerate(arranged):
            part = self._prepare_for_training(part, plans=plans, order=order)
            (micros,) = plans  # window parts are single-update (validated in train_track)
            result = self._run_update(
                part,
                micros=micros,
                order=order,
                training_progress=training_progress,
                zero_grad=False,
                do_optimizer_step=(w == m - 1),
                loss_weight=1.0 / m,
                prior_backward=prior_backward,
            )
            prior_backward = prior_backward or result.has_backward
            results.append(result)
        # The window is one optimizer step: its grad_norm is the stepping call's.
        window = replace(aggregate_update_results(results), grad_norm=results[-1].grad_norm)
        return self._guard_first_update_ratio(window)

    def _guard_first_update_ratio(self, result: TrainStepResult) -> TrainStepResult:
        """Report the pre-step importance ratio's distance from 1 and warn past ``first_update_ratio_tol``."""
        if self.first_update_ratio_tol is None or not result.has_backward:
            return result
        worst = max(
            max(1.0 - m.metrics["ratio_min"], m.metrics["ratio_max"] - 1.0) for m in result.micros if m.has_backward
        )
        if not worst <= self.first_update_ratio_tol:
            logger.warning(
                "%s: importance ratio is %.4g away from 1 before any optimizer step (tolerance %.4g); "
                "the algorithm's pi_old anchor does not match the policy being trained.",
                type(self).__name__,
                worst,
                self.first_update_ratio_tol,
            )
        return replace(result, metrics={**result.metrics, "first_update_ratio_dev": worst})

    def _run_updates(
        self,
        part: Part,
        *,
        plans: Plan,
        order: Optional[torch.Tensor],
        training_progress: float,
    ) -> TrainStepResult:
        """Run ``num_updates_per_batch`` optimizer steps over disjoint updates."""
        from unirl.utils.profiling import maybe_profile_update, profile_mode

        scope_update = profile_mode() == "one-update"
        results = []
        for micros in plans:
            cm = maybe_profile_update(self, self.fsdp_backend._rank) if scope_update else nullcontext()
            with cm:
                results.append(self._run_update(part, micros=micros, order=order, training_progress=training_progress))
        results[0] = self._guard_first_update_ratio(results[0])
        return aggregate_update_results(results, include_per_update=True)
