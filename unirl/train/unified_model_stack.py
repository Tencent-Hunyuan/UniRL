"""Unified-backbone multi-algorithm train stack (HunyuanImage3)."""

from __future__ import annotations

import logging
import math
from contextlib import nullcontext
from typing import Dict, List, Mapping, Optional, Tuple

import torch

from unirl.algorithms.base import AlgorithmStepResult, StageAlgorithm
from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.train.backend.fsdp import FSDPBackend
from unirl.train.stack import (
    TrainStepResult,
    aggregate_update_results,
    prepare_segment_anchors,
    resolve_loss_scales,
    validate_anchor_contract,
)
from unirl.train.stack.common import BaseTrainStack
from unirl.train.stack.planner import (
    CountPlanner,
    UpdatePlanner,
    arranged_slice,
)
from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams, DiffusionSamplingParams
from unirl.utils.metrics import aggregate_numeric_metrics

logger = logging.getLogger(__name__)


class UnifiedModelTrainStack(BaseTrainStack):
    """Single-backbone, multi-algorithm train stack."""

    def __init__(
        self,
        *,
        fsdp_backend: FSDPBackend,
        ar_algorithm: StageAlgorithm,
        image_algorithm: StageAlgorithm,
        micro_batch_size: int,
        max_grad_norm: float,
        num_updates_per_batch: int = 1,
        shuffle_updates: bool = False,
        shuffle_seed: int = 0,
    ) -> None:
        super().__init__(
            fsdp_backend=fsdp_backend,
            algorithms=(ar_algorithm, image_algorithm),
            micro_batch_size=micro_batch_size,
            max_grad_norm=max_grad_norm,
            num_updates_per_batch=num_updates_per_batch,
        )
        self.ar_algorithm = ar_algorithm
        self.image_algorithm = image_algorithm
        validate_anchor_contract(self.ar_algorithm)
        validate_anchor_contract(self.image_algorithm)
        self.update_planner = UpdatePlanner(
            CountPlanner(),
            shuffle_updates=shuffle_updates,
            shuffle_seed=shuffle_seed,
        )

    def _backward_part(
        self,
        algorithm: StageAlgorithm,
        part: Part,
        micro_slices: List[Tuple[int, int]],
        *,
        order: Optional[torch.Tensor],
        training_progress: float,
    ) -> tuple[TrainStepResult, bool]:
        """Backward one algorithm's Part over absolute ``micro_slices``; loss scales: see ``readme.md`` Gotchas."""
        if part.advantages is None:
            raise ValueError(
                f"UnifiedModelTrainStack.train: {type(algorithm).__name__} Part has advantages=None; "
                "upstream advantage pipeline must populate it before training."
            )
        if not micro_slices:
            raise ValueError(f"UnifiedModelTrainStack.train: empty micro_slices for {type(algorithm).__name__} Part.")

        bs = part.batch_size
        loss_scales, global_weight = resolve_loss_scales(
            algorithm,
            part,
            micros=micro_slices,
            order=order,
            backend=self.fsdp_backend,
            rank_info=self.rank_info,
            owner="UnifiedModelTrainStack",
        )
        micros: List[AlgorithmStepResult] = []
        total_loss = 0.0
        has_backward = False

        single_micro = len(micro_slices) == 1 and micro_slices[0] == (0, bs)
        for (start, end), loss_scale in zip(micro_slices, loss_scales):
            micro_track = part if single_micro else arranged_slice(part, order, start, end)
            result = algorithm.compute_loss_and_backward(
                conditions=micro_track.conditions,
                segment=micro_track.segment,
                advantages=micro_track.advantages,
                training_progress=training_progress,
                loss_scale=loss_scale,
            )
            micros.append(result)
            total_loss += result.loss * loss_scale
            has_backward = has_backward or result.has_backward

        aggregated: Mapping[str, object] = aggregate_numeric_metrics([r.metrics for r in micros if r.metrics])
        if global_weight is not None:
            has_zero_valid_tokens = global_weight <= 0.0
            if has_zero_valid_tokens:
                logger.warning(
                    "UnifiedModelTrainStack: %s has zero valid tokens across DP in this optimizer step; "
                    "its loss_scale is 0 and only the other track contributes gradient.",
                    type(algorithm).__name__,
                )
            aggregated = {
                **aggregated,
                "global_loss_weight": global_weight,
                "has_zero_valid_tokens": float(has_zero_valid_tokens),
            }
        partial = TrainStepResult(
            loss=total_loss,
            grad_norm=0.0,
            lr=0.0,
            has_backward=has_backward,
            micros=micros,
            metrics=aggregated,
            optimizer_updates=0,
        )
        return partial, has_backward

    def _train_one_step(
        self,
        ar_part: Part,
        image_part: Part,
        *,
        ar_slices: List[Tuple[int, int]],
        image_slices: List[Tuple[int, int]],
        ar_order: Optional[torch.Tensor],
        image_order: Optional[torch.Tensor],
        training_progress: float,
    ) -> Dict[str, TrainStepResult]:
        """One optimizer step: zero_grad → backward BOTH Parts over their mini-batch"""
        self.fsdp_backend.zero_grad()
        ar_result, ar_backward = self._backward_part(
            self.ar_algorithm, ar_part, ar_slices, order=ar_order, training_progress=training_progress
        )
        image_result, image_backward = self._backward_part(
            self.image_algorithm, image_part, image_slices, order=image_order, training_progress=training_progress
        )
        results: Dict[str, TrainStepResult] = {"ar": ar_result, "image": image_result}
        any_backward = ar_backward or image_backward

        if any_backward:
            # Defragment between multi-updates before NCCL gradient clipping.
            if self.num_updates_per_batch > 1 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            grad_norm = self.fsdp_backend.optimizer_step(max_grad_norm=self.max_grad_norm)
        else:
            grad_norm = 0.0
            logger.warning("UnifiedModelTrainStack._train_one_step: no algorithm reported backward; skipping step.")

        lr = self._current_lr()
        optimizer_updates = 1 if any_backward and math.isfinite(grad_norm) else 0
        # Each track's local sum(loss * loss_scale) averages to its DP-global loss over the gradient world.
        loss_sums = self.fsdp_backend.all_reduce_loss_sums([r.loss for r in results.values()])
        dp_world = self.fsdp_backend.gradient_average_world_size()
        for (name, r), loss_sum in zip(results.items(), loss_sums):
            results[name] = TrainStepResult(
                loss=loss_sum / dp_world,
                grad_norm=grad_norm,
                lr=lr,
                has_backward=r.has_backward,
                micros=r.micros,
                metrics=r.metrics,
                optimizer_updates=optimizer_updates,
            )
        return results

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def train_track(
        self,
        sample: Sample,
        *,
        training_progress: float,
        rollout_id: int | None = None,
    ) -> Dict[str, TrainStepResult]:
        """Driver-callable: prepare → backward(ar) + backward(image) → ONE step."""
        ar_part = sample.gen_part(ARSamplingParams)
        image_part = sample.gen_part(DiffusionSamplingParams)
        device = self.fsdp_backend._device
        ar_part = ar_part.to_device(device)
        image_part = image_part.to_device(device)
        (_, ar_steps, ar_order), (_, image_steps, image_order) = self.update_planner.arrange_many(
            (ar_part, image_part),
            num_updates=self.num_updates_per_batch,
            micro_batch_size=self.micro_batch_size,
            shuffle_step=rollout_id,
        )

        from unirl.utils.profiling import profile_mode

        scope = profile_mode()
        if scope == "one-update" and not getattr(self, "_warned_one_update", False):
            self._warned_one_update = True
            logger.warning(
                "UNIRL_PROFILE=one-update is not supported on the unified-model stack "
                "(no _run_updates loop); use UNIRL_PROFILE=train. No trace produced."
            )
        profiler = self._train_step_profiler if scope == "train" else None
        with profiler.record("train_track") if profiler is not None else nullcontext():
            prepare_segment_anchors(self.ar_algorithm, ar_part, ar_steps, order=ar_order)
            prepare_segment_anchors(self.image_algorithm, image_part, image_steps, order=image_order)

            per_update: List[Dict[str, TrainStepResult]] = []
            for u in range(self.num_updates_per_batch):
                per_update.append(
                    self._train_one_step(
                        ar_part,
                        image_part,
                        ar_slices=ar_steps[u],
                        image_slices=image_steps[u],
                        ar_order=ar_order,
                        image_order=image_order,
                        training_progress=training_progress,
                    )
                )
        if profiler is not None:
            profiler.step()

        self.on_rollout_end()

        results: Dict[str, TrainStepResult] = {}
        for name in ("ar", "image"):
            updates = [upd[name] for upd in per_update]
            results[name] = aggregate_update_results(updates, include_per_update=True)
        return results


__all__ = ["UnifiedModelTrainStack"]
