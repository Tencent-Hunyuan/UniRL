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
        for name in ("ar", "image"):
            for (start, end) in micro_slices(mini_batch_u):
                algorithm[name].compute_loss_and_backward(loss_scale=1/N, ...)  # grads accumulate
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
from contextlib import nullcontext
from dataclasses import replace
from typing import Dict, List, Mapping, Tuple

import torch

from unirl.algorithms import AlgorithmStepResult, StageAlgorithm
from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.distributed.group.remote import Remote
from unirl.train.backend.fsdp import FSDPBackend
from unirl.train.stack import TrainStepResult, _build_micro_batch_slices
from unirl.train.stack.base import _aggregate_update_results, _train_skips
from unirl.train.stack.planner.types import _positive_int, _update_ranges
from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams, DiffusionSamplingParams
from unirl.utils.misc import aggregate_numeric_metrics

logger = logging.getLogger(__name__)


def _lineage_skips(sample: Sample, **kwargs) -> list:
    """:func:`_train_skips` over a whole lineage — this stack is dispatched a Sample.

    Load-bearing here rather than merely defensive: unlike ``TrainStack``'s
    selective ``_align_track_to_model``, ``train_track`` below calls
    ``Part.to_device`` on each gen Part, which is recursive and would haul a
    surviving decoded payload onto the GPU.
    """
    return [_train_skips(part, **kwargs) for part in sample.parts]


class UnifiedModelTrainStack(Remote):
    """Single-backbone, multi-algorithm train stack.

    Holds one shared :class:`FSDPBackend` plus explicit AR and image
    :class:`StageAlgorithm` siblings. Each algorithm trains its own Part but
    backward-accumulates into the same
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
    ) -> None:
        super().__init__()
        if int(micro_batch_size) < 1:
            raise ValueError(f"UnifiedModelTrainStack.micro_batch_size must be >= 1; got {micro_batch_size}.")
        if float(max_grad_norm) <= 0.0:
            raise ValueError(f"UnifiedModelTrainStack.max_grad_norm must be > 0; got {max_grad_norm}.")
        self.fsdp_backend = fsdp_backend
        self.ar_algorithm = ar_algorithm
        self.image_algorithm = image_algorithm
        self.micro_batch_size = int(micro_batch_size)
        self.max_grad_norm = float(max_grad_norm)
        self.num_updates_per_batch = _positive_int(
            name="UnifiedModelTrainStack.num_updates_per_batch", value=num_updates_per_batch
        )
        if self.num_updates_per_batch > 1:
            for name, algo in (("ar", self.ar_algorithm), ("image", self.image_algorithm)):
                if not getattr(algo, "supports_multi_update", False):
                    raise ValueError(
                        f"num_updates_per_batch={self.num_updates_per_batch} requires every algorithm's "
                        f"π_old anchor to stay frozen across the N optimizer steps, but the {name!r} "
                        f"algorithm ({type(algo).__name__}) sets supports_multi_update=False. Set "
                        f"num_updates_per_batch=1."
                    )

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

    def prepare_segment(self, algorithm: StageAlgorithm, part: Part) -> None:
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
        if part.segment is None:
            return
        prepare = getattr(algorithm, "prepare_segment", None)
        if prepare is None:
            return
        recomputes = getattr(algorithm, "recomputes_anchor", None)
        if recomputes is None or not recomputes():
            prepare(conditions=part.conditions, segment=part.segment)
            return
        micro_slices = [sl for step in self._optimizer_step_slices(int(part.batch_size)) for sl in step]
        if len(micro_slices) == 1:
            prepare(conditions=part.conditions, segment=part.segment)
            return
        anchor_fields = getattr(algorithm, "anchor_fields", ())
        collected: Dict[str, List[torch.Tensor]] = {field: [] for field in anchor_fields}
        for start, end in micro_slices:
            micro = part.slice(start, end)
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
            setattr(part.segment, field, torch.cat(parts, dim=0))

    def _backward_part(
        self,
        algorithm: StageAlgorithm,
        part: Part,
        micro_slices: List[Tuple[int, int]],
        *,
        training_progress: float,
    ) -> tuple[TrainStepResult, bool]:
        """Backward one algorithm's Part over the given absolute ``micro_slices``
        (no zero_grad / no optimizer step).

        Returns ``(per_algorithm_result, has_backward)``. ``zero_grad`` and the shared
        ``optimizer_step`` are owned by :meth:`_train_one_step` so both algorithms
        accumulate into one step. ``micro_slices`` are absolute ranges into
        ``part`` for ONE optimizer step (one ``num_updates_per_batch``
        mini-batch), produced by :meth:`_optimizer_step_slices`.
        """
        if part.advantages is None:
            raise ValueError(
                f"UnifiedModelTrainStack.train: {type(algorithm).__name__} Part has advantages=None; "
                "upstream advantage pipeline must populate it before training."
            )
        if not micro_slices:
            raise ValueError(f"UnifiedModelTrainStack.train: empty micro_slices for {type(algorithm).__name__} Part.")

        bs = int(part.batch_size)
        update_total = sum(end - start for start, end in micro_slices)
        micros: List[AlgorithmStepResult] = []
        total_loss = 0.0
        has_backward = False

        single_micro = len(micro_slices) == 1 and micro_slices[0] == (0, bs)
        for start, end in micro_slices:
            micro_track = part if single_micro else part.slice(start, end)
            loss_scale = (end - start) / float(update_total)
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
        training_progress: float,
    ) -> Dict[str, TrainStepResult]:
        """One optimizer step: zero_grad → backward BOTH Parts over their mini-batch
        slices → shared optimizer_step → stamp grad_norm / lr onto each stage result.
        """
        self.fsdp_backend.zero_grad()
        ar_result, ar_backward = self._backward_part(
            self.ar_algorithm, ar_part, ar_slices, training_progress=training_progress
        )
        image_result, image_backward = self._backward_part(
            self.image_algorithm, image_part, image_slices, training_progress=training_progress
        )
        results: Dict[str, TrainStepResult] = {"ar": ar_result, "image": image_result}
        any_backward = ar_backward or image_backward

        if any_backward:
            # Defragment between multi-updates before NCCL gradient clipping.
            if self.num_updates_per_batch > 1 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            grad_norm = float(self.fsdp_backend.optimizer_step(max_grad_norm=float(self.max_grad_norm)))
        else:
            grad_norm = 0.0
            logger.warning("UnifiedModelTrainStack._train_one_step: no algorithm reported backward; skipping step.")

        lr = self._current_lr()
        optimizer_updates = 1 if any_backward and math.isfinite(grad_norm) else 0
        for name, r in list(results.items()):
            results[name] = TrainStepResult(
                loss=r.loss,
                grad_norm=grad_norm,
                lr=lr,
                has_backward=r.has_backward,
                micros=r.micros,
                metrics=r.metrics,
                optimizer_updates=optimizer_updates,
            )
        return results

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

    @distributed(dispatch_mode=Dispatch.DP_SCATTER, skips=_lineage_skips)
    def train_track(
        self,
        sample: Sample,
        *,
        training_progress: float,
    ) -> Dict[str, TrainStepResult]:
        """Driver-callable: prepare → backward(ar) + backward(image) → ONE step.

        Takes the whole ``[input, ar, image]`` lineage so DP_SCATTER shards it by
        whole prompt-tree (``Sample.chunk`` → tree-shard), landing both stages on
        this worker with a CONSISTENT prompt set. Passing the two Parts separately
        was wrong at dp>1: ``infer_batch_size`` takes the first arg (``ar``=P*N), so
        the P*N*M image Part trips ``pytree_chunk``'s batch-mismatch branch and gets
        REPLICATED to every rank. ``prepare_segment`` (image only) populates
        ``segment.sde_logp``; the two ``compute_loss_and_backward`` calls accumulate
        gradients into the shared backbone's single LoRA adapter; one
        ``optimizer_step`` applies them; per-shard results merge back on collect.
        ``prepare_segment`` freezes each Part's π_old anchor ONCE; then the
        shard is split into ``num_updates_per_batch`` disjoint mini-batches and one
        optimizer step runs per mini-batch (each: backward ar + image over its
        mini-batch → one shared step). The 2nd+ step is off-policy, so the clip /
        ratio trust region engages; ``num_updates_per_batch=1`` is the prior
        single-step behavior. Per-track results are reduced across the updates;
        per-shard results merge back via ``pytree_cat`` on collect.
        """
        ar_part = sample.gen_part(ARSamplingParams)
        image_part = sample.gen_part(DiffusionSamplingParams)
        device = self.fsdp_backend._device
        ar_part = ar_part.to_device(device)
        image_part = image_part.to_device(device)

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
            self.prepare_segment(self.ar_algorithm, ar_part)
            self.prepare_segment(self.image_algorithm, image_part)

            ar_steps = self._optimizer_step_slices(int(ar_part.batch_size))
            image_steps = self._optimizer_step_slices(int(image_part.batch_size))
            per_update: List[Dict[str, TrainStepResult]] = []
            for u in range(self.num_updates_per_batch):
                per_update.append(
                    self._train_one_step(
                        ar_part,
                        image_part,
                        ar_slices=ar_steps[u],
                        image_slices=image_steps[u],
                        training_progress=float(training_progress),
                    )
                )
        if profiler is not None:
            profiler.step()

        self.on_rollout_end()

        results: Dict[str, TrainStepResult] = {}
        for name in ("ar", "image"):
            updates = [upd[name] for upd in per_update]
            aggregated = _aggregate_update_results(updates)
            if len(updates) > 1:
                aggregated = replace(
                    aggregated,
                    per_update=tuple(
                        {**dict(r.metrics), "loss": float(r.loss), "grad_norm": float(r.grad_norm), "lr": float(r.lr)}
                        for r in updates
                    ),
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
