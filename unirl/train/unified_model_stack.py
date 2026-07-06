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
from dataclasses import replace
from typing import Dict, List, Mapping, Tuple

import torch

from unirl.algorithms import AlgorithmStepResult, StageAlgorithm
from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.distributed.group.remote import Remote
from unirl.train.backend.fsdp import FSDPBackend
from unirl.train.stack import TrainStepResult, _build_micro_batch_slices
from unirl.train.stack.base import _aggregate_update_results
from unirl.train.stack.planner.types import _positive_int, _update_ranges
from unirl.types.rollout_resp import RolloutTrack
from unirl.utils.misc import aggregate_numeric_metrics

logger = logging.getLogger(__name__)


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

        single_micro = len(micro_slices) == 1 and micro_slices[0] == (0, bs)
        for start, end in micro_slices:
            micro_track = resp_track if single_micro else resp_track.slice(start, end)
            result = algorithm.compute_loss_and_backward(
                conditions=micro_track.conditions,
                segment=micro_track.segment,
                advantages=micro_track.advantages,
                training_progress=training_progress,
                loss_scale=loss_scale,
            )
            micros.append(result)
            total_loss += result.loss
            has_backward = has_backward or result.has_backward

        aggregated: Mapping[str, object] = aggregate_numeric_metrics([r.metrics for r in micros if r.metrics])
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
    ) -> Dict[str, TrainStepResult]:
        """One optimizer step: zero_grad → backward BOTH tracks over their mini-batch
        slices → shared optimizer_step → stamp grad_norm / lr onto each track's result.
        """
        self.fsdp_backend.zero_grad()
        results: Dict[str, TrainStepResult] = {}
        any_backward = False
        for name in self.algorithms:
            partial, has_backward = self._backward_track(
                name, tracks[name], slices_by_track[name], training_progress=training_progress
            )
            results[name] = partial
            any_backward = any_backward or has_backward

        if any_backward:
            # Multi-update only: the prior update's forward/backward churn fragments the
            # CUDA pool, so this step's clip_grad_norm NCCL all_reduce can fail to find a
            # contiguous buffer (OOM with free-but-fragmented memory — exactly the
            # num_updates_per_batch>1 optimizer-step OOM). Returning the freed activation
            # blocks to the driver first defragments. Gated on >1 so the single-update
            # path (and the LoRA recipe) pays nothing.
            if self.num_updates_per_batch > 1 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            grad_norm = float(self.fsdp_backend.optimizer_step(max_grad_norm=float(self.max_grad_norm)))
        else:
            grad_norm = 0.0
            logger.warning("UnifiedModelTrainStack._train_one_step: no algorithm reported backward; skipping step.")

        lr = self._current_lr()
        for name, r in list(results.items()):
            results[name] = TrainStepResult(
                loss=r.loss, grad_norm=grad_norm, lr=lr, has_backward=r.has_backward, micros=r.micros, metrics=r.metrics
            )
        return results

    def on_rollout_end(self) -> None:
        """Per-rollout-boundary hook — delegates to the FSDPBackend's EMA."""
        self.fsdp_backend.on_rollout_end()

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def train_track(
        self,
        ar_track: RolloutTrack,
        image_track: RolloutTrack,
        *,
        training_progress: float,
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

        tracks = {"ar": ar_track, "image": image_track}
        # Freeze each track's π_old anchor once, before the multi-update loop.
        for name in self.algorithms:
            self.prepare_segment(name, tracks[name])

        # N optimizer steps over disjoint mini-batches (each track sliced by the same
        # shared _optimizer_step_slices; M=1 keeps ar/image 1:1 and equally sized).
        steps_by_track = {name: self._optimizer_step_slices(int(tracks[name].batch_size)) for name in self.algorithms}
        per_update: List[Dict[str, TrainStepResult]] = []
        for u in range(self.num_updates_per_batch):
            slices_by_track = {name: steps_by_track[name][u] for name in self.algorithms}
            per_update.append(self._train_one_step(tracks, slices_by_track, training_progress=float(training_progress)))

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
