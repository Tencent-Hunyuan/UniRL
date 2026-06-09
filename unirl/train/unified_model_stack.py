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

    backend.zero_grad()
    for name in ("ar", "image"):
        for (start, end) in micro_slices(track.batch_size):
            algorithm[name].compute_loss_and_backward(loss_scale=1/N, ...)   # grads accumulate
    if has_backward:
        grad_norm = backend.optimizer_step(max_grad_norm=...)                 # ONE step
    return {name: TrainStepResult, ...}

This is the multi-stage train stack — several stage algorithms share one
optimizer step, in contrast to the single-stage ``TrainStack``.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Mapping

from unirl.algorithms import AlgorithmStepResult, StageAlgorithm
from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.distributed.group.remote import Remote
from unirl.train.backend.fsdp import FSDPBackend
from unirl.train.stack import TrainStepResult, _build_micro_batch_slices
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

    def prepare_segment(self, name: str, resp_track: RolloutTrack) -> None:
        """Pre-step hook for one algorithm — no-op if its ``segment`` is None
        or the algorithm has no ``prepare_segment`` (GRPO doesn't)."""
        if resp_track.segment is None:
            return
        algorithm = self.algorithms[name]
        prepare = getattr(algorithm, "prepare_segment", None)
        if prepare is None:
            return
        prepare(conditions=resp_track.conditions, segment=resp_track.segment)

    def _train_one(
        self,
        name: str,
        resp_track: RolloutTrack,
        *,
        training_progress: float,
    ) -> tuple[TrainStepResult, bool]:
        """Backward one algorithm's track (no zero_grad / no optimizer step).

        Returns ``(per_algorithm_result, has_backward)``. ``zero_grad`` and the
        shared ``optimizer_step`` are owned by :meth:`train` so both algorithms
        accumulate into one step.
        """
        if resp_track.advantages is None:
            raise ValueError(
                f"UnifiedModelTrainStack.train: track {name!r} has advantages=None; "
                "upstream advantage pipeline must populate it before training."
            )

        bs = int(resp_track.batch_size)
        micro_slices = _build_micro_batch_slices(total_size=bs, micro_batch_size=int(self.micro_batch_size))
        if not micro_slices:
            raise ValueError(f"UnifiedModelTrainStack.train: empty batch for track {name!r} (batch_size={bs}).")

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
        # grad_norm / lr are filled by ``train`` after the shared optimizer step.
        partial = TrainStepResult(
            loss=total_loss,
            grad_norm=0.0,
            lr=0.0,
            has_backward=has_backward,
            micros=micros,
            metrics=aggregated,
        )
        return partial, has_backward

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
        both). ``prepare_segment`` (image only) populates ``segment.sde_logp``;
        the two ``compute_loss_and_backward`` calls accumulate gradients into the
        shared backbone's single LoRA adapter; one ``optimizer_step`` applies
        them; per-shard loss/grad_norm/metrics merge back via ``pytree_merge``.
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
        for name in self.algorithms:
            self.prepare_segment(name, tracks[name])

        self.fsdp_backend.zero_grad()

        results: Dict[str, TrainStepResult] = {}
        any_backward = False
        for name in self.algorithms:
            partial, has_backward = self._train_one(name, tracks[name], training_progress=float(training_progress))
            results[name] = partial
            any_backward = any_backward or has_backward

        if any_backward:
            grad_norm = float(self.fsdp_backend.optimizer_step(max_grad_norm=float(self.max_grad_norm)))
        else:
            grad_norm = 0.0
            logger.warning(
                "UnifiedModelTrainStack.train_track: no algorithm reported backward; skipping optimizer step."
            )

        self.on_rollout_end()

        # Stamp the shared grad_norm / lr onto every algorithm's result so the
        # log line reads naturally per track (TrainStepResult is frozen → rebuild).
        lr = self._current_lr()
        for name, r in list(results.items()):
            results[name] = TrainStepResult(
                loss=r.loss,
                grad_norm=grad_norm,
                lr=lr,
                has_backward=r.has_backward,
                micros=r.micros,
                metrics=r.metrics,
            )
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
