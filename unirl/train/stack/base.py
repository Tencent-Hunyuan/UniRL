"""Shared train-stack base class, step result type, and per-update aggregation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
from typing import TYPE_CHECKING, List, Mapping, Sequence, Tuple

from unirl.algorithms.base import AlgorithmStepResult
from unirl.distributed.group.remote import Remote
from unirl.utils.metrics import aggregate_numeric_metrics

if TYPE_CHECKING:
    from unirl.algorithms.base import StageAlgorithm
    from unirl.train.backend.fsdp import FSDPBackend
    from unirl.types.sample import Part


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


def aggregate_update_results(
    results: List["TrainStepResult"], *, include_per_update: bool = False
) -> "TrainStepResult":
    """Collapse one rollout's per-update results into a single summary."""
    if len(results) == 1:
        return results[0]
    n = len(results)
    micros: List[AlgorithmStepResult] = [m for r in results for m in r.micros]
    metrics = aggregate_numeric_metrics([r.metrics for r in results if r.metrics])
    aggregated = TrainStepResult(
        loss=sum(r.loss for r in results) / n,
        grad_norm=sum(r.grad_norm for r in results) / n,
        lr=results[-1].lr,
        has_backward=any(r.has_backward for r in results),
        micros=micros,
        metrics=metrics,
        optimizer_updates=sum(r.optimizer_updates for r in results),
    )
    if include_per_update:
        aggregated = replace(
            aggregated,
            per_update=tuple(
                {
                    **r.metrics,
                    "loss": r.loss,
                    "grad_norm": r.grad_norm,
                    "lr": r.lr,
                    "optimizer_updates": r.optimizer_updates,
                }
                for r in results
            ),
        )
    return aggregated


class BaseTrainStack(Remote):
    """Constructor validation and backend lifecycle shared by stacks with independent training loops."""

    def __init__(
        self,
        *,
        fsdp_backend: FSDPBackend,
        algorithms: Sequence[StageAlgorithm],
        micro_batch_size: int,
        max_grad_norm: float,
        num_updates_per_batch: int,
    ) -> None:
        super().__init__()
        owner = type(self).__name__
        for name, value in (("micro_batch_size", micro_batch_size), ("num_updates_per_batch", num_updates_per_batch)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{owner}.{name} must be an int >= 1; got {value!r}.")
        if isinstance(max_grad_norm, bool) or not isinstance(max_grad_norm, (int, float)) or not max_grad_norm > 0.0:
            raise ValueError(f"{owner}.max_grad_norm must be a number > 0; got {max_grad_norm!r}.")
        if num_updates_per_batch > 1:
            for algorithm in algorithms:
                if not algorithm.supports_multi_update:
                    raise ValueError(
                        f"num_updates_per_batch={num_updates_per_batch} requires the old-logp anchor to stay "
                        f"frozen across the N optimizer steps, but {type(algorithm).__name__} sets "
                        "supports_multi_update=False. Set num_updates_per_batch=1."
                    )
        self.fsdp_backend = fsdp_backend
        self.micro_batch_size = micro_batch_size
        self.max_grad_norm = max_grad_norm
        self.num_updates_per_batch = num_updates_per_batch

    def on_rollout_end(self) -> None:
        """Delegate the rollout boundary to the backend's EMA."""
        self.fsdp_backend.on_rollout_end()

    def _align_track_inputs(self, part: Part) -> None:
        """Move the segment, conditions, and advantages to the training device; media payloads stay on the host."""
        device = self.fsdp_backend._device
        if part.segment is not None:
            part.segment = part.segment.to_device(device)
        part.conditions = {k: None if v is None else v.to_device(device) for k, v in part.conditions.items()}
        if part.advantages is not None:
            part.advantages = part.advantages.to(device=device)

    @cached_property
    def _train_step_profiler(self):
        """Per-worker train-step profiler, or ``None`` unless ``UNIRL_PROFILE`` is set."""
        from unirl.utils.profiling import maybe_build_train_profiler

        return maybe_build_train_profiler(self.fsdp_backend._rank)

    def _current_lr(self) -> float:
        """Learning rate of the optimizer's first param group."""
        return self.fsdp_backend.optimizer.param_groups[0]["lr"]
