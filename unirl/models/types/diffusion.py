"""Diffusion-specific interfaces for rollout and post-training."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Protocol, Tuple, TypeVar, runtime_checkable

import torch

from unirl.models.types.replay_result import ReplayResult
from unirl.types.segments import LatentSegment

if TYPE_CHECKING:
    from unirl.sde.kernels import StepStrategy


B = TypeVar("B")
C = TypeVar("C")


class RolloutPredictionCache:
    """Copy each denoiser output into step-major host storage and expose it as [B, T, *latent]."""

    def __init__(self, num_steps: int) -> None:
        if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 1:
            raise ValueError(f"RolloutPredictionCache.num_steps must be a positive int; got {num_steps!r}.")
        self.num_steps = num_steps
        self.buffer: Optional[torch.Tensor] = None
        self.ready: Optional[torch.cuda.Event] = None
        self._recorded: set[int] = set()

    def store(self, step: int, prediction: torch.Tensor) -> None:
        """Queue a contiguous step copy on the rollout stream at the prediction's native precision."""
        if not 0 <= step < self.num_steps or step in self._recorded:
            raise ValueError(f"RolloutPredictionCache expected each step in 0..{self.num_steps - 1} once; got {step}.")
        if self.buffer is None:
            self.buffer = torch.empty(
                (self.num_steps, *prediction.shape), device="cpu", dtype=prediction.dtype, pin_memory=prediction.is_cuda
            )
        elif prediction.shape != self.buffer.shape[1:] or prediction.dtype != self.buffer.dtype:
            raise ValueError("RolloutPredictionCache received inconsistent step prediction shape or dtype.")
        self.buffer[step].copy_(prediction.detach(), non_blocking=prediction.is_cuda)
        self._recorded.add(step)
        if prediction.is_cuda:
            self.ready = torch.cuda.current_stream(prediction.device).record_event()

    def finish(self) -> torch.Tensor:
        """Wait for the queued copies before the segment is sliced or transported."""
        if self.buffer is None or len(self._recorded) != self.num_steps:
            raise RuntimeError(
                f"RolloutPredictionCache recorded {len(self._recorded)} of {self.num_steps} rollout predictions."
            )
        if self.ready is not None:
            self.ready.synchronize()
        return self.buffer.transpose(0, 1)


@runtime_checkable
class DiffusionStep(Protocol[B, C]):
    """A single diffusion transition (per-step math kernel)."""

    def forward(
        self,
        *,
        strategy: "StepStrategy",
        noise_pred: torch.Tensor,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        prev_sample: Optional[torch.Tensor] = None,
        sigma_max: float = 0.99,
        eta: float = 1.0,
        step_index: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]: ...

    def step(
        self,
        model: B,
        conditions: C,
        *,
        strategy: "StepStrategy",
        sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        guidance_scale: float,
        prev_sample: Optional[torch.Tensor] = None,
        sigma_max: float = 0.99,
        eta: float = 1.0,
        step_index: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]: ...

    def step_with_logp(
        self,
        model: B,
        conditions: C,
        *,
        strategy: "StepStrategy",
        sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        guidance_scale: float,
        prev_sample: Optional[torch.Tensor] = None,
        sigma_max: float = 0.99,
        eta: float = 1.0,
        step_index: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]: ...


@runtime_checkable
class DiffusionStage(Protocol[C]):
    """Rollout-level diffusion stage: ``C → LatentSegment``."""

    def diffuse(
        self,
        conditions: C,
        *,
        schedule: torch.Tensor,
        params: object,
    ) -> LatentSegment: ...

    def replay(
        self,
        conditions: C,
        *,
        segment: LatentSegment,
        params: object,
        step_indices: Optional[List[int]] = None,
    ) -> ReplayResult: ...

    def predict_noise_at_step(
        self,
        conditions: C,
        *,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        params: object,
    ) -> torch.Tensor:
        """Single ``(xt, sigma)`` model forward — no scheduler iteration."""
        ...

    def nft_clean_latents(self, segment: LatentSegment) -> torch.Tensor:
        """Clean ``x0`` for forward-process training (defaults to final trajectory latent)."""
        return segment.latents[:, -1]


def combine_modality_logp(
    video_logp: torch.Tensor,
    audio_logp: torch.Tensor,
    *,
    n_video: int,
    n_audio: int,
) -> torch.Tensor:
    """Element-weighted mean of the per-step video/audio log-probs."""
    total = n_video + n_audio
    return (video_logp * n_video + audio_logp * n_audio) / total


__all__ = ["DiffusionStage", "DiffusionStep", "ReplayResult", "RolloutPredictionCache", "combine_modality_logp"]
