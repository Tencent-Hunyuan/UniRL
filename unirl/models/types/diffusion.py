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


__all__ = ["DiffusionStage", "DiffusionStep", "ReplayResult", "combine_modality_logp"]
