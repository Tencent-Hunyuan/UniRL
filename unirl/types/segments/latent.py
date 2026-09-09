"""LatentSegment — SoA latent rollouts; ``sde_logp[:, s]`` is the transition at step ``sde_indices[s]``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from unirl.distributed.tensor.batch import FieldKind, field, shared_field
from unirl.types.conditions.base import Condition, Modality
from unirl.types.conditions.image import ImageLatentCondition
from unirl.types.segments.base import Segment


@dataclass
class LatentSegment(Segment):
    """Diffusion-style latent trajectory across a sigma schedule."""

    modality: Modality = shared_field(default=Modality.IMAGE)

    latents: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)  # [N_segs, K, …] K = stored steps
    initial_latents: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)
    sigmas: Optional[torch.Tensor] = shared_field(default=None)  # [T+1] float — the full schedule
    indices: Optional[torch.Tensor] = shared_field(default=None)  # [K] long — step of each snapshot
    sde_logp: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)  # [N_segs, S], S = len(sde_indices)
    sde_means: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)  # [N_segs, S] + *latent_shape
    sde_indices: Optional[torch.Tensor] = shared_field(default=None)  # [S] long — step per sde_logp slot
    log_probs: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)
    loss_mask: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)
    aux_latents: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)

    def as_condition(self) -> Optional[Condition]:
        """Promote the *final* step's latent into an ``ImageLatentCondition``."""
        if self.modality is not Modality.IMAGE:
            return None
        if self.latents is None:
            return None
        return ImageLatentCondition(latents=self.latents[:, -1])

    def latents_at(self, step_idx: int) -> torch.Tensor:
        """Return ``latents`` at the given trajectory step."""
        if self.indices is None or self.latents is None:
            raise RuntimeError("LatentSegment.latents_at: missing indices or latents")
        matches = (self.indices == int(step_idx)).nonzero(as_tuple=False).flatten()
        if matches.numel() == 0:
            raise KeyError(
                f"LatentSegment.latents_at: step_idx={step_idx} not in stored indices={self.indices.tolist()}"
            )
        return self.latents[:, int(matches[0].item())]

    def aux_latents_at(self, step_idx: int) -> torch.Tensor:
        """``aux_latents`` at a trajectory step, via the same sparse ``indices`` map as :meth:`latents_at`."""
        if self.indices is None or self.aux_latents is None:
            raise RuntimeError("LatentSegment.aux_latents_at: missing indices or aux_latents")
        matches = (self.indices == int(step_idx)).nonzero(as_tuple=False).flatten()
        if matches.numel() == 0:
            raise KeyError(
                f"LatentSegment.aux_latents_at: step_idx={step_idx} not in stored indices={self.indices.tolist()}"
            )
        return self.aux_latents[:, int(matches[0].item())]


def make_image_segment(**kwargs) -> LatentSegment:
    """Build a ``LatentSegment`` with ``modality=Modality.IMAGE``."""
    return LatentSegment(modality=Modality.IMAGE, **kwargs)


def make_video_segment(**kwargs) -> LatentSegment:
    """Build a ``LatentSegment`` with ``modality=Modality.VIDEO``."""
    return LatentSegment(modality=Modality.VIDEO, **kwargs)


def make_audio_segment(**kwargs) -> LatentSegment:
    """Build a ``LatentSegment`` with ``modality=Modality.AUDIO``."""
    return LatentSegment(modality=Modality.AUDIO, **kwargs)


__all__ = [
    "LatentSegment",
    "make_audio_segment",
    "make_image_segment",
    "make_video_segment",
]
