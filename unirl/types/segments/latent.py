"""LatentSegment — SoA container for latent (image / video / audio) rollouts.

``latents`` shape is ``[N_segs, K, …]`` where K is the trajectory length
(stored denoising steps) and the trailing dims depend on modality. Modality
is set via factory helpers (``make_image_segment`` etc.) since one container
class serves all latent modalities.

SDE log probs are stored densely in ``sde_logp`` of shape ``[N_segs, S]``
where ``S = len(sde_indices)`` — one elementwise-mean log-prob per SDE
transition. ``sde_indices`` is a ``[S]`` int tensor naming the step index
each slot corresponds to. No NaN sentinels: every slot is a valid SDE
log-prob; non-SDE steps simply aren't represented. This mirrors the
``Trajectory.index_map`` pattern (``trajectory_store.py``) — replay code
reads ``sde_logp[:, s]`` and uses ``sde_indices[s]`` for the step lookup.

``sde_logp`` may be populated either at rollout time by a native log-prob
source (the rollout engines best-effort emit it — SGLang, vllm_omni) or by
the trainer via :meth:`StageAlgorithm.prepare_segment`. Which one is the
authoritative π_old anchor is a training-layer decision
(``algorithm.old_logp_source``): ``"rollout"`` keeps the engine's emission,
``"replay"`` recomputes and overwrites it. When the engine emits nothing it
leaves ``sde_logp = None`` and ``prepare_segment`` fills it (replay) or
raises (rollout). Both paths produce the same ``[N_segs, S]`` shape with
``S == len(sde_indices)``.
"""

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
    """Diffusion-style latent trajectory across a sigma schedule.

    ``modality`` is a per-instance ``shared_field`` (NOT a ``ClassVar``) so
    it survives :meth:`Batch.select` / :meth:`Batch.slice` /
    :meth:`Batch.clone` / :meth:`Batch.concat`. Each of those ops
    rebuilds the instance via ``type(self)(**kwargs)`` walking declared
    dataclass fields; a ``ClassVar`` modality wouldn't appear in the
    field set and the rebuilt instance would silently revert to the
    class default ``Modality.IMAGE``. That regression would break
    downstream modality-aware dispatch (e.g. `RolloutResp.split()` calls
    ``select`` per group; the resulting per-group segments must keep
    their video / audio modality). The ``shared_field`` declaration
    makes modality batch-shared metadata — every sample in a segment
    has the same modality (you don't mix image and video in one
    segment) — which is exactly the ``SHARED`` semantics.
    """

    modality: Modality = shared_field(default=Modality.IMAGE)

    latents: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)
    sigmas: Optional[torch.Tensor] = shared_field(default=None)
    indices: Optional[torch.Tensor] = shared_field(default=None)
    sde_logp: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)
    sde_means: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)
    sde_indices: Optional[torch.Tensor] = shared_field(default=None)
    log_probs: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)
    loss_mask: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)

    def as_condition(self) -> Optional[Condition]:
        """Promote the *final* step's latent into an ``ImageLatentCondition``.

        Only image-modality segments promote in this prototype; video/audio
        promotion paths are deferred to first consumer.
        """
        if self.modality is not Modality.IMAGE:
            return None
        if self.latents is None:
            return None
        return ImageLatentCondition(latents=self.latents[:, -1])

    def latents_at(self, step_idx: int) -> torch.Tensor:
        """Return ``latents`` at the given trajectory step.

        The segment stores latents at a sparse subset of ``[0, T+1)`` defined
        by ``indices``; this maps ``step_idx`` back to the storage slot ``k``
        where ``indices[k] == step_idx`` and returns ``latents[:, k]``. Raises
        ``KeyError`` if the step isn't stored.
        """
        if self.indices is None or self.latents is None:
            raise RuntimeError("LatentSegment.latents_at: missing indices or latents")
        matches = (self.indices == int(step_idx)).nonzero(as_tuple=False).flatten()
        if matches.numel() == 0:
            raise KeyError(
                f"LatentSegment.latents_at: step_idx={step_idx} not in stored indices={self.indices.tolist()}"
            )
        return self.latents[:, int(matches[0].item())]


def make_image_segment(**kwargs) -> LatentSegment:
    """Build a ``LatentSegment`` with ``modality=Modality.IMAGE``.

    Convenience wrapper over the dataclass ctor: pre-fix this factory
    used ``object.__setattr__`` to stamp a ``ClassVar`` modality as an
    instance attr, which was wiped on every ``Batch.select`` /
    ``clone`` rebuild. Now ``modality`` is a real ``shared_field``, so
    the dataclass ``__init__`` propagates it through all Batch ops
    without further help. ``make_*`` factories are still preferred at
    call sites for grep-ability ("which factory built this?") and
    forward compat (modality-specific construction logic can land
    inside the factory later).
    """
    return LatentSegment(modality=Modality.IMAGE, **kwargs)


def make_video_segment(**kwargs) -> LatentSegment:
    """Build a ``LatentSegment`` with ``modality=Modality.VIDEO``.

    See :func:`make_image_segment` for the rationale on the factory
    pattern after the ``ClassVar → shared_field`` migration.
    """
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
