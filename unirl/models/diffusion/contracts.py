"""Domain-level diffusion stage contracts for rollout and post-training.

``DiffusionStage[C]`` describes the framework-facing stage surface without
prescribing a denoising-loop implementation. Dense single-stream, packed, and
joint multimodal stages may all implement this contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Protocol, TypeVar, runtime_checkable

import torch

from unirl.types.segments import LatentSegment

if TYPE_CHECKING:
    from unirl.models.types.replay_result import ReplayResult


C = TypeVar("C")


@runtime_checkable
class DiffusionStage(Protocol[C]):
    """Rollout-level diffusion stage: ``C → LatentSegment``.

    The schedule and sampling parameters are passed at call time rather than
    stored on the protocol. The implementation may use the shared
    single-stream loop, packed geometry, or a coupled multimodal transition.

    The conditions type ``C`` is per-bundle: SD3 declares
    ``SD3Conditions(Batch)`` with ``text: TextEmbedCondition``; FLUX
    would declare its own; etc.

    ``replay`` recomputes log-probs for the SDE transitions stored in a
    prior rollout's ``LatentSegment``, plus the per-step Gaussian mean
    μ_θ used by KL penalties. Returns a :class:`ReplayResult` with
    ``log_probs`` shape ``[B, S']`` aligned with ``segment.sde_logp`` (or
    a slice of it when ``step_indices`` selects a subset) and
    ``prev_sample_means`` shape ``[B, S', *latent_shape]``. Used by
    GRPO/DiffusionNFT-style replay during training.
    """

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
        """Single ``(xt, sigma)`` model forward — no scheduler iteration.

        Returns the raw noise prediction at an arbitrary ``(sample, sigma)``
        pair. Forward-process algorithms (DiffusionNFT et al.) build ``xt`` via the
        flow-matching forward diffusion ``xt = (1 - t) * x0 + t * noise``
        and call this to obtain the model's prediction without traversing
        an SDE trajectory. CFG batching + guidance scale handling are the
        same as ``diffuse`` / ``replay`` (delegated to the same kernel).
        """
        ...


__all__ = ["DiffusionStage"]
