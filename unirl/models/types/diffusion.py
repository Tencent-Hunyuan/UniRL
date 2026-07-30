"""Diffusion-specific interfaces for rollout and post-training.

Pipeline-level: ``DiffusionStage[C]`` — ``C → LatentSegment``, iterates a
``DiffusionStep`` over a sigma schedule. Parameterized on the conditions
container type ``C`` so concrete bundles can declare their own typed
container (e.g. ``SD3Conditions`` with ``text: TextEmbedCondition``) and get
typed access inside the stage.

Step-level kernel: ``DiffusionStep[B, C]`` — per-step transition that
takes the model bundle ``B`` and conditions ``C`` and runs both the
model forward and the SDE transition. The strategy is supplied per call
so the kernel itself is stateless.
"""

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
    """A single diffusion transition (per-step math kernel).

    The kernel is stateless: it takes a ``model`` bundle, ``conditions``
    container, and an SDE ``strategy`` per call, runs the model forward
    (CFG / noise prediction) internally, then applies the SDE transition.

    ``prev_sample=None`` means sampling mode; providing ``prev_sample``
    means log-prob replay mode for training.

    ``forward()`` is a lower-level escape hatch that takes a precomputed
    ``noise_pred``; it is useful for unit testing the SDE math without a
    model and for power users who want to share a noise prediction
    across multiple transitions.
    """

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
    """Rollout-level diffusion stage: ``C → LatentSegment``.

    A ``DiffusionStage`` holds a bundle + a ``DiffusionStep`` kernel + an
    SDE strategy and runs the loop. The schedule (sigma values) and any
    sampling params are passed into ``diffuse`` at call time, not held on
    the instance — so one stage can serve many configurations.

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


__all__ = ["DiffusionStage", "DiffusionStep", "ReplayResult"]
