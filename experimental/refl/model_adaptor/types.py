"""Recipe-local BPTT stage contract for the refl recipe.

``diffuse_with_grad`` is deliberately NOT part of the core
:class:`~unirl.models.types.diffusion.DiffusionStage` protocol: concrete
stages inherit that Protocol *explicitly*, so a protocol-level stub would
become a real ``None``-returning method on every diffusion stage in the
repo and make ``hasattr``-based capability checks meaningless. While REFL
is the only BPTT consumer, the contract lives here; if a second consumer
appears outside ``experimental/refl``, promote it to core as a separate opt-in
``@runtime_checkable`` protocol (the ``DifferentiableReward`` /
``LatentShapeProvider`` idiom), not as a method on ``DiffusionStage``.

Implementors (``Wan21ReflDiffusionStage`` / ``Wan22ReflDiffusionStage``)
provide::

    diffuse_with_grad(conditions, *, schedule, params, initial_latents=None)
        -> DiffuseWithGradResult
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DiffuseWithGradResult:
    """Output of a recipe stage's ``diffuse_with_grad``.

    ``kl_loss`` is per-sample ``[B]`` (zeros when the KL branch is off) so
    DP-scattered consumers round-trip each shard's own KL, never a
    cross-shard aggregate.
    """

    z_final: torch.Tensor
    kl_loss: torch.Tensor


__all__ = ["DiffuseWithGradResult"]
