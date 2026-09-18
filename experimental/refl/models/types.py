"""Recipe-local BPTT stage contract for the refl recipe."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DiffuseWithGradResult:
    """Output of a stage's ``diffuse_with_grad``; ``kl_loss`` is per-sample ``[B]``, zeros when KL is off."""

    z_final: torch.Tensor
    kl_loss: torch.Tensor


__all__ = ["DiffuseWithGradResult"]
