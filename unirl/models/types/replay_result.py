"""Structured return type for trainable-stage replay.

A stage's ``replay`` recomputes log-probs (and, for diffusion, the per-step
``prev_sample_mean`` used by KL penalties) for the transitions stored in a
prior rollout's segment. The values come from the same kernel call that
sampling used, so callers can rely on them lining up with the rollout's
stored ``segment.sde_logp`` (or its slice when ``step_indices`` subsets).

Diffusion stages populate ``log_probs`` and ``prev_sample_means`` (the
mean of the SDE Gaussian — μ_θ — used as the second moment in the KL
penalty). AR stages currently return a plain ``Tensor`` (signature
divergence with diffusion is intentional for now); when AR replay grows
``logits``-based KL support, the ``logits`` field on this result will be
the canonical home.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ReplayResult:
    """Per-stage replay output. ``log_probs`` is always populated; the
    others are stage-specific and may be ``None``."""

    log_probs: torch.Tensor
    """Aligned with ``segment.sde_logp`` (or its slice when ``step_indices``
    subsets). Shape ``[B, S']`` for diffusion replay."""

    prev_sample_means: Optional[torch.Tensor] = None
    """The SDE transition's mean μ_θ at each replayed step. Shape
    ``[B, S', *latent_shape]`` for diffusion. Used by GRPO's KL penalty.
    ``None`` when the stage doesn't produce it."""

    logits: Optional[torch.Tensor] = None
    """Per-step token logits at each replayed position. Shape
    ``[B, S', V]`` for AR. Reserved for future full-categorical KL
    or entropy penalty support; not needed for Binary KL (which uses
    only per-token log-probs). Currently not populated."""


__all__ = ["ReplayResult"]
