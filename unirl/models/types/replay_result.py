"""Structured return type for trainable-stage replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ReplayResult:
    """Per-stage replay output."""

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

    values: Optional[torch.Tensor] = None
    """Per-token critic predictions ``V_t``. Packed ``[total_tokens]`` for AR.
    ``None`` when replay did not request a value head."""


__all__ = ["ReplayResult"]
