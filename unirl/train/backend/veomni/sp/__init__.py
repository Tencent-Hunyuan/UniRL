"""Ulysses sequence-parallel (SP) patches for the VeOmni backend."""

from __future__ import annotations

import logging

from torch import nn

logger = logging.getLogger(__name__)


def apply_sequence_parallelism(model: nn.Module, sp_size: int) -> None:
    """Install the Ulysses SP patch on ``model`` in place (no-op if sp_size<=1)."""
    if sp_size <= 1:
        return

    from unirl.train.backend.veomni.sp import ar, diffusion

    if ar.is_ar_causal_lm(model):
        ar.apply_ar_sequence_parallelism(model, sp_size)
        return

    if diffusion.is_diffusers_transformer(model):
        diffusion.apply_diffusion_sequence_parallelism(model, sp_size)
        return

    raise NotImplementedError(
        f"apply_sequence_parallelism: no SP patcher for {type(model).__name__} "
        "(neither an HF causal-LM nor a diffusers transformer)."
    )


__all__ = ["apply_sequence_parallelism"]
