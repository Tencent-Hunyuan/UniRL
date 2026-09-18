"""Ulysses SP for diffusers transformers (package)."""

from unirl.train.backend.veomni.sp.diffusion import models  # noqa: F401 -- registers per-model wrappers
from unirl.train.backend.veomni.sp.diffusion.ulysses import (
    FORWARD_WRAPPERS,
    apply_diffusion_sequence_parallelism,
    is_diffusers_transformer,
)

__all__ = ["apply_diffusion_sequence_parallelism", "is_diffusers_transformer", "FORWARD_WRAPPERS"]
