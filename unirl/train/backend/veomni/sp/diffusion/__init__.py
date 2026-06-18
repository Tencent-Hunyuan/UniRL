"""Ulysses SP for diffusers transformers (package).

Public API only. The shared machinery -- the two attention-SP mechanisms, the boundary-hook
helpers, and the dispatch/registry -- lives in :mod:`.ulysses`; per-model boundary adapters
live in :mod:`.models` and self-register via ``@register``. To add a model, see :mod:`.models`.
"""

from unirl.train.backend.veomni.sp.diffusion import models  # noqa: F401 -- registers per-model wrappers
from unirl.train.backend.veomni.sp.diffusion.ulysses import (
    FORWARD_WRAPPERS,
    apply_diffusion_sequence_parallelism,
    is_diffusers_transformer,
)

__all__ = ["apply_diffusion_sequence_parallelism", "is_diffusers_transformer", "FORWARD_WRAPPERS"]
