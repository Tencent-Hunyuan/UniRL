"""Compatibility exports for legacy diffusion model contracts.

New code should import :class:`DiffusionStage` and the explicitly scoped
single-stream primitives from :mod:`unirl.models.diffusion`. This module keeps
the original ``DiffusionStep`` name for existing integrations.
"""

from unirl.models.diffusion.contracts import DiffusionStage
from unirl.models.diffusion.single_stream import SingleStreamDiffusionStep as DiffusionStep
from unirl.models.types.replay_result import ReplayResult

__all__ = ["DiffusionStage", "DiffusionStep", "ReplayResult"]
