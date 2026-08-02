"""Compatibility exports for the canonical diffusion model contracts.

New code should import these contracts from :mod:`unirl.models.diffusion`.
This module preserves the original public path for existing integrations.
"""

from unirl.models.diffusion.contracts import DiffusionStage, DiffusionStep
from unirl.models.types.replay_result import ReplayResult

__all__ = ["DiffusionStage", "DiffusionStep", "ReplayResult"]
