"""PE (Prompt Enhancement) composed pipeline.

Bundles a diffusion :class:`Bundle` (e.g. :class:`SD3Bundle`) together
with an AR LLM :class:`Bundle` (e.g. :class:`Qwen3Bundle`) and runs the
two-phase flow "LLM rewrites prompt → diffusion samples image".

Importing this package re-exports its bundle / pipeline classes;
recipes wire them by ``_target_`` dotpath.
"""

from unirl.models.pe.bundle import PEBundle
from unirl.models.pe.pipeline import PEPipeline

__all__ = [
    "PEBundle",
    "PEPipeline",
]
