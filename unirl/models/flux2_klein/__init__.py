"""FLUX.2-klein-9B pipeline on the new four-tier typed architecture.

Re-expression of ``main_flux_bundle/unirl/models/flux2.py``
(Klein branch) + ``main_flux_bundle/unirl/samplers/fsdp/flux2_sampler.py``
against the typed ``Bundle`` / ``Pipeline`` / ``EmbedStage`` /
``DiffusionStage`` / ``DecodeStage`` protocols. Sibling of
:mod:`unirl.models.sd3` and :mod:`unirl.models.qwen_image`.

Importing this package re-exports its bundle / pipeline / config classes;
recipes wire them by ``_target_`` dotpath.
"""

from unirl.models.flux2_klein.bundle import Flux2KleinBundle
from unirl.models.flux2_klein.conditions import Flux2KleinConditions
from unirl.models.flux2_klein.config import Flux2KleinPipelineConfig
from unirl.models.flux2_klein.diffusion import (
    Flux2KleinDiffusionParams,
    Flux2KleinDiffusionStage,
    Flux2KleinDiffusionStep,
)
from unirl.models.flux2_klein.pipeline import Flux2KleinPipeline
from unirl.models.flux2_klein.schedule import (
    Flux2KleinSchedulePolicy,
    build_flux2_klein_schedule_policy,
)
from unirl.models.flux2_klein.text_embed import Flux2KleinTextEmbedStage
from unirl.models.flux2_klein.vae import Flux2KleinVAEDecodeStage

__all__ = [
    "Flux2KleinBundle",
    "Flux2KleinConditions",
    "Flux2KleinDiffusionParams",
    "Flux2KleinDiffusionStage",
    "Flux2KleinDiffusionStep",
    "Flux2KleinPipeline",
    "Flux2KleinPipelineConfig",
    "Flux2KleinSchedulePolicy",
    "Flux2KleinTextEmbedStage",
    "Flux2KleinVAEDecodeStage",
    "build_flux2_klein_schedule_policy",
]
