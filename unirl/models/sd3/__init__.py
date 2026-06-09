"""SD3 pipeline — parallel prototype on the new four-tier architecture.

Re-expression of :class:`unirl.models.sd3.SD3ModelBundle` against the
typed ``Bundle`` / ``Pipeline`` / ``EmbedStage`` / ``DiffusionStage`` /
``DecodeStage`` protocols. Legacy SD3 keeps serving production GRPO/DiffusionNFT;
this package proves the new contracts end-to-end before the algorithm /
training-backend migration lands.

Importing this package re-exports its bundle / pipeline / config classes;
recipes wire them by ``_target_`` dotpath.
"""

from unirl.models.sd3.bundle import SD3Bundle
from unirl.models.sd3.conditions import SD3Conditions
from unirl.models.sd3.config import SD3PipelineConfig
from unirl.models.sd3.pipeline import SD3Pipeline

__all__ = [
    "SD3Bundle",
    "SD3Conditions",
    "SD3Pipeline",
    "SD3PipelineConfig",
]
