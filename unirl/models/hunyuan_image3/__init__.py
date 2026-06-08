"""HunyuanImage 3.0 pipeline — concrete user of the typed Stage protocols.

Re-expression of HunyuanImage3's t2t / i2t / t2i / it2i task topologies
against the typed ``Bundle`` / ``Pipeline`` / ``EmbedStage`` /
``EncodeStage`` / ``DecodeStage`` / ``ARStage`` / ``DiffusionStage``
protocols. Mirrors :mod:`unirl.models.sd3` 1:1 in layout.

PR 2 (current) ships the bundle skeleton + t2i diffusion path. PR 3 lands
the AR stage (t2t / i2t). PR 4 wires the full t2i with AR multi-pass.
PR 5 adds it2i.

Importing this package re-exports its bundle / pipeline / config classes;
recipes wire them by ``_target_`` dotpath.
"""

from unirl.models.hunyuan_image3.ar import (
    HunyuanImage3ARParams,
    HunyuanImage3ARStage,
    HunyuanImage3ARStep,
)
from unirl.models.hunyuan_image3.bundle import HunyuanImage3Bundle
from unirl.models.hunyuan_image3.conditions import (
    HunyuanImage3ARConditions,
    HunyuanImage3DiffusionConditions,
    HunyuanImage3FusedMultimodalCondition,
)
from unirl.models.hunyuan_image3.config import HunyuanImage3PipelineConfig
from unirl.models.hunyuan_image3.diffusion import (
    HunyuanImage3DiffusionStage,
    HunyuanImage3DiffusionStep,
)
from unirl.models.hunyuan_image3.pipeline import HunyuanImage3Pipeline
from unirl.models.hunyuan_image3.text_embed import (
    HunyuanImage3TextEmbedStage,
)
from unirl.models.hunyuan_image3.vae import (
    HunyuanImage3VAEDecodeStage,
    HunyuanImage3VAEEncodeStage,
)
from unirl.models.hunyuan_image3.vit_encode import (
    HunyuanImage3VitEncodeStage,
)

__all__ = [
    "HunyuanImage3ARConditions",
    "HunyuanImage3ARParams",
    "HunyuanImage3ARStage",
    "HunyuanImage3ARStep",
    "HunyuanImage3Bundle",
    "HunyuanImage3DiffusionConditions",
    "HunyuanImage3DiffusionStage",
    "HunyuanImage3DiffusionStep",
    "HunyuanImage3FusedMultimodalCondition",
    "HunyuanImage3Pipeline",
    "HunyuanImage3PipelineConfig",
    "HunyuanImage3TextEmbedStage",
    "HunyuanImage3VAEDecodeStage",
    "HunyuanImage3VAEEncodeStage",
    "HunyuanImage3VitEncodeStage",
]
