"""HunyuanImage 3.0 pipeline — concrete user of the typed Stage protocols."""

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
    HunyuanImage3VAECondition,
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
    "HunyuanImage3VAECondition",
    "HunyuanImage3Pipeline",
    "HunyuanImage3PipelineConfig",
    "HunyuanImage3TextEmbedStage",
    "HunyuanImage3VAEDecodeStage",
    "HunyuanImage3VAEEncodeStage",
    "HunyuanImage3VitEncodeStage",
]
