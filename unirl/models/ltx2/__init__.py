"""LTX-2 / LTX-2.3 model package for UniRL.

Supports:
- T2V: text-to-video (LTX-2 and LTX-2.3)
- I2V: image-to-video with first-frame conditioning
- T2AV: text-to-audio-video joint generation (LTX-2.3 only)

HuggingFace models:
- LTX-2: Lightricks/LTX-Video-2
- LTX-2.3: Lightricks/LTX-Video-2.3

Architecture: ~2.4B parameter video DiT with Gemma3 text encoding,
3D VAE (32x spatial, 8x temporal compression, 128 latent channels),
flow matching scheduler, and unified video+audio attention (LTX-2.3).
"""

from unirl.models.ltx2.bundle import LTX2Bundle
from unirl.models.ltx2.conditions import LTX2Conditions
from unirl.models.ltx2.config import LTX2PipelineConfig
from unirl.models.ltx2.diffusion import LTX2DiffusionStage, LTX2DiffusionStep
from unirl.models.ltx2.pipeline import LTX2Pipeline
from unirl.models.ltx2.schedule import LTX2SchedulePolicy, build_ltx2_schedule_policy
from unirl.models.ltx2.text_embed import LTX2TextEmbedStage
from unirl.models.ltx2.vae import LTX2AudioDecodeStage, LTX2VAEDecodeStage, LTX2VAEEncodeStage

__all__ = [
    "LTX2Bundle",
    "LTX2Conditions",
    "LTX2DiffusionStage",
    "LTX2DiffusionStep",
    "LTX2Pipeline",
    "LTX2PipelineConfig",
    "LTX2SchedulePolicy",
    "LTX2TextEmbedStage",
    "LTX2VAEDecodeStage",
    "LTX2VAEEncodeStage",
    "LTX2AudioDecodeStage",
    "build_ltx2_schedule_policy",
]
