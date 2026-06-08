"""HunyuanVideo-1.5 pipeline on the typed four-tier architecture.

Implements the typed ``Bundle`` / ``Pipeline`` / ``EmbedStage`` /
``DiffusionStage`` / ``DecodeStage`` protocols. Sibling of
:mod:`unirl.models.wan21` (text-to-video peer) and
:mod:`unirl.models.hunyuan_image3` (Hunyuan-family peer).

Importing this package re-exports its bundle / pipeline / config classes;
recipes wire them by ``_target_`` dotpath.

Scope (v1)
----------
- **Text-to-Video (T2V)**: full support. Dual text encoders
  (Qwen2.5-VL MLLM + ByT5 glyph). CFG with stacked dual-stream
  classifier-free guidance (standard ``uncond + scale * (cond - uncond)``).
- **Image-to-Video (I2V)**: deferred. The transformer's
  ``cond_latents`` / ``cond_mask`` packing slots are zero-filled in v1;
  when I2V lands it will add a ``vision`` stage producing both an
  ``image_embeds`` SigLIP condition AND an image-latent condition that
  participates in the channel-dim concat inside ``predict_noise``.
"""

from unirl.models.hunyuan_video15.bundle import HunyuanVideo15Bundle
from unirl.models.hunyuan_video15.conditions import HunyuanVideo15Conditions
from unirl.models.hunyuan_video15.config import HunyuanVideo15PipelineConfig
from unirl.models.hunyuan_video15.pipeline import HunyuanVideo15Pipeline

__all__ = [
    "HunyuanVideo15Bundle",
    "HunyuanVideo15Conditions",
    "HunyuanVideo15Pipeline",
    "HunyuanVideo15PipelineConfig",
]
