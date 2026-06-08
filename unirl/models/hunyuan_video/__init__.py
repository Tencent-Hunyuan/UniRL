"""HunyuanVideo-1.0 pipeline on the four-tier typed architecture.

Implements the HunyuanVideo 1.0 (original) model using the typed
``Bundle`` / ``Pipeline`` / ``EmbedStage`` / ``DiffusionStage`` /
``DecodeStage`` protocols. Sibling of
:mod:`unirl.models.hunyuan_video15` (the 1.5 variant).

Key differences vs HunyuanVideo-1.5:
- Dual text encoders: LLaMA (3D: [B, seq, 4096]) + CLIP (2D pooled: [B, 768])
  (vs Qwen2.5-VL MLLM + ByT5 in 1.5)
- No channel-dim packing (in_channels=16, NOT 2*C+1)
- Guidance embedding (``guidance_embeds=True``) instead of CFG
- ``pooled_projections`` kwarg (CLIP pooled) instead of ``encoder_hidden_states_2``
- Different VAE: spatial_compression=8, temporal_compression=4, latent_channels=16
  (vs spatial=16, temporal=4, latent_channels=32 in 1.5)

Importing this package re-exports its bundle / pipeline / config classes;
recipes wire them by ``_target_`` dotpath.
"""

from unirl.models.hunyuan_video.bundle import HunyuanVideoBundle
from unirl.models.hunyuan_video.conditions import HunyuanVideoConditions
from unirl.models.hunyuan_video.config import HunyuanVideoPipelineConfig
from unirl.models.hunyuan_video.pipeline import HunyuanVideoPipeline

__all__ = [
    "HunyuanVideoBundle",
    "HunyuanVideoConditions",
    "HunyuanVideoPipeline",
    "HunyuanVideoPipelineConfig",
]
