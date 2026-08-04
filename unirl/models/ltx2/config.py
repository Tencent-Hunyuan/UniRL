"""Construction config for the LTX-2 / LTX-2.3 T2V / I2V / T2AV pipeline.

LTX-2 supports text-to-video (T2V) and image-to-video (I2V).
LTX-2.3 extends with text-to-audio-video (T2AV) joint generation.

The pipeline auto-detects which capabilities are available from the
checkpoint (audio components present → LTX-2.3 mode enabled).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from unirl.config.validation import validate_precision_type

LTX2_SPATIAL_COMPRESSION = 32
LTX2_TEMPORAL_COMPRESSION = 8
LTX2_LATENT_CHANNELS = 128


@dataclass
class LTX2PipelineConfig:
    """Construction args for ``LTX2Pipeline.from_config``.

    Covers both LTX-2 (video-only) and LTX-2.3 (video+audio).
    Audio components are loaded only when ``enable_audio=True`` and the
    checkpoint contains audio VAE + vocoder weights.
    """

    pretrained_model_ckpt_path: str
    vae_ckpt_path: Optional[str] = None
    text_encoder_ckpt_path: Optional[str] = None

    model_precision: Any = "bf16"
    vae_dtype: Any = None
    text_encoder_dtype: Any = None
    device: Any = None

    autocast_precision: str = "bf16"
    trajectory_precision: str = "fp16"
    logprob_precision: str = "fp32"

    shift: float = 1.0

    max_sequence_length: int = 512

    enable_audio: bool = False

    audio_joint_sde: bool = True

    default_height: int = 512
    default_width: int = 768
    default_num_frames: int = 121  # ~5s at 24fps
    default_frame_rate: float = 24.0

    weight_sync_param_name_prefix: str = "transformer."

    use_lora: bool = False
    lora_target_modules: Optional[list] = None

    aux_components_on_cpu: bool = False

    def __post_init__(self) -> None:
        validate_precision_type(self.model_precision, field="LTX2PipelineConfig.model_precision")


__all__ = ["LTX2PipelineConfig"]
