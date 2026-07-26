"""Construction config for the typed Qwen-Image-Edit-Plus pipeline.

Mirrors :class:`unirl.models.qwen_image.QwenImagePipelineConfig` field-for-
field. The Edit-Plus checkpoint ships the same ``transformer/``, ``vae/``,
``text_encoder/``, ``tokenizer/``, ``scheduler/`` subfolders as base
Qwen-Image; only ``transformer/config.json`` differs (``in_channels=64``
vs ``16`` for the wider input projection that absorbs the source-image
latent concat). The bundle reads ``in_channels`` automatically, so no new
field is needed here.

V1 scope: the low-resolution 384² condition-image path into the Qwen2.5-VL
text encoder (``encode_prompt(image=...)``) is deferred — V1 does standard
text encoding + VAE latent concat only. When V2 adds that path, a
``condition_image_size`` field will land here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from unirl.config.validation import validate_precision_type

# Edit-Plus shares base Qwen-Image's canonical dynamic-shift params verbatim
# (same VAE scale factor 8, patch size 2, scheduler_config.json shape); the
# shift is derived from the noise latent's ``image_seq_len`` only — the
# source-image concat happens inside ``predict_noise`` after the schedule is
# fixed. ``pipeline.build_schedule_policy`` uses the same function.
from unirl.models.qwen_image.config import _qwen_image_dynamic_overrides


@dataclass
class QwenImageEditPlusPipelineConfig:
    """Construction args for :meth:`QwenImageEditPlusPipeline.from_config`.

    Field-for-field compatible with
    :class:`unirl.models.qwen_image.QwenImagePipelineConfig` — the Edit-Plus
    bundle inherits :meth:`QwenImageBundle.from_config` unchanged, so every
    knob (paths, precision, LoRA hints, meta-init, dynamic-shift) carries
    the same meaning.
    """

    pretrained_model_ckpt_path: str
    vae_ckpt_path: Optional[str] = None
    text_encoder_ckpt_path: Optional[str] = None
    vae_dtype: Any = None
    text_encoder_dtype: Any = None
    model_precision: Any = "bf16"
    device: Any = None

    autocast_precision: str = "bf16"
    trajectory_precision: str = "fp16"
    logprob_precision: str = "fp32"

    shift: float = 3.0

    weight_sync_param_name_prefix: str = "transformer."

    max_sequence_length: int = 512

    use_lora: bool = False
    lora_target_modules: Optional[List[str]] = None

    # Trainer-side TE (~15 GiB/rank). False for separate-engine: engine encodes;
    # trainer replays captured conditions (keeps VRAM for colocated engine boot).
    load_text_encoder: bool = True
    # Trainer-side VAE. False for separate-engine recipes (engine owns encode/decode).
    load_vae: bool = True
    meta_init_transformer: bool = False

    use_dynamic_shifting: bool = True
    dynamic_shift_overrides: Dict[str, Any] = field(default_factory=_qwen_image_dynamic_overrides)

    def __post_init__(self) -> None:
        validate_precision_type(self.model_precision, field="QwenImageEditPlusPipelineConfig.model_precision")


__all__ = ["QwenImageEditPlusPipelineConfig"]
