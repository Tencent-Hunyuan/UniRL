"""Construction config for the typed Qwen-Image-Edit-Plus pipeline.

Mirrors :class:`unirl.models.qwen_image.QwenImagePipelineConfig` field-for-
field. The Edit-Plus checkpoint ships the same ``transformer/``, ``vae/``,
``text_encoder/``, ``tokenizer/``, ``scheduler/`` subfolders as base
Qwen-Image; only ``transformer/config.json`` differs (``in_channels=64``
vs ``16`` for the wider input projection that absorbs the source-image
latent concat). The bundle reads ``in_channels`` automatically, so no new
field is needed here.

Condition-image conditioning: ``use_condition_image_prompt`` (default True)
feeds the source image into the Qwen2.5-VL text encoder — the correct
Edit-Plus behavior (mirrors upstream ``encode_prompt(image=...)`` and the
SGLang rollout path). Set it False for Edit **text-only** encoding: same
edit chat template / drop-64, empty image prefix (upstream
``_get_qwen_prompt_embeds(..., image=None)``). This is **not** a switch to
base Qwen-Image's text-only stage (different system prompt / drop-34).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from unirl.config.validation import validate_precision_type
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

    batch_replay_steps: bool = False

    shift: float = 3.0

    weight_sync_param_name_prefix: str = "transformer."

    max_sequence_length: int = 512

    use_condition_image_prompt: bool = True

    use_lora: bool = False
    lora_target_modules: Optional[List[str]] = None

    load_text_encoder: bool = True
    load_vae: bool = True
    meta_init_transformer: bool = False

    use_dynamic_shifting: bool = True
    dynamic_shift_overrides: Dict[str, Any] = field(default_factory=_qwen_image_dynamic_overrides)

    def __post_init__(self) -> None:
        validate_precision_type(self.model_precision, field="QwenImageEditPlusPipelineConfig.model_precision")


__all__ = ["QwenImageEditPlusPipelineConfig"]
