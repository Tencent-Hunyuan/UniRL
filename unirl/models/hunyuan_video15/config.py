"""Construction config for the typed HunyuanVideo-1.5 pipeline.

Sibling of :class:`unirl.models.wan21.WAN21PipelineConfig`.
Carries weights+precision knobs only; LoRA injection, FSDP wrapping,
gradient checkpointing, and offload control all live in
``cfg.training.policies`` (``LoRAPolicy`` / ``FSDPPolicy``) — the bundle
is weights+params only.

Three knobs are HunyuanVideo-1.5-specific (vs the SD3 / Qwen-Image /
WAN21 sibling configs):

- ``text_encoder_2_ckpt_path``: ByT5 glyph encoder lives in a separate
  HuggingFace subfolder (``text_encoder_2`` / ``tokenizer_2``); recipes
  that load Qwen-VL + ByT5 from the same checkpoint dir can leave this
  ``None`` and the bundle falls back to ``pretrained_model_ckpt_path``.
- ``image_encoder_ckpt_path``: SigLIP vision encoder for I2V (only used
  when ``load_vision_encoder=True``).
- ``mllm_*`` / ``byt5_max_length`` / ``vision_*``: shape parameters
  copied verbatim from the upstream HunyuanVideo15 pipeline; tweaking
  these without also retraining the transformer breaks the model.

``shift`` defaults to 5.0 (the upstream HunyuanVideo-1.5 default; SD3
uses 3.0). Unlike Qwen-Image, HunyuanVideo-1.5 uses **static** shift,
not dynamic-mu — :class:`FlowMatchSchedulePolicy.from_pretrained` will
read ``use_dynamic_shifting=False`` from the checkpoint's
``scheduler_config.json`` and stick to the static branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from unirl.config.validation import validate_precision_type


@dataclass
class HunyuanVideo15PipelineConfig:
    """Construction args for ``HunyuanVideo15Pipeline.from_config``.

    ``device`` may be runtime-injected by the actor after compose; the
    other fields are set at compose time and read once during pipeline
    construction.
    """

    pretrained_model_ckpt_path: str
    vae_ckpt_path: Optional[str] = None
    text_encoder_ckpt_path: Optional[str] = None
    text_encoder_2_ckpt_path: Optional[str] = None
    image_encoder_ckpt_path: Optional[str] = None
    vae_dtype: Any = None
    text_encoder_dtype: Any = None
    model_precision: Any = "bf16"
    device: Any = None

    autocast_precision: str = "bf16"
    trajectory_precision: str = "fp16"
    logprob_precision: str = "fp32"

    shift: float = 5.0

    weight_sync_param_name_prefix: str = "transformer."

    meta_init_transformer: bool = False

    latent_channels: Optional[int] = None

    mllm_max_length: int = 1000
    mllm_crop_start: int = 108
    mllm_skip_layers: int = 2
    byt5_max_length: int = 256

    vision_num_semantic_tokens: int = 729
    vision_states_dim: int = 1152
    load_vision_encoder: bool = False

    load_vae: bool = True

    use_lora: bool = False
    lora_target_modules: Optional[List[str]] = None

    def __post_init__(self) -> None:
        validate_precision_type(self.model_precision, field="HunyuanVideo15PipelineConfig.model_precision")


__all__ = ["HunyuanVideo15PipelineConfig"]
