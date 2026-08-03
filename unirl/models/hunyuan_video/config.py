"""Construction config for the typed HunyuanVideo-1.0 pipeline.

Sibling of :class:`unirl.models.hunyuan_video15.HunyuanVideo15PipelineConfig`.
Carries weights+precision knobs only; LoRA injection, FSDP wrapping,
gradient checkpointing, and offload control all live in
``cfg.training.policies`` (``LoRAPolicy`` / ``FSDPPolicy``) -- the bundle
is weights+params only.

HunyuanVideo-1.0-specific vs 1.5:

- No ``text_encoder_2_ckpt_path`` needed (CLIP is always co-located with
  the main checkpoint under ``text_encoder_2/`` + ``tokenizer_2/``).
- No ``byt5_*``, ``mllm_*``, ``vision_*`` fields (1.0 uses LLaMA + CLIP,
  not Qwen2.5-VL + ByT5 + SigLIP).
- ``llama_max_length`` / ``clip_max_length`` / ``crop_start`` shape the
  text encoding (LLaMA prompt template crops the first ``crop_start``
  tokens after encoding).
- ``guidance_embeds=True`` on the transformer -- guidance scale is passed
  as a tensor, NOT as CFG stacking.

``shift`` defaults to 5.0 (the upstream HunyuanVideo default). Static
shift only (``use_dynamic_shifting=False``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from unirl.config.validation import validate_precision_type


@dataclass
class HunyuanVideoPipelineConfig:
    """Construction args for ``HunyuanVideoPipeline.from_config``.

    ``device`` may be runtime-injected by the actor after compose; the
    other fields are set at compose time and read once during pipeline
    construction.
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

    shift: float = 5.0

    weight_sync_param_name_prefix: str = "transformer."

    meta_init_transformer: bool = False

    latent_channels: Optional[int] = None

    vae_use_tiling: bool = False

    llama_max_length: int = 256
    crop_start: int = 95
    clip_max_length: int = 77
    hidden_state_skip_layer: int = 2

    use_lora: bool = False
    lora_target_modules: Optional[List[str]] = None

    def __post_init__(self) -> None:
        validate_precision_type(self.model_precision, field="HunyuanVideoPipelineConfig.model_precision")
        self.hidden_state_skip_layer = int(self.hidden_state_skip_layer)
        if self.hidden_state_skip_layer < 0:
            raise ValueError(
                f"HunyuanVideoPipelineConfig.hidden_state_skip_layer must be >= 0, got {self.hidden_state_skip_layer}"
            )


__all__ = ["HunyuanVideoPipelineConfig"]
