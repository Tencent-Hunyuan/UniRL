"""Construction config for the typed Z-Image pipeline.

Sibling of :class:`unirl.models.sd3.SD3PipelineConfig` and
:class:`unirl.models.qwen_image.QwenImagePipelineConfig`. Carries
weights+precision knobs only; LoRA injection, FSDP wrapping, gradient
checkpointing, and offload control all live in ``cfg.training.policies``
(``LoRAPolicy`` / ``FSDPPolicy``) — the bundle is weights+params only.

``shift`` lives here so the hosting engine can build the
:class:`FlowMatchSchedulePolicy` at startup. Both Z-Image variants ship a
``scheduler/scheduler_config.json`` with ``use_dynamic_shifting: false``
(plain static FlowMatch shift), but the shift value differs:
**base ``Tongyi-MAI/Z-Image`` uses ``shift: 6.0``**, while the distilled
``Z-Image-Turbo`` uses ``shift: 3.0``. The default below targets the base
model. :meth:`ZImagePipeline.build_schedule_policy` pins the static posture
from this value (unlike Qwen-Image / Flux.2, which are dynamic-shift).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from unirl.config.validation import validate_precision_type


@dataclass
class ZImagePipelineConfig:
    """Construction args for ``ZImagePipeline.from_config``.

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

    shift: float = 6.0

    weight_sync_param_name_prefix: str = "transformer."

    max_sequence_length: int = 512

    use_lora: bool = False
    lora_target_modules: Optional[List[str]] = None

    load_vae: bool = True

    def __post_init__(self) -> None:
        validate_precision_type(self.model_precision, field="ZImagePipelineConfig.model_precision")


__all__ = ["ZImagePipelineConfig"]
