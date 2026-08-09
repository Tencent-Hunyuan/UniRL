"""Construction config for the typed Boogu-Image pipeline.

Sibling of :class:`unirl.models.z_image.ZImagePipelineConfig` and
:class:`unirl.models.qwen_image.QwenImagePipelineConfig`. Carries
weights+precision knobs only; LoRA injection, FSDP wrapping, gradient
checkpointing, and offload control all live in the training backend config —
the bundle is weights+params only.

``shift`` lives here so the hosting engine can build the
:class:`FlowMatchSchedulePolicy` at startup. The released
``Boogu-Image-0.1-Base`` scheduler config is a **static** v1 time shift
(``do_shift: true, dynamic_time_shift: false, time_shift_version: "v1",
seq_len: 4096``). Boogu's v1 logistic time shift over ``t`` is algebraically
identical to the standard static sigma shift
``σ' = s·σ / (1 + (s−1)·σ)`` with ``s = e^μ`` and
``μ = lin(seq_len)`` where ``lin`` maps 256→0.5, 4096→1.15 — so the released
checkpoint's schedule is exactly UniRL's static shift with
``s = e^1.15 = 3.158192909689768`` (verified to 1 fp32 ulp against the
reference scheduler). :meth:`BooguImagePipeline.build_schedule_policy` pins
the static posture from this value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from unirl.config.validation import validate_precision_type

BOOGU_IMAGE_BASE_STATIC_SHIFT = 3.158192909689768

_ATTENTION_BACKENDS = ("sdpa", "flash2_varlen")


@dataclass
class BooguImagePipelineConfig:
    """Construction args for ``BooguImagePipeline.from_config``.

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

    shift: float = BOOGU_IMAGE_BASE_STATIC_SHIFT

    weight_sync_param_name_prefix: str = "transformer."

    max_sequence_length: int = 1280

    use_lora: bool = False
    lora_target_modules: Optional[List[str]] = None

    load_text_encoder: bool = True

    load_vae: bool = True

    meta_init_transformer: bool = False

    attention_backend: str = "sdpa"

    def __post_init__(self) -> None:
        validate_precision_type(self.model_precision, field="BooguImagePipelineConfig.model_precision")
        if self.attention_backend not in _ATTENTION_BACKENDS:
            raise ValueError(
                f"BooguImagePipelineConfig.attention_backend must be one of "
                f"{_ATTENTION_BACKENDS}, got {self.attention_backend!r}"
            )


__all__ = ["BooguImagePipelineConfig", "BOOGU_IMAGE_BASE_STATIC_SHIFT"]
