"""Construction config for the typed Qwen-Image pipeline.

Sibling of :class:`unirl.models.sd3.SD3PipelineConfig`. Carries
weights+precision knobs only; LoRA injection, FSDP wrapping, gradient
checkpointing, and offload control all live in ``cfg.training.policies``
(``LoRAPolicy`` / ``FSDPPolicy``) — the bundle is weights+params only.

``shift`` lives here so the hosting engine can build the
:class:`FlowMatchSchedulePolicy` at startup. Qwen-Image's
``scheduler/scheduler_config.json`` ships with
``use_dynamic_shifting: True`` + ``base_shift / max_shift /
base_image_seq_len / max_image_seq_len``, so
:meth:`FlowMatchSchedulePolicy.from_pretrained` picks up the dynamic
branch automatically. The ``shift`` argument is only used as a fallback
when no pretrained dir is available (tests / ad-hoc smoke).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from unirl.config.validation import validate_precision_type


def _qwen_image_dynamic_overrides() -> Dict[str, Any]:
    """Canonical dynamic-shift params for Qwen-Image.

    Mirrors ``Qwen/Qwen-Image``'s ``scheduler/scheduler_config.json``
    (max_shift 0.9, max_image_seq_len 8192, shift_terminal 0.02 — NOT the
    Flux-flavored ``calculate_shift`` defaults this dict previously carried).
    Used by vllm_omni / sglang engines that read
    ``model_config.dynamic_shift_overrides`` when constructing
    :class:`FlowMatchSchedulePolicy` for an HF-repo-id path where
    ``scheduler/scheduler_config.json`` can't be read directly; a local
    checkpoint dir reads the real JSON and must produce the same values.

    Trainside engine reads these via ``QwenImagePipeline.build_schedule_policy``.
    Keep the two paths in sync if anything here changes.
    """
    return {
        "base_shift": 0.5,
        "max_shift": 0.9,
        "base_image_seq_len": 256,
        "max_image_seq_len": 8192,
        "time_shift_type": "exponential",
        "shift_terminal": 0.02,
        "vae_scale_factor": 8,
        "patch_size": 2,
    }


@dataclass
class QwenImagePipelineConfig:
    """Construction args for ``QwenImagePipeline.from_config``.

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

    batch_replay_steps: bool = False

    shift: float = 3.0

    weight_sync_param_name_prefix: str = "transformer."

    max_sequence_length: int = 512

    use_lora: bool = False
    lora_target_modules: Optional[List[str]] = None

    load_text_encoder: bool = True
    load_vae: bool = True
    meta_init_transformer: bool = False

    use_dynamic_shifting: bool = True
    dynamic_shift_overrides: Dict[str, Any] = field(default_factory=_qwen_image_dynamic_overrides)

    def __post_init__(self) -> None:
        validate_precision_type(self.model_precision, field="QwenImagePipelineConfig.model_precision")


__all__ = ["QwenImagePipelineConfig"]
