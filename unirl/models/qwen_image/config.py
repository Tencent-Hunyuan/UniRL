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

    # Stage-level precision / numerical policy. Lives here (not on
    # QwenImageDiffusionParams) because these are operator/runtime knobs,
    # not per-request shape.
    autocast_precision: str = "bf16"
    trajectory_precision: str = "fp16"
    logprob_precision: str = "fp32"

    # Static-shift fallback for FlowMatchSchedulePolicy when the pretrained
    # path is not a real local directory (HF repo ID / tests). In practice
    # the real Qwen-Image checkpoint enables dynamic shifting and the
    # ``shift`` value here is ignored — kept aligned with diffusers'
    # default to avoid surprising any ad-hoc consumer.
    shift: float = 3.0

    # Trainer-side policy wraps the bare DiT, while vLLM-Omni loads it
    # under the pipeline's ``transformer.*`` namespace.
    weight_sync_param_name_prefix: str = "transformer."

    # Qwen-Image-specific: text token budget for the Qwen-VL encoder.
    # Hard upper bound is 1024 (the tokenizer cap, including the chat
    # template prefix); see ``QwenImageTextEmbedStage`` for the slicing
    # contract.
    max_sequence_length: int = 512

    # LoRA hints for rollout-side engines (e.g. ``sglang``). Mirrors
    # SD3PipelineConfig; the trainer-side LoRA injection lives in
    # ``cfg.training.policies`` via LoRAPolicy.
    use_lora: bool = False
    lora_target_modules: Optional[List[str]] = None

    # Whether the TRAINER-side bundle loads the Qwen2.5-VL text encoder
    # (~15 GiB bf16 PER RANK). Trainside rollout requires it (the pipeline
    # encodes prompts in-process). Separate-engine recipes (vllm-omni)
    # should set ``false``: the engine encodes prompts in its own workers
    # and the trainer replays from the CAPTURED conditions — the trainer
    # copy is dead weight that starves the colocated engine's boot
    # (LIN-382 qwen probe OOM: 50 MiB free at engine TE load).
    load_text_encoder: bool = True

    # Dynamic-shift declaration for vllm_omni / sglang engines that build
    # ``FlowMatchSchedulePolicy`` from the model_config alone (they don't
    # have a Pipeline instance at engine-init time). Qwen-Image was
    # trained with dynamic shifting; without this declaration, an
    # HF-repo-id path (e.g. ``Qwen/Qwen-Image``) would silently fall
    # back to ``static_only`` because ``scheduler/scheduler_config.json``
    # isn't locally readable at policy-build time. See
    # ``QwenImagePipeline.build_schedule_policy`` for the trainside
    # equivalent.
    use_dynamic_shifting: bool = True
    dynamic_shift_overrides: Dict[str, Any] = field(default_factory=_qwen_image_dynamic_overrides)

    def __post_init__(self) -> None:
        validate_precision_type(self.model_precision, field="QwenImagePipelineConfig.model_precision")


__all__ = ["QwenImagePipelineConfig"]
