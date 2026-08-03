"""Construction config for the typed FLUX.2-klein-9B pipeline.

Sibling of :class:`unirl.models.sd3.SD3PipelineConfig` and
:class:`unirl.models.qwen_image.QwenImagePipelineConfig`.
Carries weights+precision knobs only; LoRA injection, FSDP wrapping,
gradient checkpointing, and offload control all live in
``cfg.training.policies`` (``LoRAPolicy`` / ``FSDPPolicy``) — the
bundle is weights+params only.

Klein uses the official FLUX.2 empirical-μ shifting (a function of packed
image-seq-len + number of inference steps). The pipeline's
``build_schedule_policy()`` returns a ``Flux2KleinSchedulePolicy`` whose
:meth:`compute_mu` override supplies that empirical μ; the shared
dynamic-shift application in
:meth:`unirl.sde.runtime.FlowMatchSchedulePolicy.compute_sigma` does the rest. The
static ``shift`` field is only consulted by the static-shift fallback,
which Klein does not use (``use_dynamic_shifting=True``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from unirl.config.validation import validate_precision_type


@dataclass
class Flux2KleinPipelineConfig:
    """Construction args for ``Flux2KleinPipeline.from_config``.

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

    shift: float = 1.0

    weight_sync_param_name_prefix: str = "transformer."

    meta_init_transformer: bool = False

    max_sequence_length: int = 512

    qwen3_extraction_layers: Tuple[int, ...] = (9, 18, 27)

    use_lora: bool = False
    lora_target_modules: Optional[List[str]] = None

    load_vae: bool = True

    def __post_init__(self) -> None:
        validate_precision_type(self.model_precision, field="Flux2KleinPipelineConfig.model_precision")

    def build_schedule_policy(self):
        """Build the Klein-specific schedule policy without a Pipeline instance.

        Delegates to the same helper used by
        :meth:`Flux2KleinPipeline.build_schedule_policy` so engines that hold
        a config but never instantiate a Pipeline (SGLang / vLLM-Omni run the
        model in a subprocess) can still get the empirical-μ schedule.

        ``Flux2KleinSchedulePolicy(use_dynamic_shifting=True)`` overrides
        :meth:`FlowMatchSchedulePolicy.compute_mu` with the FLUX.2
        empirical formula; everything else (base grid, exponential time
        shift, terminal zero) is the shared dynamic-shift path. The
        static ``shift`` field is the documented static-fallback only
        and is unused when dynamic shifting is active.
        """
        from unirl.models.flux2_klein.schedule import build_flux2_klein_schedule_policy

        return build_flux2_klein_schedule_policy(self.shift)


__all__ = ["Flux2KleinPipelineConfig"]
