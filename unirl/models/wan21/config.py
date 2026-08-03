"""Construction config for the new typed WAN 2.1 T2V / I2V pipeline.

Mirrors :class:`unirl.models.sd3.config.SD3PipelineConfig` shape —
weights+params only, no LoRA / FSDP / offload knobs. Those lifecycle
concerns live outside the bundle (in ``training`` Policy stack).

WAN 2.1 supports both T2V and I2V. The I2V path activates when the
transformer checkpoint declares ``image_dim > 0`` — the bundle then
loads a CLIP vision tower (``CLIPVisionModel`` + image processor) from
``image_encoder_ckpt_path`` (or, when ``None``, the ``image_encoder/``
subfolder under ``pretrained_model_ckpt_path``). Setting
``image_encoder_ckpt_path`` against a ``image_dim == 0`` checkpoint is
an error — there is no silent fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from unirl.config.validation import validate_precision_type


@dataclass
class WAN21PipelineConfig:
    """Construction args for ``WAN21Pipeline.from_config``.

    ``device`` may be runtime-injected by the actor after compose; the
    other fields are set at compose time and read once during pipeline
    construction.
    """

    pretrained_model_ckpt_path: str
    vae_ckpt_path: Optional[str] = None
    text_encoder_ckpt_path: Optional[str] = None
    # I2V (CLIP vision tower) override. Setting this on a ``image_dim == 0`` checkpoint raises — there is no silent fallback.
    image_encoder_ckpt_path: Optional[str] = None
    vae_dtype: Any = None
    text_encoder_dtype: Any = None
    model_precision: Any = "bf16"
    device: Any = None

    # Stage-level precision / numerical policy. Lives here (not on WAN21DiffusionParams) because these are operator/runtime knobs, not per-request shape.
    autocast_precision: str = "bf16"
    trajectory_precision: str = "fp16"
    logprob_precision: str = "fp32"

    shift: float = 5.0

    max_sequence_length: int = 512

    weight_sync_param_name_prefix: str = "transformer."

    # Optional LoRA hints for rollout-side engines (sglang in particular). The SGLang rollout server still needs to know at construction time whether to boot in LoRA mode and which target modules to wrap — those flags travel through this model_config.
    use_lora: bool = False
    lora_target_modules: Optional[List[str]] = None

    # Meta-init the transformer (build on the meta device; the backend loads weights after sharding) instead of eager ``from_pretrained``.
    meta_init_transformer: bool = False

    load_vae: bool = True

    def __post_init__(self) -> None:
        validate_precision_type(self.model_precision, field="WAN21PipelineConfig.model_precision")


__all__ = ["WAN21PipelineConfig"]
