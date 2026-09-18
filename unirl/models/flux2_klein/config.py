"""Construction config for the typed FLUX.2-klein-9B pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from unirl.config.validation import validate_precision_type


@dataclass
class Flux2KleinPipelineConfig:
    """Construction args for ``Flux2KleinPipeline.from_config``."""

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
        """Build the Klein-specific schedule policy without a Pipeline instance."""
        from unirl.models.flux2_klein.schedule import build_flux2_klein_schedule_policy

        return build_flux2_klein_schedule_policy(self.shift)


__all__ = ["Flux2KleinPipelineConfig"]
