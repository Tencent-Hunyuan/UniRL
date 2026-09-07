from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from unirl.config.validation import validate_precision_type

JANUS_PRO_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass
class JanusProPipelineConfig:
    pretrained_model_ckpt_path: str
    trust_remote_code: bool = True

    model_precision: Any = "bf16"
    device: Any = None

    autocast_precision: str = "bf16"
    logprob_precision: str = "fp32"

    use_gradient_checkpointing: bool = False

    # Dedicated-sync modes should address params under the full multimodal
    # wrapper. The trainside recipe does not use this, but keeping it explicit
    # mirrors the other model packages.
    weight_sync_param_name_prefix: str = "language_model."

    use_lora: bool = False
    lora_target_modules: Optional[List[str]] = None

    freeze_vision_tower: bool = True
    freeze_aligner: bool = True
    freeze_generation_tower: bool = True

    max_prompt_length: int = 4096
    user_role: str = "<|User|>"
    assistant_role: str = "<|Assistant|>"
    image_placeholder: str = "<image_placeholder>"

    def __post_init__(self) -> None:
        validate_precision_type(self.model_precision, field="JanusProPipelineConfig.model_precision")
        unsupported_unfrozen = [
            name
            for name, frozen in (
                ("freeze_vision_tower", self.freeze_vision_tower),
                ("freeze_aligner", self.freeze_aligner),
                ("freeze_generation_tower", self.freeze_generation_tower),
            )
            if not frozen
        ]
        if unsupported_unfrozen:
            raise ValueError(
                "Janus-Pro training currently optimizes only language-model decoder blocks; "
                f"{', '.join(unsupported_unfrozen)} must remain true."
            )
        if self.lora_target_modules is None:
            self.lora_target_modules = list(JANUS_PRO_LORA_TARGETS)


__all__ = ["JANUS_PRO_LORA_TARGETS", "JanusProPipelineConfig"]
