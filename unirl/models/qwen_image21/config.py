"""Construction config for the Qwen-Image-2.1 bundle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from unirl.config.validation import validate_precision_type


@dataclass
class QwenImage21PipelineConfig:
    """Construction args for ``QwenImage21Bundle.from_config``."""

    pretrained_model_ckpt_path: str
    model_precision: Any = "bf16"

    def __post_init__(self) -> None:
        validate_precision_type(self.model_precision, field="QwenImage21PipelineConfig.model_precision")


__all__ = ["QwenImage21PipelineConfig"]
