"""Construction config for SenseNova-U1.5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

from unirl.config.validation import validate_precision_type

SENSENOVA_U1_GEN_LORA_TARGETS: Tuple[str, ...] = (
    "self_attn.q_proj_mot_gen",
    "self_attn.k_proj_mot_gen",
    "self_attn.v_proj_mot_gen",
    "self_attn.o_proj_mot_gen",
    "mlp_mot_gen.gate_proj",
    "mlp_mot_gen.up_proj",
    "mlp_mot_gen.down_proj",
)


@dataclass
class SenseNovaU1PipelineConfig:
    """Construction args for the released SenseNova-U1.5 pixel-flow model."""

    pretrained_model_ckpt_path: str
    model_precision: Any = "bf16"
    device: Any = None

    autocast_precision: str = "bf16"
    trajectory_precision: str = "bf16"
    logprob_precision: str = "fp32"

    timestep_shift: float = 3.0
    # The pristine vendor's forced "flash" mode requires Flash-Attention 2,
    # while UniRL's supported engine stack ships Flash-Attention 4. Keep
    # automatic fallback and explicit SDPA until a tested FA4 adapter exists.
    attention_backend: str = "auto"
    full_finetune_generation: bool = True

    # FSDP wraps SenseNovaU1TrainableModel, whose NEOChatModel child is `model`.
    weight_sync_param_name_prefix: str = "model."

    def __post_init__(self) -> None:
        validate_precision_type(self.model_precision, field="SenseNovaU1PipelineConfig.model_precision")
        if self.attention_backend not in {"auto", "sdpa"}:
            raise ValueError(
                "SenseNovaU1PipelineConfig.attention_backend must be one of "
                f"('auto', 'sdpa'); got {self.attention_backend!r}. "
                "The vendored 'flash' backend requires Flash-Attention 2, which "
                "is incompatible with UniRL's Flash-Attention 4 stack."
            )
        if float(self.timestep_shift) <= 0:
            raise ValueError(f"SenseNovaU1PipelineConfig.timestep_shift must be positive; got {self.timestep_shift}.")


__all__ = ["SENSENOVA_U1_GEN_LORA_TARGETS", "SenseNovaU1PipelineConfig"]
