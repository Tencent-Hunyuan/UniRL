"""Dataclass configs for the VideoAlign reward model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass
class TrainingConfig:
    """Inference-relevant slice of the trainer's :class:`TrainingArguments`."""

    output_dir: str = ""
    bf16: bool = False
    fp16: bool = False
    gradient_checkpointing: bool = False
    disable_flash_attn2: bool = False

    max_length: Optional[int] = None
    dataset_num_proc: Optional[int] = None
    center_rewards_coefficient: Optional[float] = None
    vision_lr: Optional[float] = None
    merger_lr: Optional[float] = None
    special_token_lr: Optional[float] = None
    conduct_eval: Optional[bool] = True
    load_from_pretrained: Optional[str] = None
    load_from_pretrained_step: Optional[int] = None
    logging_epochs: Optional[float] = None
    eval_epochs: Optional[float] = None
    save_epochs: Optional[float] = None
    remove_unused_columns: Optional[bool] = False
    save_full_model: Optional[bool] = False


@dataclass
class PEFTLoraConfig:
    """LoRA wiring for the reward model."""

    lora_enable: bool = False
    vision_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None
    lora_namespan_exclude: Optional[List[str]] = None
    lora_modules_to_save: Optional[List[str]] = None
    lora_task_type: str = "CAUSAL_LM"
    use_rslora: bool = False
    num_lora_modules: int = -1

    def __post_init__(self) -> None:
        if isinstance(self.lora_target_modules, list) and len(self.lora_target_modules) == 1:
            self.lora_target_modules = self.lora_target_modules[0]
        if isinstance(self.lora_namespan_exclude, list) and len(self.lora_namespan_exclude) == 1:
            self.lora_namespan_exclude = self.lora_namespan_exclude[0]


@dataclass
class ModelConfig:
    """Backbone + reward-head configuration."""

    model_name_or_path: Optional[str] = None
    model_revision: str = "main"

    output_dim: int = 1

    use_special_tokens: bool = False

    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    tune_merger: bool = field(default=False)

    torch_dtype: Optional[Literal["auto", "bfloat16", "float16", "float32"]] = None
    trust_remote_code: bool = False
    attn_implementation: Optional[str] = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    bnb_4bit_quant_type: Literal["fp4", "nf4"] = "nf4"
    use_bnb_nested_quant: bool = False

    reward_token: Literal["last", "mean", "special"] = "last"

    loss_type: Literal["bt", "reg", "btt", "margin", "constant_margin", "scaled", "regular"] = "regular"

    def __post_init__(self) -> None:
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("You can't use 8 bit and 4 bit precision at the same time")


__all__ = ["ModelConfig", "PEFTLoraConfig", "TrainingConfig"]
