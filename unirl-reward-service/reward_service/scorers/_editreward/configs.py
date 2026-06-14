"""Configuration dataclasses for EditReward inference.

Vendored from EditReward/utils/parser.py — stripped of TrainingArguments
dependency (uses a lightweight stub instead).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional, Tuple

from omegaconf import OmegaConf


@dataclass
class DataConfig:
    train_json_list: List[str] = field(default_factory=list)
    val_json_list: List[str] = field(default_factory=list)
    test_json_list: List[str] = field(default_factory=list)
    soft_label: bool = False
    confidence_threshold: Optional[float] = None
    max_pixels: Optional[int] = 256 * 28 * 28
    min_pixels: Optional[int] = 256 * 28 * 28
    with_instruction: bool = True
    tied_threshold: Optional[float] = None
    reward_dim: str = "dim1"


@dataclass
class TrainingConfig:
    """Lightweight stub replacing transformers.TrainingArguments for inference."""

    output_dir: str = "/tmp/editreward_output"
    bf16: bool = True
    fp16: bool = False
    disable_flash_attn2: bool = False
    disable_dropout: bool = False


@dataclass
class PEFTLoraConfig:
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

    def __post_init__(self):
        if isinstance(self.lora_target_modules, list) and len(self.lora_target_modules) == 1:
            self.lora_target_modules = self.lora_target_modules[0]
        if isinstance(self.lora_namespan_exclude, list) and len(self.lora_namespan_exclude) == 1:
            self.lora_namespan_exclude = self.lora_namespan_exclude[0]


@dataclass
class ModelConfig:
    model_name_or_path: Optional[str] = None
    model_revision: str = "main"
    rm_head_type: str = "default"
    rm_head_kwargs: Optional[dict] = None
    pooling_strategy: str = "min"
    output_dim: int = 1
    use_special_tokens: bool = False
    freeze_vision_tower: bool = False
    freeze_llm: bool = False
    tune_merger: bool = False
    trainable_visual_layers: Optional[int] = -1
    torch_dtype: Optional[Literal["auto", "bfloat16", "float16", "float32"]] = None
    trust_remote_code: bool = False
    attn_implementation: Optional[str] = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    bnb_4bit_quant_type: Literal["fp4", "nf4"] = "nf4"
    use_bnb_nested_quant: bool = False
    reward_token: Literal["last", "mean", "special"] = "last"
    loss_type: str = "regular"
    loss_hyperparameters: dict = field(default_factory=dict)
    checkpoint_path: Optional[str] = None


def parse_args_with_yaml(
    dataclass_types: Tuple[type, ...],
    config_path: str,
    is_train: bool = True,
) -> Tuple[Any, ...]:
    """Parse YAML config into dataclass instances (inference-only, no HfArgumentParser)."""
    raw = OmegaConf.to_container(OmegaConf.load(config_path))
    if not is_train:
        raw.pop("deepspeed", None)

    results = []
    for dc_type in dataclass_types:
        # Extract fields that belong to this dataclass
        valid_fields = {f.name for f in dc_type.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in raw.items() if k in valid_fields}
        results.append(dc_type(**kwargs))

    return tuple(results), config_path
