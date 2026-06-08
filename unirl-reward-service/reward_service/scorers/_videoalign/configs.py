"""Dataclasses needed to parse a VideoReward checkpoint's
``model_config.json`` and configure inference.

Mirrors the upstream layout (``data.DataConfig``, ``utils.ModelConfig``,
``utils.PEFTLoraConfig``) so a JSON saved by the upstream trainer can be
loaded with field-by-field ``DataclassName(**dict)`` calls.

Stripped from upstream:

* ``utils.TrainingConfig`` — was a subclass of ``transformers.TrainingArguments``
  and pulled in the entire trainer surface. Inference only reads four
  flags (``bf16`` / ``fp16`` / ``gradient_checkpointing`` /
  ``disable_flash_attn2``) so we use a small ``MinimalTrainingArgs``
  dataclass instead and surface those four flags directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union


@dataclass
class DataConfig:
    """Vendored verbatim from upstream ``data.DataConfig``.

    Captured per-checkpoint so prompts / sampling / spatial budgets stay
    aligned with the values the model was trained on.
    """

    meta_data: str = "/path/to/dataset/meta_data.csv"
    data_dir: str = "/path/to/dataset"
    meta_data_test: Optional[str] = None
    max_frame_pixels: int = 240 * 320
    num_frames: Optional[float] = None
    fps: float = 2.0
    p_shuffle_frames: float = 0.0
    p_color_jitter: float = 0.0
    eval_dim: Union[str, List[str]] = "VQ"
    prompt_template_type: str = "none"
    add_noise: bool = False
    sample_type: str = "uniform"
    use_tied_data: bool = True


@dataclass
class PEFTLoraConfig:
    """Vendored from upstream ``utils.PEFTLoraConfig``."""

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
    """Vendored from upstream ``utils.ModelConfig``."""

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
    loss_type: Literal["bt", "reg", "btt", "margin", "constant_margin", "scaled"] = "regular"

    def __post_init__(self):
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("You can't use 8 bit and 4 bit precision at the same time")


@dataclass
class MinimalTrainingArgs:
    """Slim replacement for upstream's ``utils.TrainingConfig`` (a
    ``transformers.TrainingArguments`` subclass).

    The inference path only reads four flags from training_args; pulling
    in the whole HF Trainer surface to access them is overkill — and
    creates a hard dependency on a specific HF transformers version that
    fights with the per-scorer venv pinning.
    """

    load_from_pretrained: Optional[str] = None
    load_from_pretrained_step: Optional[int] = None
    bf16: bool = False
    fp16: bool = False
    gradient_checkpointing: bool = False
    disable_flash_attn2: bool = False
    output_dir: str = ""


def load_configs_from_json(config_path: str):
    """Parse a checkpoint's ``model_config.json``.

    Returns the same shape upstream's ``inference.load_configs_from_json``
    used to: ``(data_config_dict, training_args, model_config_dict,
    peft_lora_config_dict, inference_config | None)``. We hand back raw
    dicts so the caller can pick which dataclass to wrap them in.
    """
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # data_dir / meta_data are training-time paths irrelevant to inference;
    # drop them so a frozen checkpoint can be moved across hosts without
    # carrying stale absolute paths.
    config_dict["data_config"].pop("meta_data", None)
    config_dict["data_config"].pop("data_dir", None)

    return (
        config_dict["data_config"],
        None,  # training_args slot, not used by the inference path
        config_dict["model_config"],
        config_dict["peft_lora_config"],
        config_dict.get("inference_config"),
    )
