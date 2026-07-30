"""Dataclass configs for the VideoAlign reward model.

Inference-only subset of the originals at
``mmrl/recipes/rewards/videoalign/vendor/videoalign/utils.py``. We keep the
exact field names + default values that appear in checkpoints'
``model_config.json``, so ``ModelConfig(**dict_from_json)`` /
``PEFTLoraConfig(**dict_from_json)`` continue to round-trip every public
VideoAlign release. Training-only fields (``vision_lr``, ``merger_lr``,
``conduct_eval``, ``logging_epochs`` …) live on :class:`TrainingConfig` and
are accepted-but-ignored by the inference path — they only end up here
because ``model_config.json`` was written by the trainer.

Why we DON'T inherit from ``transformers.TrainingArguments``
-----------------------------------------------------------
The mmrl vendor's ``TrainingConfig`` extends ``TrainingArguments`` which
runs an ``__post_init__`` validating distributed launch fields
(``local_rank``, ``deepspeed`` …). At reward *inference* time the call site
is ``TrainingConfig(load_from_pretrained=..., bf16=..., output_dir="")`` and
the only fields we actually read downstream are ``bf16`` / ``fp16`` /
``gradient_checkpointing`` / ``disable_flash_attn2``. So we re-declare a
plain ``@dataclass`` with just those fields plus a generous ``**kwargs``
catch-all (``__init__`` ignores unknown keys via ``__init_subclass__`` —
no, simpler: we filter in the loader). That keeps load-time light and
removes the transformers private-symbol coupling entirely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass
class TrainingConfig:
    """Inference-relevant slice of the trainer's :class:`TrainingArguments`.

    Only ``bf16`` / ``fp16`` / ``gradient_checkpointing`` /
    ``disable_flash_attn2`` are actually consumed by the inference path
    (see :func:`experimental.refl.reward.videoalign.model.factory.create_model_and_processor`).
    The other fields are kept so ``TrainingConfig(**model_config_json["training_args"])``
    still works when someone wants to introspect the saved config.
    """

    # Inference-time knobs (read by ``create_model_and_processor``)
    output_dir: str = ""
    bf16: bool = False
    fp16: bool = False
    gradient_checkpointing: bool = False
    disable_flash_attn2: bool = False

    # Train-only — kept as no-op defaults so __init__ accepts them.
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
    """LoRA wiring for the reward model.

    For inference we typically want ``lora_enable=True`` only if the
    checkpoint was saved as a LoRA split (``adapter_model.safetensors`` +
    ``non_lora_state_dict.pth``) — :func:`load_model_from_checkpoint`
    auto-detects which branch to take. The other fields drive the LoRA
    target-module discovery inside :func:`create_model_and_processor`.
    """

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
        # Mirror the upstream normalisation: a single-element list is
        # flattened to a scalar so peft's ``target_modules`` accepts it.
        if isinstance(self.lora_target_modules, list) and len(self.lora_target_modules) == 1:
            self.lora_target_modules = self.lora_target_modules[0]
        if isinstance(self.lora_namespan_exclude, list) and len(self.lora_namespan_exclude) == 1:
            self.lora_namespan_exclude = self.lora_namespan_exclude[0]


@dataclass
class ModelConfig:
    """Backbone + reward-head configuration.

    Read from ``model_config.json::model_config``. Most fields just pass
    through to :meth:`Qwen2VLRewardModelBT.from_pretrained`.
    """

    model_name_or_path: Optional[str] = None
    model_revision: str = "main"

    # Output dimensionality of the reward head: 3 for joint (VQ, MQ, TA)
    # heads, 1 for single-attribute checkpoints.
    output_dim: int = 1

    # Whether the checkpoint introduces ``<|VQ_reward|>`` / ``<|MQ_reward|>``
    # / ``<|TA_reward|>`` special tokens (matches the
    # ``detailed_special`` prompt template). When True, the reward is read
    # from those token positions instead of the last token.
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

    # Where the reward is read from the hidden-state sequence:
    # ``"last"``    — last non-pad token (standard).
    # ``"mean"``    — masked mean over the prompt.
    # ``"special"`` — special-token positions (set automatically when
    #                 ``use_special_tokens=True`` and the checkpoint
    #                 declares ``additional_special_tokens``).
    reward_token: Literal["last", "mean", "special"] = "last"

    # Bradley-Terry / regression flavour — train-only. Inference ignores
    # this entirely; kept for round-trip compatibility with
    # ``model_config.json``.
    loss_type: Literal["bt", "reg", "btt", "margin", "constant_margin", "scaled", "regular"] = "regular"

    def __post_init__(self) -> None:
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("You can't use 8 bit and 4 bit precision at the same time")


__all__ = ["ModelConfig", "PEFTLoraConfig", "TrainingConfig"]
