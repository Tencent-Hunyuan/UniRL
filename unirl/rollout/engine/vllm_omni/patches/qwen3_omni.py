"""Qwen3-Omni model and verl compatibility patches."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_APPLIED = False


def _register_qwen3_omni_automodel() -> None:
    """Register the Thinker model and its FSDP requirements."""
    try:
        from transformers import AutoModelForCausalLM
        from transformers.models.qwen3_omni_moe import (
            Qwen3OmniMoeConfig,
            Qwen3OmniMoeForConditionalGeneration,
        )
    except ImportError:
        return

    # Register with verl when available.
    try:
        from verl.utils.model import _architecture_to_auto_class

        _architecture_to_auto_class.setdefault(
            "Qwen3OmniMoeForConditionalGeneration", AutoModelForCausalLM
        )
    except ImportError:
        pass

    def _qwen3_omni_get_input_embeddings(self: Any) -> Any:
        return self.thinker.get_input_embeddings()

    def _qwen3_omni_set_input_embeddings(self: Any, value: Any) -> None:
        self.thinker.set_input_embeddings(value)

    def _qwen3_omni_forward(
        self: Any,
        input_ids: Any = None,
        attention_mask: Any = None,
        position_ids: Any = None,
        past_key_values: Any = None,
        inputs_embeds: Any = None,
        labels: Any = None,
        use_cache: Any = None,
        output_attentions: Any = None,
        output_hidden_states: Any = None,
        return_dict: Any = None,
        **kwargs: Any,
    ) -> Any:
        return self.thinker(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

    Qwen3OmniMoeForConditionalGeneration.forward = _qwen3_omni_forward
    Qwen3OmniMoeForConditionalGeneration.get_input_embeddings = _qwen3_omni_get_input_embeddings
    Qwen3OmniMoeForConditionalGeneration.set_input_embeddings = _qwen3_omni_set_input_embeddings
    # Use the actual decoder layer class.
    Qwen3OmniMoeForConditionalGeneration._no_split_modules = ["Qwen3OmniMoeThinkerTextDecoderLayer"]
    # verl strips components unused by Thinker-only training.
    Qwen3OmniMoeForConditionalGeneration._verl_strip_modules = [
        "talker",
        "code2wav",
        "code_predictor",
    ]

    # Keep verl's FSDP initialization on the meta-tensor path.
    logger.warning(
        "unirl.vllm_omni.patches.qwen3_omni: forcing tie_word_embeddings=False on "
        "Qwen3OmniMoeConfig — tied embeddings disable the FSDP meta-tensor init "
        "path and OOM on 30B-A3B."
    )

    class _FalseTieDescriptor:
        def __get__(self, obj: Any, objtype: Any = None) -> bool:
            return False

        def __set__(self, obj: Any, value: Any) -> None:
            pass

    Qwen3OmniMoeConfig.tie_word_embeddings = _FalseTieDescriptor()
    AutoModelForCausalLM.register(Qwen3OmniMoeConfig, Qwen3OmniMoeForConditionalGeneration)


def _patch_hf_processor_for_qwen3_omni() -> None:
    """Teach verl's processor loader about Qwen3-Omni."""
    try:
        from transformers.models.qwen3_omni_moe import Qwen3OmniMoeThinkerForConditionalGeneration
    except ImportError:
        return

    try:
        import verl.utils.tokenizer as _vt
    except ImportError:
        return

    import types

    _original_hf_processor = _vt.hf_processor

    def _patched_hf_processor(name_or_path: Any, **kwargs: Any) -> Any:
        result = _original_hf_processor(name_or_path, **kwargs)
        if result is not None:
            return result

        try:
            from transformers import AutoConfig, AutoProcessor, PreTrainedTokenizerBase

            processor = AutoProcessor.from_pretrained(name_or_path, **kwargs)
            if isinstance(processor, PreTrainedTokenizerBase):
                return None
            if processor.__class__.__name__ != "Qwen3OmniMoeProcessor":
                return None

            config = AutoConfig.from_pretrained(name_or_path, **kwargs)
            # Processor helpers consume the nested Thinker config.
            processor.config = config.thinker_config
            processor.spatial_merge_size = config.thinker_config.vision_config.spatial_merge_size
            model_class = Qwen3OmniMoeThinkerForConditionalGeneration
            processor.get_rope_index = types.MethodType(model_class.get_rope_index, processor)
            processor.get_llm_pos_ids_for_vision = types.MethodType(
                model_class.get_llm_pos_ids_for_vision, processor
            )
            return processor
        except Exception:
            return None

    _vt.hf_processor = _patched_hf_processor
    # Refresh verl's already-imported re-export.
    import sys as _sys

    _utils_mod = _sys.modules.get("verl.utils")
    if _utils_mod is not None and hasattr(_utils_mod, "hf_processor"):
        _utils_mod.hf_processor = _patched_hf_processor


def _patch_hf_tokenizer_for_qwen3_omni() -> None:
    """Load a separate chat template through verl when needed."""
    import functools
    import json
    import os

    try:
        import verl.utils.tokenizer as _vt
    except ImportError:
        return

    _original_hf_tokenizer = _vt.hf_tokenizer

    @functools.wraps(_original_hf_tokenizer)
    def _patched_hf_tokenizer(name_or_path: Any, *args: Any, **kwargs: Any) -> Any:
        tokenizer = _original_hf_tokenizer(name_or_path, *args, **kwargs)

        if getattr(tokenizer, "chat_template", None) is None and isinstance(name_or_path, str):
            chat_template_path = os.path.join(name_or_path, "chat_template.json")
            if os.path.exists(chat_template_path):
                try:
                    with open(chat_template_path) as f:
                        data = json.load(f)
                        chat_template = data.get("chat_template")
                        if chat_template:
                            tokenizer.chat_template = chat_template
                except (OSError, json.JSONDecodeError):
                    pass

        return tokenizer

    _vt.hf_tokenizer = _patched_hf_tokenizer

    # Refresh already-imported verl bindings.
    import sys

    for mod_name in list(sys.modules.keys()):
        if not mod_name.startswith("verl"):
            continue
        mod = sys.modules.get(mod_name)
        if (
            mod is not None
            and hasattr(mod, "hf_tokenizer")
            and mod.__dict__.get("hf_tokenizer") is _original_hf_tokenizer
        ):
            mod.hf_tokenizer = _patched_hf_tokenizer


def apply() -> None:
    """Apply all Qwen3-Omni patches once."""
    global _APPLIED
    if _APPLIED:
        return
    _register_qwen3_omni_automodel()
    _patch_hf_processor_for_qwen3_omni()
    _patch_hf_tokenizer_for_qwen3_omni()
    _APPLIED = True


def apply_on_driver() -> None:
    """Apply the same patches on the driver."""
    apply()


__all__ = ["apply", "apply_on_driver"]
