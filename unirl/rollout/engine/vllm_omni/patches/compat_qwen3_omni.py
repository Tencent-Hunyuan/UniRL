"""Compatibility helpers for Qwen3-Omni on the pinned vLLM-Omni runtime."""

from __future__ import annotations

from functools import wraps
from typing import Any

_PATCH_SENTINEL = "_unirl_qwen3_omni_lora_compat"
_AUDIO_TRUNCATION_SENTINEL = "_unirl_audio_truncation_lengths_patched"


def patch_qwen3_omni_thinker_class(model_cls: type[Any]) -> None:
    """Backport Qwen3-Omni Thinker LoRA support from vLLM-Omni #3915."""
    if getattr(model_cls, _PATCH_SENTINEL, False) or getattr(model_cls, "supports_lora", False):
        return

    model_cls.supports_lora = True
    for name, default in (
        ("is_3d_moe_weight", False),
        ("is_non_gated_moe", False),
        ("embedding_modules", {}),
        ("lora_skip_prefixes", []),
    ):
        if not hasattr(model_cls, name):
            setattr(model_cls, name, default)

    packed_modules_mapping = dict(model_cls.packed_modules_mapping)
    packed_modules_mapping.setdefault("attn.qkv", ["attn.q", "attn.k", "attn.v"])
    model_cls.packed_modules_mapping = packed_modules_mapping

    original_init = model_cls.__init__

    @wraps(original_init)
    def _patched_init(self: Any, *, vllm_config: Any, prefix: str = "") -> None:
        model_config = vllm_config.model_config
        full_config = model_config.hf_config
        thinker_config = getattr(full_config, "thinker_config", None)
        if thinker_config is None:
            original_init(self, vllm_config=vllm_config, prefix=prefix)
            return

        model_config.hf_config = thinker_config
        try:
            original_init(self, vllm_config=vllm_config, prefix=prefix)
        finally:
            model_config.hf_config = full_config

    model_cls.__init__ = _patched_init

    original_get_mrope_input_positions = model_cls.get_mrope_input_positions

    @wraps(original_get_mrope_input_positions)
    def _patched_get_mrope_input_positions(
        self: Any,
        input_tokens: list[int],
        mm_features: list[Any],
        **kwargs: Any,
    ) -> Any:
        del kwargs
        return original_get_mrope_input_positions(self, input_tokens, mm_features)

    model_cls.get_mrope_input_positions = _patched_get_mrope_input_positions
    setattr(model_cls, _PATCH_SENTINEL, True)


def patch_qwen3_omni_audio_truncation(processor_cls: type[Any]) -> None:
    """Keep reconstructed audio lengths within truncated Whisper features."""
    if getattr(processor_cls, _AUDIO_TRUNCATION_SENTINEL, False):
        return

    import torch

    original = processor_cls._call_hf_processor

    @wraps(original)
    def _patched_call_hf_processor(
        self: Any,
        prompt: Any,
        mm_data: Any,
        mm_kwargs: Any,
        tok_kwargs: Any,
    ) -> Any:
        outputs = original(
            self,
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )
        audio_kwargs = mm_kwargs.get("audio_kwargs") or {}
        truncation = bool(mm_kwargs.get("truncation", audio_kwargs.get("truncation", False)))
        lengths = outputs.get("audio_feature_lengths")
        if not truncation or lengths is None:
            return outputs

        feature_extractor = self.info.get_feature_extractor(**mm_kwargs)
        max_frames = int(feature_extractor.n_samples // feature_extractor.hop_length)
        lengths_tensor = torch.as_tensor(lengths).clamp_max(max_frames)
        outputs["audio_feature_lengths"] = lengths_tensor
        old_mask = outputs.get("feature_attention_mask")
        if isinstance(old_mask, list):
            outputs["feature_attention_mask"] = [
                torch.ones(int(length), dtype=mask.dtype if isinstance(mask, torch.Tensor) else torch.float32)
                for length, mask in zip(lengths_tensor.tolist(), old_mask)
            ]
        elif isinstance(old_mask, torch.Tensor):
            positions = torch.arange(max_frames, device=old_mask.device)
            outputs["feature_attention_mask"] = (
                positions.unsqueeze(0) < lengths_tensor.to(old_mask.device).unsqueeze(1)
            ).to(old_mask.dtype)
        return outputs

    processor_cls._call_hf_processor = _patched_call_hf_processor
    setattr(processor_cls, _AUDIO_TRUNCATION_SENTINEL, True)


__all__ = ["patch_qwen3_omni_audio_truncation", "patch_qwen3_omni_thinker_class"]
