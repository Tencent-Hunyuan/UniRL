"""LoRA sync guard for Qwen3-Omni direct-TTS Phase 1."""

from __future__ import annotations

from typing import Any, Mapping

from unirl.distributed.weight_sync.lora.remote import RemoteLoraWeightSync

_CANONICAL_PREFIX = "talker."
_ENGINE_PREFIX = "base_model.model.talker."
_FORBIDDEN_COMPONENTS = (".code_predictor.", "thinker.", "code2wav.")


def validate_talker_lora_keys(
    tensors: Mapping[str, Any],
    *,
    engine_envelope: bool = False,
) -> None:
    """Reject empty, non-Talker, MTP, Thinker, or Code2Wav LoRA payloads."""

    if not isinstance(tensors, Mapping) or not tensors:
        raise RuntimeError("Talker LoRA sync requires a non-empty tensor mapping")
    prefix = _ENGINE_PREFIX if engine_envelope else _CANONICAL_PREFIX
    invalid = []
    for raw_name in tensors:
        name = str(raw_name)
        if not name.startswith(prefix):
            invalid.append(name)
            continue
        lowered = name.lower()
        if any(component in lowered for component in _FORBIDDEN_COMPONENTS):
            invalid.append(name)
            continue
        if not (name.endswith(".lora_A.weight") or name.endswith(".lora_B.weight")):
            invalid.append(name)
    if invalid:
        scope = "engine-envelope" if engine_envelope else "canonical"
        raise RuntimeError(
            "Phase-1 direct-TTS LoRA sync accepts only trainable non-MTP "
            f"Talker adapter tensors in {scope} form; invalid={invalid[:12]}"
        )


class Qwen3OmniTalkerLoraWeightSync(RemoteLoraWeightSync):
    """Remote sync that proves the extracted payload is Talker-layer0 only."""

    def _extract(self):
        lora_tensors, peft_config = super()._extract()
        validate_talker_lora_keys(lora_tensors)
        return lora_tensors, peft_config


__all__ = ["Qwen3OmniTalkerLoraWeightSync", "validate_talker_lora_keys"]
