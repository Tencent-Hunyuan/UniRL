"""Environment-only configuration available in every spawned vLLM process."""

from __future__ import annotations

import os
from dataclasses import dataclass

PUBLIC_REFERENCE = "public_reference"
QWEN3_MOE_MODEL = "qwen3_moe_30b_a3b"
REQUIRED_PATCHES = ("common", QWEN3_MOE_MODEL)


@dataclass(frozen=True)
class PluginConfig:
    profile: str
    model: str
    patches: tuple[str, ...]
    strict: bool


def _required_environment(name: str) -> str:
    try:
        value = os.environ[name].strip()
    except KeyError as error:
        raise RuntimeError(f"{name} is required when UNIRL_PARITY_ENABLE=1") from error
    if not value:
        raise RuntimeError(f"{name} must be non-empty when UNIRL_PARITY_ENABLE=1")
    return value


def load_config() -> PluginConfig:
    profile = _required_environment("UNIRL_PARITY_PROFILE").lower()
    if profile != PUBLIC_REFERENCE:
        raise ValueError(f"unknown UNIRL_PARITY_PROFILE={profile!r}")
    model = _required_environment("UNIRL_PARITY_MODEL").lower()
    if model != QWEN3_MOE_MODEL:
        raise ValueError(f"unsupported UNIRL_PARITY_MODEL={model!r}")
    raw_patches = _required_environment("UNIRL_PARITY_PATCHES")
    patch_values = tuple(value.strip() for value in raw_patches.split(","))
    if any(not value for value in patch_values) or len(patch_values) != len(set(patch_values)):
        raise ValueError("UNIRL_PARITY_PATCHES must contain unique, non-empty patch names")
    if patch_values != REQUIRED_PATCHES:
        raise ValueError(
            f"UNIRL_PARITY_PATCHES must enable the complete ordered contract {REQUIRED_PATCHES!r}; got {patch_values!r}"
        )
    patches = patch_values
    raw_strict = _required_environment("UNIRL_PARITY_STRICT")
    if raw_strict != "1":
        raise ValueError("UNIRL_PARITY_STRICT must be exactly '1' for the public-reference contract")
    strict = True
    return PluginConfig(profile=profile, model=model, patches=patches, strict=strict)


__all__ = [
    "PUBLIC_REFERENCE",
    "QWEN3_MOE_MODEL",
    "REQUIRED_PATCHES",
    "PluginConfig",
    "load_config",
]
