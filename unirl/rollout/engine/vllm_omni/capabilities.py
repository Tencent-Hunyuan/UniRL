"""Fail-closed probes for optional vLLM-Omni runtime contracts."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from importlib import metadata
from typing import Callable, FrozenSet

DIRECT_TTS_CONTRACT_VERSION = 1
DIRECT_TTS_RUNTIME_MODULE = "vllm_omni.model_executor.models.qwen3_omni.qwen3_omni"
DIRECT_TTS_CAPABILITY_ATTR = "UNIRL_DIRECT_TTS_ROLLOUT_CAPABILITIES"
DIRECT_TTS_REQUIRED_CAPABILITIES: FrozenSet[str] = frozenset(
    {
        "direct_tts_prefix_without_thinker_forward",
        "layer0_processed_logprobs",
        "residual_codes_15",
        "prefix_ids_echo",
        "speaker_id_echo",
        "behavior_sampling_echo",
        "codec_eos_status",
        "code2wav_audio_24khz",
    }
)


@dataclass(frozen=True)
class DirectTTSRuntimeCapability:
    engine_version: str
    contract_version: int
    capabilities: FrozenSet[str]


def require_direct_tts_runtime(
    *,
    import_module: Callable[[str], object] = importlib.import_module,
    distribution_version: Callable[[str], str] = metadata.version,
) -> DirectTTSRuntimeCapability:
    """Require the patched direct-TTS output contract before engine spawn.

    Version numbers alone are deliberately insufficient: upstream vLLM-Omni
    0.20.0 has Qwen3-Omni's spoken-response Thinker→Talker pipeline, but no
    direct-TTS prefix input and no RL trajectory export. A compatible build
    must advertise the exact contract on the Qwen3-Omni model module.
    """

    try:
        engine_version = str(distribution_version("vllm-omni"))
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "qwen3_omni_talker requires a patched vllm-omni runtime; the "
            "'vllm-omni' distribution is not installed. Upstream 0.20.0 is "
            "not sufficient because it only exposes the spoken-response "
            "Thinker→Talker handoff."
        ) from exc

    try:
        module = import_module(DIRECT_TTS_RUNTIME_MODULE)
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "qwen3_omni_talker cannot import the Qwen3-Omni runtime module "
            f"{DIRECT_TTS_RUNTIME_MODULE!r} from vllm-omni {engine_version}."
        ) from exc

    advertised = getattr(module, DIRECT_TTS_CAPABILITY_ATTR, None)
    if not isinstance(advertised, dict):
        raise RuntimeError(
            f"vllm-omni {engine_version} lacks {DIRECT_TTS_CAPABILITY_ATTR}. "
            "This build cannot provide direct-TTS prefix execution plus the "
            "lossless layer0/residual/logprob/audio rollout contract; refusing "
            "to start instead of falling back to spoken-response or fabricated data."
        )
    try:
        contract_version = int(advertised["contract_version"])
        capabilities = frozenset(str(value) for value in advertised["capabilities"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"vllm-omni {engine_version} advertises a malformed "
            f"{DIRECT_TTS_CAPABILITY_ATTR} payload: {advertised!r}."
        ) from exc
    if contract_version != DIRECT_TTS_CONTRACT_VERSION:
        raise RuntimeError(
            f"vllm-omni {engine_version} direct-TTS contract version "
            f"{contract_version} != required {DIRECT_TTS_CONTRACT_VERSION}."
        )
    missing = sorted(DIRECT_TTS_REQUIRED_CAPABILITIES - capabilities)
    if missing:
        raise RuntimeError(
            f"vllm-omni {engine_version} direct-TTS runtime is missing required "
            f"capabilities: {missing}."
        )
    return DirectTTSRuntimeCapability(
        engine_version=engine_version,
        contract_version=contract_version,
        capabilities=capabilities,
    )


__all__ = [
    "DIRECT_TTS_CAPABILITY_ATTR",
    "DIRECT_TTS_CONTRACT_VERSION",
    "DIRECT_TTS_REQUIRED_CAPABILITIES",
    "DirectTTSRuntimeCapability",
    "require_direct_tts_runtime",
]
