"""Adapter registry + config validation for the ``vllm_omni_v2`` engine."""

from __future__ import annotations

import pytest

from unirl.rollout.engine.vllm_omni_v2.adapters import (
    get_adapter,
    register_adapter,
    registered_adapters,
)
from unirl.rollout.engine.vllm_omni_v2.config import VLLMOmniV2EngineConfig

ALL_MODALITIES = ("hi3_ar_recaption", "hi3_dit_recaption", "hi3_i2t", "hi3_it2i", "hi3_t2i", "hi3_t2t", "hv15_t2v", "sd3_t2i")


def test_all_eight_modalities_registered():
    assert registered_adapters() == ALL_MODALITIES


def test_get_adapter_unknown_key_raises():
    with pytest.raises(Exception, match="unknown modality"):
        get_adapter("nope")


def test_duplicate_registration_raises():
    with pytest.raises(Exception, match="already registered"):

        @register_adapter("hi3_t2i")
        class Dup:  # pragma: no cover - registration must fail first
            pass


def test_config_normalizes_and_validates_modality():
    cfg = VLLMOmniV2EngineConfig(model_path="/x", modality="  SD3_T2I ")
    assert cfg.modality == "sd3_t2i"
    with pytest.raises(Exception, match="modality must be one of"):
        VLLMOmniV2EngineConfig(model_path="/x", modality="not_a_modality")


def test_topology_knobs_match_v1_frozensets():
    """One assertion per v1 engine.py frozenset, against the live registry."""
    by_knob = {
        # v1 _DIT_BEARING_MODALITIES (engine.py:107)
        "needs_sigmas": {"hi3_t2i", "hi3_it2i", "sd3_t2i", "hi3_dit_recaption", "hv15_t2v"},
        # v1 _HI3_MODALITIES (engine.py:101)
        "ar_lora_passthrough": {"hi3_t2i", "hi3_it2i", "hi3_i2t", "hi3_t2t", "hi3_ar_recaption"},
        # v1 _HI3_MULTI_GPU_MODALITIES (engine.py:130)
        "clear_cuda_visible": {"hi3_t2i", "hi3_it2i", "hi3_i2t", "hi3_t2t", "hi3_ar_recaption", "hi3_dit_recaption"},
        # v1 wake-up byte-copy branch (engine.py:674)
        "lora_copy_transport": {"hi3_ar_recaption", "hi3_dit_recaption"},
        # v1 tokenizer load gate, inverted (engine.py:322)
        "needs_driver_tokenizer": set(ALL_MODALITIES) - {"sd3_t2i", "hv15_t2v"},
    }
    for knob, expected in by_knob.items():
        actual = {m for m in ALL_MODALITIES if getattr(get_adapter(m), knob)}
        assert actual == expected, f"{knob}: {actual} != {expected}"


def test_omni_mode_matches_v1():
    # v1 engine.py:377: mode="text-to-image" for these four only.
    expected = {"hi3_t2i", "hi3_it2i", "sd3_t2i", "hi3_dit_recaption"}
    actual = {m for m in ALL_MODALITIES if get_adapter(m).omni_mode == "text-to-image"}
    assert actual == expected
    assert get_adapter("hv15_t2v").omni_mode is None
