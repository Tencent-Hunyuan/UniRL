"""Native wire tests — the in-process impl's pure helpers, CPU-only.

The gating test: ``backends.native`` must import without sglang installed (the
sglang import is lazy, inside ``_import_sglang_engine`` only). The rest
exercises ``payload_to_generate_kwargs`` (the /generate payload → Engine
kwargs mapping, incl. the unknown-key refusal) and ``_check_result`` (the
native twin of the HTTP impl's ``_check_update_response``, absorbing the three
native result shapes).
"""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest


def test_native_module_imports_without_sglang():
    """If this fails, the impl leaked a top-level runtime import."""
    assert importlib.util.find_spec("sglang") is None, (
        "this CPU suite assumes sglang is NOT installed — the import-hygiene assertion below would be vacuous otherwise"
    )
    import unirl.rollout.engine.sglang.backends.native  # noqa: F401


from unirl.rollout.engine.sglang.backends.native import (  # noqa: E402
    NativeBackend,
    payload_to_generate_kwargs,
)

# ---------------------------------------------------------------------------
# payload_to_generate_kwargs — the /generate payload → Engine kwargs mapping
# ---------------------------------------------------------------------------


def test_text_payload_maps_to_prompt():
    kwargs = payload_to_generate_kwargs({"text": "hi", "sampling_params": {"n": 1}})
    assert kwargs == {"prompt": "hi", "sampling_params": {"n": 1}}


def test_input_ids_payload_passes_through():
    kwargs = payload_to_generate_kwargs(
        {"input_ids": [1, 2, 3], "sampling_params": {"n": 2}, "return_logprob": True, "logprob_start_len": 0}
    )
    assert kwargs == {
        "input_ids": [1, 2, 3],
        "sampling_params": {"n": 2},
        "return_logprob": True,
        "logprob_start_len": 0,
    }


def test_vlm_payload_keeps_image_data_and_lora_path():
    """The VLM shape: templated text + base64 image, plus the LoRA stamp."""
    kwargs = payload_to_generate_kwargs(
        {
            "text": "templated <img>",
            "image_data": "base64...",
            "sampling_params": {"n": 1},
            "return_logprob": False,
            "logprob_start_len": 0,
            "lora_path": "default_v3",
        }
    )
    assert kwargs["prompt"] == "templated <img>"
    assert kwargs["image_data"] == "base64..."
    assert kwargs["lora_path"] == "default_v3"
    assert "text" not in kwargs


def test_unknown_payload_key_raises():
    """HTTP would forward an unknown key to the server; a silent drop here
    would be an invisible divergence — refuse loudly instead."""
    with pytest.raises(ValueError, match="unmapped /generate payload keys.*top_logprobs_num"):
        payload_to_generate_kwargs({"text": "hi", "top_logprobs_num": 5})


def test_payload_is_not_mutated():
    payload = {"text": "hi", "sampling_params": {"n": 1}}
    payload_to_generate_kwargs(payload)
    assert payload == {"text": "hi", "sampling_params": {"n": 1}}


# ---------------------------------------------------------------------------
# _check_result — the three native result shapes
# ---------------------------------------------------------------------------


def test_check_result_tuple_failure_raises_with_operation():
    with pytest.raises(RuntimeError, match=r"NativeBackend\.set_lora failed: no pool"):
        NativeBackend._check_result((False, "no pool"), "set_lora")


def test_check_result_tuple_success_passes():
    NativeBackend._check_result((True, "ok"), "update_from_tensor")


def test_check_result_dict_failure_raises():
    with pytest.raises(RuntimeError, match="boom"):
        NativeBackend._check_result({"success": False, "error_message": "boom"}, "release_memory")


def test_check_result_reqoutput_failure_raises():
    result = SimpleNamespace(success=False, error_message=None, message="group missing")
    with pytest.raises(RuntimeError, match="group missing"):
        NativeBackend._check_result(result, "destroy_weights_group")


def test_check_result_absent_success_assumes_ok():
    """Parity with the HTTP checker: no success field means ok."""
    NativeBackend._check_result(None, "resume_memory")
    NativeBackend._check_result({"message": "fine"}, "resume_memory")
    NativeBackend._check_result(SimpleNamespace(message="fine"), "resume_memory")
