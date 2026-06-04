"""WeightSync component tests — a recording fake of the seam, no host class.

The explicit ctor means the component constructs directly; the recorder
asserts the forwarded payloads, the versioned-nickname rotation, the
``lora_dirty`` lifecycle, and the two deliberate divergences from the
diffusion component (no ``target_modules`` on the wire; raw LoRA keys).
"""

from __future__ import annotations

import pytest
import torch
from conftest import RecordingBackend

from unirl.rollout.engine.sglang_v2.weight_sync import WeightSync


def make_sync(uses_lora: bool = True):
    backend = RecordingBackend()
    return WeightSync(backend, uses_lora=uses_lora), backend


# ---------------------------------------------------------------------------
# LoRA: versioned-nickname rotation + dirty lifecycle
# ---------------------------------------------------------------------------


def test_lora_nickname_rotates_per_push():
    sync, backend = make_sync()
    tensors = {"model.layers.0.q_proj.lora_A.weight": torch.zeros(2, 2)}
    sync.set_lora_from_tensors("default", tensors, peft_config={"r": 16})
    sync.set_lora_from_tensors("default", tensors)
    sync.set_lora_from_tensors("default", tensors)

    names = [kwargs["lora_name"] for name, kwargs in backend.calls if name == "set_lora"]
    assert names == ["default_v1", "default_v2", "default_v3"]
    assert sync.active_adapter == "default_v3"


def test_lora_tensors_and_peft_config_pass_through_raw():
    """LLM divergence: keys go to the wire RAW (no adapt_lora_for_sglang)."""
    sync, backend = make_sync()
    tensors = {"model.layers.0.q_proj.lora_A.weight": torch.zeros(2, 2)}
    peft = {"r": 16, "lora_alpha": 32, "target_modules": ["q_proj"]}
    sync.set_lora_from_tensors("default", tensors, peft_config=peft)

    _, kwargs = backend.calls[0]
    assert kwargs["lora_tensors"] is tensors  # untouched, same object
    assert kwargs["config_dict"] == peft


def test_lora_dirty_lifecycle():
    sync, _ = make_sync(uses_lora=True)
    # Fresh: in use but never pushed.
    assert sync.lora_dirty is True
    assert sync.active_adapter is None
    # Pushed: clean, adapter active.
    sync.set_lora_from_tensors("default", {"k": torch.zeros(1)})
    assert sync.lora_dirty is False
    assert sync.active_adapter == "default_v1"
    # Weights released (engine.sleep): pool gone — dirty again, no tagging.
    sync.mark_weights_released()
    assert sync.lora_dirty is True
    assert sync.active_adapter is None
    # Re-push rotates to a fresh version.
    sync.set_lora_from_tensors("default", {"k": torch.zeros(1)})
    assert sync.active_adapter == "default_v2"


def test_lora_dirty_always_false_without_lora():
    sync, _ = make_sync(uses_lora=False)
    assert sync.lora_dirty is False
    sync.mark_weights_released()
    assert sync.lora_dirty is False


# ---------------------------------------------------------------------------
# Tensor-bag + NCCL forwards
# ---------------------------------------------------------------------------


def test_update_from_tensor_forwards_without_target_modules():
    sync, backend = make_sync()
    sync.update_weights_from_tensor(serialized_named_tensors=["blob"], flush_cache=False)
    name, kwargs = backend.calls[0]
    assert name == "update_from_tensor"
    assert kwargs == {
        "serialized_named_tensors": ["blob"],
        "load_format": None,
        "flush_cache": False,
    }
    assert "target_modules" not in kwargs


def test_update_from_tensor_rejects_empty():
    sync, _ = make_sync()
    with pytest.raises(ValueError, match="non-empty"):
        sync.update_weights_from_tensor(serialized_named_tensors=[])


def test_distributed_update_cleans_dtype_strings():
    sync, backend = make_sync()
    sync.update_weights_from_distributed(
        names=["w1", "w2"],
        dtypes=["torch.bfloat16", "float32"],
        shapes=[[2, 2], [3]],
        group_name="g",
    )
    _, kwargs = backend.calls[0]
    # sglang expects bare dtype strings; already-clean ones pass through.
    assert kwargs["dtypes"] == ["bfloat16", "float32"]
    assert kwargs["shapes"] == [[2, 2], [3]]
    assert "target_modules" not in kwargs


def test_distributed_update_rejects_empty_names():
    sync, _ = make_sync()
    with pytest.raises(ValueError, match="non-empty"):
        sync.update_weights_from_distributed(names=[], dtypes=[], shapes=[], group_name="g")


def test_nccl_group_init_and_destroy_forward():
    sync, backend = make_sync()
    sync.init_weights_update_group(
        master_address="10.0.0.1",
        master_port="29500",  # str coercion mirrors the predecessor
        rank_offset=1,
        world_size=2,
        group_name="llm_sync",
    )
    sync.destroy_weights_update_group(group_name="llm_sync")

    name0, kwargs0 = backend.calls[0]
    assert name0 == "init_weights_group"
    assert kwargs0["master_port"] == 29500 and isinstance(kwargs0["master_port"], int)
    assert kwargs0["backend"] == "nccl"
    name1, kwargs1 = backend.calls[1]
    assert (name1, kwargs1) == ("destroy_weights_group", {"group_name": "llm_sync"})
