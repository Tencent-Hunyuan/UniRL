"""Bare-engine sequencing — the logic the core actually owns.

A real ``SGLangRolloutEngine`` instance built without ``__init__`` (no server
spawn), seam + components wired by hand: the staged sleep/wake two-flag
machine (flush ordering, the weights-released event, the post-sync re-offload
path), ``onload_weights`` idempotence, and generate's ``lora_path`` stamping +
``track_prefix`` absorption.
"""

from __future__ import annotations

import torch
from conftest import RecordingBackend, StubTokenizer, make_raw

from unirl.rollout.engine.sglang.adapters import TextLMAdapter
from unirl.rollout.engine.sglang.config import SGLangEngineConfig
from unirl.rollout.engine.sglang.engine import SGLangRolloutEngine
from unirl.rollout.engine.sglang.weight_sync import WeightSync
from unirl.types.primitives import Texts
from unirl.types.rollout_req import RolloutReq


def make_bare_engine(*, generate_results=None, uses_lora=True):
    """A real engine instance with the seam faked — no ctor, no server."""
    engine = object.__new__(SGLangRolloutEngine)
    config = SGLangEngineConfig(pretrained_model_ckpt_path="stub/model")
    backend = RecordingBackend(generate_results=generate_results)
    engine.cfg = config
    engine.rank = 0
    engine._device = torch.device("cpu")
    engine._is_offloaded = False
    engine._weights_onloaded_for_sync = False
    engine._backend = backend
    engine._weight_sync = WeightSync(backend, uses_lora=uses_lora)
    engine.adapter = TextLMAdapter(config, None, tokenizer=StubTokenizer())
    return engine, backend


# ---------------------------------------------------------------------------
# sleep — staged release + the visible flush / weights-released lines
# ---------------------------------------------------------------------------


def test_sleep_full_flushes_then_releases_and_marks():
    engine, backend = make_bare_engine()
    engine._weight_sync.set_lora_from_tensors("default", {"k": torch.zeros(1)})
    backend.calls.clear()

    engine.sleep()

    assert backend.names() == ["flush_cache", "release_memory"]  # flush BEFORE release
    assert backend.calls[1][1] == {"tags": None}  # full release
    assert engine.is_offloaded is True
    assert engine._weights_onloaded_for_sync is False
    # Weights went with the release -> the LoRA pool is gone.
    assert engine.lora_dirty is True


def test_sleep_weights_only_skips_flush():
    engine, backend = make_bare_engine()
    engine.sleep(tags=["weights"])
    assert backend.names() == ["release_memory"]  # no kv_cache targeted -> no flush
    assert backend.calls[0][1] == {"tags": ["weights"]}
    assert engine._weight_sync.lora_dirty is True


def test_sleep_kv_only_flushes_but_keeps_lora():
    engine, backend = make_bare_engine()
    engine._weight_sync.set_lora_from_tensors("default", {"k": torch.zeros(1)})
    backend.calls.clear()

    engine.sleep(tags=["kv_cache"])

    assert backend.names() == ["flush_cache", "release_memory"]
    # Weights stayed resident -> the adapter survives.
    assert engine.lora_dirty is False


def test_sleep_while_offloaded_releases_onloaded_weights_only():
    """The post-sync re-offload path: sleep -> onload_weights -> sleep."""
    engine, backend = make_bare_engine()
    engine.sleep()
    engine.onload_weights()
    backend.calls.clear()

    engine.sleep()

    assert backend.names() == ["release_memory"]  # weights-only, no flush
    assert backend.calls[0][1] == {"tags": ["weights"]}
    assert engine._weights_onloaded_for_sync is False


def test_sleep_while_offloaded_without_onload_is_noop():
    engine, backend = make_bare_engine()
    engine.sleep()
    backend.calls.clear()

    engine.sleep()

    assert backend.calls == []


# ---------------------------------------------------------------------------
# wake_up / onload_weights — staged resume
# ---------------------------------------------------------------------------


def test_wake_up_when_not_offloaded_is_noop():
    engine, backend = make_bare_engine()
    engine.wake_up()
    assert backend.calls == []


def test_wake_up_full_resumes_and_clears_flags():
    engine, backend = make_bare_engine()
    engine.sleep()
    backend.calls.clear()

    engine.wake_up()

    assert backend.names() == ["resume_memory"]
    assert backend.calls[0][1] == {"tags": None}  # full resume
    assert engine.is_offloaded is False
    assert engine._weights_onloaded_for_sync is False


def test_wake_up_after_onload_resumes_remaining_regions():
    engine, backend = make_bare_engine()
    engine.sleep()
    engine.onload_weights()
    backend.calls.clear()

    engine.wake_up()

    # Weights already resident: only kv_cache + cuda_graph come back.
    assert backend.calls[0][1] == {"tags": ["kv_cache", "cuda_graph"]}
    assert engine.is_offloaded is False


def test_wake_up_weights_tag_sets_sync_flag_without_full_wake():
    engine, backend = make_bare_engine()
    engine.sleep()
    backend.calls.clear()

    engine.wake_up(tags=["weights"])

    assert backend.calls[0][1] == {"tags": ["weights"]}
    assert engine.is_offloaded is True  # partial resume keeps the offload flag
    assert engine._weights_onloaded_for_sync is True


def test_onload_weights_resumes_once_and_absorbs_track_prefix():
    engine, backend = make_bare_engine()
    engine.sleep()
    backend.calls.clear()

    engine.onload_weights(track_prefix="ar")
    engine.onload_weights()  # second call: already onloaded -> no-op

    assert backend.names() == ["resume_memory"]
    assert backend.calls[0][1] == {"tags": ["weights"]}
    assert engine._weights_onloaded_for_sync is True


def test_onload_weights_noop_when_not_offloaded():
    engine, backend = make_bare_engine()
    engine.onload_weights()
    assert backend.calls == []


# ---------------------------------------------------------------------------
# generate — lora_path stamping
# ---------------------------------------------------------------------------


def make_req(prompts):
    return RolloutReq(
        sample_ids=[f"p{i}" for i in range(len(prompts))],
        group_ids=[f"g{i}" for i in range(len(prompts))],
        primitives={"text": Texts(texts=list(prompts))},
    )


def test_generate_stamps_active_lora_path():
    engine, backend = make_bare_engine(generate_results=[make_raw("out")])
    engine._weight_sync.set_lora_from_tensors("default", {"k": torch.zeros(1)})
    backend.calls.clear()

    resp = engine.generate(make_req(["hi"]))

    payloads = backend.calls[0][1]["requests"]
    assert payloads[0]["lora_path"] == "default_v1"
    assert resp.tracks["ar"].decoded.texts == ["out"]


def test_generate_omits_lora_path_without_adapter():
    engine, backend = make_bare_engine(generate_results=[make_raw("out")])
    engine.generate(make_req(["hi"]))
    payloads = backend.calls[0][1]["requests"]
    assert "lora_path" not in payloads[0]


def test_generate_omits_lora_path_after_weights_released():
    engine, backend = make_bare_engine(generate_results=[make_raw("out")])
    engine._weight_sync.set_lora_from_tensors("default", {"k": torch.zeros(1)})
    engine.sleep()
    engine.wake_up()
    backend.calls.clear()

    engine.generate(make_req(["hi"]))

    payloads = backend.calls[0][1]["requests"]
    assert "lora_path" not in payloads[0]  # stale adapter never referenced


# ---------------------------------------------------------------------------
# Weight-sync forwards — track_prefix absorbed, target_modules dropped
# ---------------------------------------------------------------------------


def test_forwards_absorb_track_prefix_and_target_modules():
    engine, backend = make_bare_engine()
    engine.update_weights_from_tensor(
        serialized_named_tensors=["blob"],
        target_modules=["transformer"],  # diffusion-style default: must be dropped
        track_prefix="ar",
    )
    engine.update_weights_from_distributed(
        names=["w"],
        dtypes=["torch.float32"],
        shapes=[[1]],
        group_name="g",
        target_modules=["transformer"],
        track_prefix="ar",
    )
    engine.init_weights_update_group(
        master_address="a",
        master_port=1,
        rank_offset=0,
        world_size=1,
        group_name="g",
        track_prefix="ar",
    )
    engine.destroy_weights_update_group(group_name="g", track_prefix="ar")

    for _, kwargs in backend.calls:
        assert "track_prefix" not in kwargs
        assert "target_modules" not in kwargs


def test_health_check_true_while_offloaded():
    engine, backend = make_bare_engine()
    engine.sleep()
    backend.calls.clear()
    assert engine.health_check() is True
    assert backend.calls == []  # no ping while offloaded
    engine.wake_up()
    backend.calls.clear()
    assert engine.health_check() is True
    assert backend.names() == ["ping"]
