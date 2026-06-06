"""CPU tests for the ``WeightSync`` component + the engine's offload lifecycle.

The seam is a recording fake; no SGLang, no GPU. Two layers:

- **component**: ``WeightSync`` constructed directly (explicit ctor — no host class
  to fake): forwarding + defaults, LoRA strip/alpha/nickname stability, the dirty
  flag, checksum reshaping.
- **engine orchestration**: a bare real engine (``object.__new__``) wired with the
  fake seam + a real ``WeightSync``: sleep/wake state, the weights-released event
  (sleep → ``lora_dirty``), and one forward smoke for the frozen-surface wiring.

Dispatch-marker guards live in ``test_engine.py`` (they assert on the real class).
"""

from __future__ import annotations

import pytest
import torch

from unirl.rollout.engine.sglang_diffusion.engine import SGLangDiffusionRolloutEngine
from unirl.rollout.engine.sglang_diffusion.weight_sync import WeightSync


class FakeBackend:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def record(**kw):
            self.calls.append((name, kw))
            if name == "weights_checksum":
                return {"transformer.blk0": "abcd", "transformer.blk1": "ef01"}
            return None

        return record

    def named(self, name):
        return [kw for n, kw in self.calls if n == name]


def _ws(backend=None):
    return WeightSync(
        backend or FakeBackend(),
        pipeline_prefix="transformer.",
        target_modules=["transformer"],
        uses_lora=True,
    )


def _bare_engine():
    """Real engine, no __init__ (which would spawn SGLang); seam + component wired."""
    e = object.__new__(SGLangDiffusionRolloutEngine)
    e._backend = FakeBackend()
    e._weight_sync = _ws(e._backend)
    e._is_offloaded = False
    return e


# --------------------------------------------------------------------------- #
# component: tensor-bag + NCCL forwarding
# --------------------------------------------------------------------------- #


def test_update_from_tensor_forwards_with_default_targets():
    ws = _ws()
    ws.update_weights_from_tensor(serialized_named_tensors=["blob"])
    call = ws._backend.named("update_from_tensor")[0]
    assert call["serialized_named_tensors"] == ["blob"]
    assert call["target_modules"] == ["transformer"]  # default from the ctor spec


def test_update_from_tensor_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        _ws().update_weights_from_tensor(serialized_named_tensors=[])


def test_nccl_three_phase_forwards():
    ws = _ws()
    ws.init_weights_update_group(
        master_address="127.0.0.1", master_port=40000, rank_offset=0,
        world_size=2, group_name="g",
    )
    ws.update_weights_from_distributed(
        names=["w"], dtypes=["float32"], shapes=[[2, 2]], group_name="g",
    )
    ws.destroy_weights_update_group(group_name="g")
    assert [n for n, _ in ws._backend.calls] == [
        "init_weights_group", "update_from_distributed", "destroy_weights_group"
    ]
    assert ws._backend.named("update_from_distributed")[0]["target_modules"] == ["transformer"]


def test_update_from_distributed_rejects_empty_names():
    with pytest.raises(ValueError, match="non-empty"):
        _ws().update_weights_from_distributed(
            names=[], dtypes=[], shapes=[], group_name="g"
        )


# --------------------------------------------------------------------------- #
# component: LoRA — prefix strip + alpha inject + stable nickname + dirty flag
# --------------------------------------------------------------------------- #


def _lora_tensors():
    t = torch.zeros(2, 2)
    return {
        "transformer.attn.to_q.lora_A.weight": t,
        "transformer.attn.to_q.lora_B.weight": t,
    }


def test_set_lora_strips_prefix_injects_alpha_keeps_nickname_stable():
    ws = _ws()
    ws.set_lora_from_tensors("default", _lora_tensors(), peft_config={"lora_alpha": 16})
    call = ws._backend.named("set_lora")[0]
    assert call["lora_nickname"] == "default"
    sent = call["lora_tensors"]
    assert "attn.to_q.lora_A.weight" in sent  # prefix stripped
    assert "attn.to_q.alpha" in sent          # alpha injected
    assert ws._active_adapter == "default"

    # second push re-uses the SAME nickname: SGLang's diffusion registry clears
    # and replaces the entry in place, so fresh weights are served. A versioned
    # nickname per push (the sglang_llm rotation) leaks one never-evicted
    # GPU-resident adapter copy per sync instead.
    ws.set_lora_from_tensors("default", _lora_tensors(), peft_config={"lora_alpha": 16})
    assert ws._backend.named("set_lora")[1]["lora_nickname"] == "default"


def test_lora_dirty_lifecycle():
    ws = _ws()
    assert ws.lora_dirty is True  # uses_lora, nothing loaded yet
    ws.set_lora_from_tensors("default", _lora_tensors(), peft_config={"lora_alpha": 16})
    assert ws.lora_dirty is False
    ws.mark_weights_released()  # the event the engine's sleep() fires
    assert ws.lora_dirty is True


def test_checksum_reshaped_to_vllm_omni_shape():
    out = _ws().loaded_param_checksums(names=["transformer.blk0", "transformer.blk1"])
    assert out == {0: [{"transformer.blk0": "abcd", "transformer.blk1": "ef01"}]}


# --------------------------------------------------------------------------- #
# engine orchestration: sleep/wake + the weights-released event + forward wiring
# --------------------------------------------------------------------------- #


def test_sleep_releases_with_tags_and_marks_lora_dirty():
    e = _bare_engine()
    e._weight_sync._lora_loaded = True  # as if an adapter had been pushed
    e.sleep()
    call = e._backend.named("release_memory")[0]
    assert list(call["tags"]) == ["transformer", "vae", "text_encoder"]
    assert list(call["cpu_backup_tags"]) == ["vae", "text_encoder"]
    assert e.is_offloaded is True
    assert e.lora_dirty is True  # sleep fired mark_weights_released()


def test_wake_up_resumes_only_when_offloaded():
    e = _bare_engine()
    e.wake_up()  # not offloaded → no-op
    assert e._backend.named("resume_memory") == []
    e.sleep()
    e.wake_up()
    assert len(e._backend.named("resume_memory")) == 1
    assert e.is_offloaded is False


def test_onload_weights_wakes():
    e = _bare_engine()
    e.sleep()
    e.onload_weights()
    assert e.is_offloaded is False
    assert len(e._backend.named("resume_memory")) == 1


def test_engine_forward_reaches_backend():
    # One smoke for the frozen-surface forwarding (engine → component → seam);
    # the per-method behavior is covered by the component tests above.
    e = _bare_engine()
    e.update_weights_from_tensor(serialized_named_tensors=["blob"], track_prefix="ignored")
    call = e._backend.named("update_from_tensor")[0]
    assert call["serialized_named_tensors"] == ["blob"]
    assert call["target_modules"] == ["transformer"]
