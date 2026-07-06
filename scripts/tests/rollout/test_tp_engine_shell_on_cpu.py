"""Tier 0 smoke tests — SGLangRolloutEngine tp_rank>0 no-op shell (no GPU).

The riskiest correctness invariant of the colocate TP design: a tp_rank>0
worker must construct WITHOUT booting SGLang, reserving ports, or touching a
GPU, and every rollout verb on it must be a safe no-op. If this regresses, an
8-GPU tp=2 run would try to spawn 4 duplicate multi-GPU servers and OOM.

These tests never require a GPU: the shell path returns before any CUDA work.

Run:  pytest scripts/tests/rollout/test_tp_engine_shell_on_cpu.py
"""

from __future__ import annotations

import pytest

from unirl.rollout.engine.sglang.config import SGLangEngineConfig
from unirl.rollout.engine.sglang.engine import SGLangRolloutEngine


def _shell(tp_rank: int = 1, tp_size: int = 2) -> SGLangRolloutEngine:
    cfg = SGLangEngineConfig(pretrained_model_ckpt_path="/tmp/model", model_family="text", tp_size=tp_size)
    return SGLangRolloutEngine(
        config=cfg,
        rank=tp_rank,
        tp_rank=tp_rank,
        tp_size=tp_size,
        tp_device_ids=[0, 1],
    )


def test_shell_constructs_without_backend_or_gpu():
    eng = _shell()
    assert eng._is_tp_zero is False
    assert eng._backend is None
    assert eng._weight_sync is None
    assert eng.adapter is None


def test_shell_lifecycle_verbs_are_noops():
    eng = _shell()
    # None of these may raise, and none may touch a (nonexistent) backend.
    eng.sleep()
    eng.wake_up()
    eng.onload_weights()
    assert eng.health_check() is True
    assert eng.lora_dirty is False
    eng.shutdown()  # idempotent, no backend to close


def test_shell_weight_sync_verbs_are_noops():
    eng = _shell()
    eng.update_weights_from_tensor(serialized_named_tensors=["x"], load_format="flattened_bucket")
    eng.init_weights_update_group(
        master_address="127.0.0.1", master_port=1234, rank_offset=1, world_size=3, group_name="g"
    )
    eng.update_weights_from_distributed(names=["w"], dtypes=["float16"], shapes=[[2, 2]], group_name="g")
    eng.destroy_weights_update_group(group_name="g")
    eng.set_lora_from_tensors("adapter", {})


def test_shell_double_shutdown_is_safe():
    eng = _shell()
    eng.shutdown()
    eng.shutdown()


@pytest.mark.parametrize("tp_rank", [1, 2, 3])
def test_all_nonzero_tp_ranks_are_shells(tp_rank):
    eng = _shell(tp_rank=tp_rank, tp_size=4)
    assert eng._is_tp_zero is False
    assert eng._backend is None
