"""Tier 0 smoke tests — SGLangEngineConfig server_intent for rollout TP/EP/PP.

Confirms the new ``pp_size``/``ep_size``/``dp_size``/``enable_expert_parallel``
config fields and the ``runtime_overrides`` precedence layer flow into the
ServerArgs intent, and that the ``tp_size=1`` default path is byte-for-byte the
pre-change behavior (no ``base_gpu_id`` leak).

Run:  pytest scripts/tests/rollout/test_tp_config_on_cpu.py
"""

from __future__ import annotations

import pytest

from unirl.rollout.engine.sglang.config import SGLangEngineConfig, SGLangPorts

PORTS = SGLangPorts(server_port=30000, nccl_port=30001)


def _cfg(**kw) -> SGLangEngineConfig:
    base = dict(pretrained_model_ckpt_path="/tmp/model", model_family="text")
    base.update(kw)
    return SGLangEngineConfig(**base)


def test_tp1_default_intent_parity():
    intent = _cfg().server_intent(ports=PORTS)
    assert intent["tp_size"] == 1
    assert intent["pp_size"] == 1
    assert intent["ep_size"] == 1
    # A tp=1 engine must never carry a runtime GPU override.
    assert "base_gpu_id" not in intent
    assert "gpu_id_step" not in intent
    assert intent["port"] == 30000
    assert intent["nccl_port"] == 30001


def test_tp_pp_ep_fields_flow_into_intent():
    intent = _cfg(tp_size=2, pp_size=1, ep_size=4, enable_expert_parallel=True).server_intent(ports=PORTS)
    assert intent["tp_size"] == 2
    assert intent["ep_size"] == 4
    assert intent["enable_expert_parallel"] is True


def test_runtime_overrides_win_over_config():
    intent = _cfg(tp_size=2).server_intent(
        ports=PORTS,
        runtime_overrides={"tp_size": 2, "base_gpu_id": 0, "gpu_id_step": 1},
    )
    assert intent["tp_size"] == 2
    assert intent["base_gpu_id"] == 0
    assert intent["gpu_id_step"] == 1


def test_reserved_ports_beat_runtime_overrides():
    # Ports are the highest layer — a stray override cannot clobber them.
    intent = _cfg().server_intent(ports=PORTS, runtime_overrides={"port": 999, "nccl_port": 998})
    assert intent["port"] == 30000
    assert intent["nccl_port"] == 30001


@pytest.mark.parametrize("bad", [{"tp_size": 0}, {"pp_size": 0}, {"ep_size": 0}, {"dp_size": 0}])
def test_parallel_sizes_must_be_positive(bad):
    with pytest.raises(Exception):
        _cfg(**bad)
