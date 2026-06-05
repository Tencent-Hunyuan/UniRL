"""Boot-time resolution — config → ServerArgs intent.

The legacy 33000/100/+11/+23 port math and the MASTER_PORT save/restore dance are
gone: the engine reserves a typed ``SGLangDiffusionPorts`` and
``SGLangDiffusionEngineConfig.server_intent`` spells all three ports into the
intent — including ``master_port``, the spawned workers' dist init, which the
pinned fork reads from ServerArgs (no env manipulation anywhere).
"""

from __future__ import annotations

from types import SimpleNamespace

from unirl.rollout.engine.sglang_diffusion.config import (
    SGLangDiffusionEngineConfig,
    SGLangDiffusionPorts,
)


def _cfg(**over):
    return SGLangDiffusionEngineConfig(sampling=None, **over)


def _model_config(**over):
    base = dict(
        pretrained_model_ckpt_path="/ckpt",
        use_lora=False,
        lora_target_modules=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_ports_overlay_server_scheduler_and_master():
    ports = SGLangDiffusionPorts(server_port=40000, scheduler_port=40011, master_port=40023)
    intent = _cfg(tp_size=2).server_intent(model_config=_model_config(), ports=ports)
    assert intent["port"] == 40000
    assert intent["scheduler_port"] == 40011
    # master_port rides the typed ServerArgs path (the fork's dist init reads
    # ServerArgs.master_port, which otherwise self-settles to a random scanned port).
    assert intent["master_port"] == 40023
    assert intent["model_path"] == "/ckpt"
    assert intent["num_gpus"] == 1
    assert intent["tp_size"] == 2


def test_lora_intent_when_enabled():
    ports = SGLangDiffusionPorts(server_port=1, scheduler_port=2, master_port=3)
    mc = _model_config(use_lora=True, lora_target_modules=["attn.to_q"])
    intent = _cfg().server_intent(model_config=mc, ports=ports)
    assert intent["lora_merge_mode"] == "online"
    assert intent["lora_target_modules"] == ["attn.to_q"]


def test_engine_kwargs_passthrough_and_extra_override():
    ports = SGLangDiffusionPorts(server_port=1, scheduler_port=2, master_port=3)
    cfg = _cfg(engine_kwargs={"mem_fraction_static": 0.8})
    intent = cfg.server_intent(
        model_config=_model_config(), ports=ports, extra={"model_path": "/override"}
    )
    assert intent["mem_fraction_static"] == 0.8       # escape-hatch passthrough
    assert intent["model_path"] == "/override"        # adapter extra wins over typed


def test_ports_win_over_engine_kwargs():
    ports = SGLangDiffusionPorts(server_port=40000, scheduler_port=40011, master_port=40023)
    cfg = _cfg(engine_kwargs={"port": 1, "scheduler_port": 2, "master_port": 3})
    intent = cfg.server_intent(model_config=_model_config(), ports=ports)
    assert intent["port"] == 40000
    assert intent["scheduler_port"] == 40011
    assert intent["master_port"] == 40023


def test_remote_mode_uses_cfg_ports():
    cfg = _cfg(local_mode=False, host="10.0.0.1", port=9000, scheduler_port=9011)
    intent = cfg.server_intent(model_config=_model_config(), ports=None)
    assert intent["host"] == "10.0.0.1"
    assert intent["port"] == 9000
    assert intent["scheduler_port"] == 9011
    # No reserved set in remote mode → no master_port intent (the external
    # server already settled its own).
    assert "master_port" not in intent
