"""Config tests — server_intent precedence, ports, validation, family derivation."""

from __future__ import annotations

import pytest

from unirl.rollout.engine.sglang_v2.config import SGLangV2EngineConfig, SGLangV2Ports


def make_config(**kwargs):
    kwargs.setdefault("pretrained_model_ckpt_path", "stub/model")
    return SGLangV2EngineConfig(**kwargs)


PORTS = SGLangV2Ports(server_port=31000, nccl_port=31001)


# ---------------------------------------------------------------------------
# server_intent — the 4-layer precedence
# ---------------------------------------------------------------------------


def test_server_intent_defaults():
    intent = make_config().server_intent(ports=PORTS)
    assert intent["model_path"] == "stub/model"
    assert intent["host"] == "0.0.0.0"  # bind-all so cross-node peers connect
    assert intent["tp_size"] == 1
    assert intent["mem_fraction_static"] == 0.88
    assert intent["port"] == 31000
    assert intent["nccl_port"] == 31001


def test_server_intent_escape_hatch_flows_but_loses_to_typed_fields():
    cfg = make_config(
        tp_size=2,
        engine_kwargs={
            "mem_fraction_static": 0.3,  # escape hatch beats the default
            "skip_server_warmup": True,  # non-typed key flows through
            "enable_lora": True,
            "tp_size": 8,  # typed field must win
        },
    )
    intent = cfg.server_intent(ports=PORTS)
    assert intent["mem_fraction_static"] == 0.3
    assert intent["skip_server_warmup"] is True
    assert intent["enable_lora"] is True
    assert intent["tp_size"] == 2


def test_server_intent_ports_beat_everything():
    cfg = make_config(engine_kwargs={"port": 1234, "nccl_port": 5678})
    intent = cfg.server_intent(ports=PORTS)
    assert intent["port"] == 31000
    assert intent["nccl_port"] == 31001


def test_server_intent_adapter_extras_beat_escape_hatch():
    cfg = make_config(engine_kwargs={"context_length": 2048})
    intent = cfg.server_intent(ports=PORTS, extra={"context_length": 8192})
    assert intent["context_length"] == 8192


def test_server_intent_host_typed_field():
    intent = make_config(host="10.0.0.5").server_intent(ports=PORTS)
    assert intent["host"] == "10.0.0.5"


# ---------------------------------------------------------------------------
# Ports
# ---------------------------------------------------------------------------


def test_ports_reserve_distinct_in_range():
    ports = SGLangV2Ports.reserve()
    assert ports.server_port != ports.nccl_port
    assert 1 <= ports.server_port <= 65535
    assert 1 <= ports.nccl_port <= 65535


def test_ports_from_ports_round_trip():
    ports = SGLangV2Ports.from_ports([30001, 30002])
    assert (ports.server_port, ports.nccl_port) == (30001, 30002)


def test_ports_reject_duplicates():
    with pytest.raises(ValueError, match="distinct"):
        SGLangV2Ports(server_port=30001, nccl_port=30001)


# ---------------------------------------------------------------------------
# Validation + adapter-family derivation
# ---------------------------------------------------------------------------


def test_validation_errors():
    with pytest.raises(ValueError, match="pretrained_model_ckpt_path"):
        SGLangV2EngineConfig()
    with pytest.raises(ValueError, match="tp_size"):
        make_config(tp_size=0)
    with pytest.raises(ValueError, match="concurrency"):
        make_config(concurrency=0)
    with pytest.raises(ValueError, match="max_new_tokens"):
        make_config(max_new_tokens=0)
    with pytest.raises(ValueError, match="temperature"):
        make_config(temperature=0.0)
    with pytest.raises(ValueError, match="top_p"):
        make_config(top_p=1.5)


def test_backend_default_normalization_and_validation():
    assert make_config().backend == "http"
    assert make_config(backend="NATIVE ").backend == "native"
    with pytest.raises(ValueError, match="backend"):
        make_config(backend="grpc")


def test_model_family_derived_from_image_token():
    assert make_config().model_family == "text"
    assert make_config(image_token="<img>").model_family == "vlm"


def test_model_family_explicit_override_and_normalization():
    assert make_config(model_family="VLM").model_family == "vlm"
    # Explicit override wins over the image_token heuristic.
    assert make_config(image_token="<img>", model_family="text").model_family == "text"


def test_model_family_validated_against_registry():
    with pytest.raises(ValueError, match="model_family"):
        make_config(model_family="nope")


def test_v1_recipe_shaped_config_constructs():
    """The parity-recipe promise: a v1-shaped field set needs no new keys."""
    cfg = SGLangV2EngineConfig(
        pretrained_model_ckpt_path="Qwen/Qwen3-4B-Base",
        tp_size=1,
        max_new_tokens=8192,
        temperature=1.0,
        top_p=0.9,
        concurrency=16,
        samples_pre_expanded=True,
        system_instruction="/no_think",
        chat_template_kwargs={"enable_thinking": False},
        engine_kwargs={"mem_fraction_static": 0.6, "skip_server_warmup": True},
    )
    assert cfg.model_family == "text"
    intent = cfg.server_intent(ports=PORTS)
    assert intent["mem_fraction_static"] == 0.6
