import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

from unirl.rollout.engine.sglang.backends.http import HTTPBackend, asdict_drop_none
from unirl.rollout.engine.sglang.backends.native import NativeBackend


@dataclass
class _DataclassRequest:
    value: int
    optional: str | None = None


class _StructRequest:
    __struct_fields__ = ("value", "optional")

    def __init__(self, *, value, optional=None):
        self.value = value
        self.optional = optional


class _LoraRequest:
    def __init__(
        self,
        *,
        lora_name,
        config_dict,
        serialized_named_tensors,
    ):
        self.lora_name = lora_name
        self.config_dict = config_dict
        self.serialized_named_tensors = serialized_named_tensors


class _Serializer:
    @staticmethod
    def serialize(value, *, output_str):
        assert output_str is True
        return f"serialized:{len(value)}"


def test_http_request_conversion_supports_dataclass_and_msgspec_struct():
    assert asdict_drop_none(_DataclassRequest(value=3)) == {"value": 3}
    assert asdict_drop_none(_StructRequest(value=4)) == {"value": 4}


def test_http_lora_request_sends_one_payload_per_tp_rank():
    runtime = {
        "LoadLoRAAdapterFromTensorsReqInput": _LoraRequest,
        "MultiprocessingSerializer": _Serializer,
    }
    process = SimpleNamespace(is_alive=lambda: True)
    backend = HTTPBackend(
        process,
        "http://localhost",
        concurrency=1,
        tp_size=2,
        runtime=runtime,
    )
    captured = {}
    backend._post_struct = lambda path, req, operation: captured.update(
        path=path,
        request=req,
        operation=operation,
    )

    backend.set_lora(
        lora_name="adapter",
        lora_tensors={"a": object(), "b": object()},
        config_dict={"r": 8},
    )

    request = captured["request"]
    assert captured["path"] == "/load_lora_adapter_from_tensors"
    assert len(request.serialized_named_tensors) == 2
    assert request.serialized_named_tensors[0] == request.serialized_named_tensors[1]


class _Engine:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.loaded = None

    def load_lora_adapter_from_tensors(self, **kwargs):
        self.loaded = kwargs
        return SimpleNamespace(success=True)


def test_native_lora_uses_public_engine_api():
    engine = _Engine()
    backend = NativeBackend(engine, concurrency=1, runtime={})
    tensors = {"layer": object()}

    backend.set_lora(
        lora_name="adapter",
        lora_tensors=tensors,
        config_dict={"r": 8},
    )

    assert engine.loaded == {
        "lora_name": "adapter",
        "tensors": tensors,
        "config_dict": {"r": 8},
    }
    engine.loop.close()
