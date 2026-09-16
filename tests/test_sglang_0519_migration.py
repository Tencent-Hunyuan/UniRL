from dataclasses import dataclass
from types import SimpleNamespace

from unirl.rollout.engine.sglang_diffusion.backends.native import SGLangBackend
from unirl.rollout.engine.sglang_diffusion.config import (
    SGLangDiffusionEngineConfig,
)


@dataclass
class _TensorRequest:
    serialized_named_tensors: list
    target_modules: list
    load_format: str | None = None
    weight_update_mode: str | None = None
    lora_alpha: int | None = None
    lora_rank: int | None = None


class _SchedulerClient:
    def __init__(self):
        self.requests = []

    def forward(self, request):
        self.requests.append(request)
        return SimpleNamespace(
            output={"success": True, "message": "ok"},
            error=None,
        )


class _Serializer:
    @staticmethod
    def serialize(value):
        return ("serialized", value)


def _backend():
    client = _SchedulerClient()
    runtime = {
        "UpdateWeightFromTensorReqInput": _TensorRequest,
        "MultiprocessingSerializer": _Serializer,
        "sync_scheduler_client": client,
    }
    return SGLangBackend(None, runtime, SimpleNamespace()), client


def test_tensor_update_uses_native_request_and_preserves_flush_cache():
    backend, client = _backend()

    backend.update_from_tensor(
        serialized_named_tensors=["payload"],
        target_modules=["transformer"],
        load_format="flattened_bucket",
        flush_cache=False,
    )

    request = client.requests[0]
    assert request.serialized_named_tensors == ["payload"]
    assert request.target_modules == ["transformer"]
    assert request.load_format == "flattened_bucket"
    assert request.flush_cache is False


def test_lora_update_uses_native_lora_merge_mode():
    backend, client = _backend()
    tensors = {"layer.lora_A.weight": object()}

    backend.set_lora(
        lora_tensors=tensors,
        target_modules=["transformer"],
        lora_alpha=32,
        lora_rank=16,
    )

    request = client.requests[0]
    assert request.serialized_named_tensors == [("serialized", tensors)]
    assert request.target_modules == ["transformer"]
    assert request.weight_update_mode == "lora_merge"
    assert request.lora_alpha == 32
    assert request.lora_rank == 16


def test_lora_rollout_defaults_to_upstream_dynamic_mode():
    config = SGLangDiffusionEngineConfig(model_family="sd3")
    model_config = SimpleNamespace(
        pretrained_model_ckpt_path="/model",
        use_lora=True,
        lora_target_modules=None,
    )

    intent = config.server_intent(model_config=model_config, ports=None)

    assert intent["lora_merge_mode"] == "dynamic"
