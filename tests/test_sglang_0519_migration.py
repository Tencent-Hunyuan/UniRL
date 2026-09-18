from dataclasses import dataclass
from types import SimpleNamespace

import torch

from unirl.rollout.engine.sglang_diffusion._patches import (
    patch_gpu_worker,
    patch_rollout_trajectory,
)
from unirl.rollout.engine.sglang_diffusion.backends.native import (
    SGLangBackend,
    _strip_flattened_bucket_module_prefix,
)
from unirl.rollout.engine.sglang_diffusion.config import (
    SGLangDiffusionEngineConfig,
)
from unirl.rollout.engine.sglang_diffusion.weight_sync import (
    _partition_lora_tensors,
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
    def serialize(value, output_str=False):
        del output_str
        return ("serialized", value)

    @staticmethod
    def deserialize(value):
        if isinstance(value, tuple) and value[0] == "payload":
            return value[1]
        return {"metadata": []}


def _backend():
    client = _SchedulerClient()
    runtime = {
        "UpdateWeightFromTensorReqInput": _TensorRequest,
        "MultiprocessingSerializer": _Serializer,
        "sync_scheduler_client": client,
    }
    return SGLangBackend(None, runtime, SimpleNamespace()), client


def test_diffusion_server_args_can_omit_srt_memory_saver_knobs():
    assert patch_gpu_worker._memory_saver_options(SimpleNamespace()) == (False, True)
    assert patch_gpu_worker._memory_saver_options(SimpleNamespace(enable_memory_saver=True, pin_cpu_memory=False)) == (
        True,
        False,
    )


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


def test_tensor_update_strips_legacy_pipeline_prefix_for_upstream_module_loader():
    backend, client = _backend()
    payload = {
        "flattened_tensor": object(),
        "metadata": [
            SimpleNamespace(name="transformer.pos_embed.proj.weight"),
            SimpleNamespace(name="transformer.proj_out.weight"),
        ],
    }

    backend.update_from_tensor(
        serialized_named_tensors=[("payload", payload)],
        target_modules=["transformer"],
        load_format="flattened_bucket",
        flush_cache=True,
    )

    request = client.requests[0]
    normalized = request.serialized_named_tensors[0][1]
    assert [item.name for item in normalized["metadata"]] == [
        "pos_embed.proj.weight",
        "proj_out.weight",
    ]


def test_tensor_prefix_strip_is_noop_for_relative_or_multi_module_payload():
    relative = {"metadata": [SimpleNamespace(name="proj_out.weight")]}
    multi = {"metadata": [SimpleNamespace(name="transformer.proj_out.weight")]}

    assert not _strip_flattened_bucket_module_prefix(relative, ["transformer"])
    assert not _strip_flattened_bucket_module_prefix(multi, ["transformer", "transformer_2"])
    assert relative["metadata"][0].name == "proj_out.weight"
    assert multi["metadata"][0].name == "transformer.proj_out.weight"


def test_lora_update_uses_native_lora_merge_mode():
    backend, client = _backend()
    tensors = {"layer.lora_A.weight": object()}

    backend.set_lora(
        lora_tensors=tensors,
        target_module="transformer",
        lora_alpha=32,
        lora_rank=16,
    )

    request = client.requests[0]
    assert request.serialized_named_tensors == [("serialized", list(tensors.items()))]
    assert request.target_modules == ["transformer"]
    assert request.weight_update_mode == "lora_merge"
    assert request.lora_alpha == 32
    assert request.lora_rank == 16


def test_lora_rollout_defaults_to_upstream_dynamic_mode():
    config = SGLangDiffusionEngineConfig(
        model_family="sd3",
        target_modules=("sglang_q", "sglang_v"),
    )
    model_config = SimpleNamespace(
        pretrained_model_ckpt_path="/model",
        use_lora=True,
        lora_target_modules=("training_q",),
    )

    intent = config.server_intent(model_config=model_config, ports=None)

    assert intent["lora_merge_mode"] == "dynamic"
    assert intent["lora_target_modules"] == ["sglang_q", "sglang_v"]


def test_dual_transformer_lora_payload_is_partitioned_per_upstream_request():
    high = object()
    low = object()

    groups = _partition_lora_tensors(
        {
            "blocks.0.lora_A.weight": high,
            "transformer_2.blocks.0.lora_A.weight": low,
        },
        ["transformer", "transformer_2"],
    )

    assert groups == {
        "transformer": {"blocks.0.lora_A.weight": high},
        "transformer_2": {"blocks.0.lora_A.weight": low},
    }


@dataclass
class _DitTrajectory:
    latents: torch.Tensor | None = None
    timesteps: torch.Tensor | None = None
    sigmas: torch.Tensor | None = None


@dataclass
class _DebugTensors:
    rollout_variance_noises: torch.Tensor | None = None
    rollout_prev_sample_means: torch.Tensor | None = None
    rollout_noise_std_devs: torch.Tensor | None = None
    rollout_model_outputs: torch.Tensor | None = None


@dataclass
class _RolloutData:
    rollout_log_probs: torch.Tensor | None = None
    rollout_debug_tensors: _DebugTensors | None = None
    denoising_env: object | None = None
    dit_trajectory: _DitTrajectory | None = None


def test_grouped_rollout_preserves_latest_scheduler_sigmas(monkeypatch):
    monkeypatch.setattr(
        patch_rollout_trajectory,
        "_rl_dataclasses",
        lambda: (_RolloutData, _DitTrajectory, _DebugTensors),
    )
    sigmas = torch.tensor([1.0, 0.5, 0.0])
    batches = [
        SimpleNamespace(
            rollout_trajectory_data=_RolloutData(
                rollout_log_probs=torch.tensor([[float(index)]]),
                dit_trajectory=_DitTrajectory(
                    latents=torch.tensor([[float(index)]]),
                    timesteps=torch.tensor([1000.0, 500.0]),
                    sigmas=sigmas,
                ),
            )
        )
        for index in range(2)
    ]

    merged = patch_rollout_trajectory._concat_rollout_trajectory_data(batches)
    sliced = patch_rollout_trajectory._slice_rollout_trajectory_keepdim(merged, 1)

    assert merged.dit_trajectory.sigmas is sigmas
    assert sliced.dit_trajectory.sigmas is sigmas
    assert sliced.dit_trajectory.latents.tolist() == [[1.0]]
