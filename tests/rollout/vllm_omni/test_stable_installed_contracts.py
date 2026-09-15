from __future__ import annotations

import importlib
from importlib.metadata import version

import pytest

pytest.importorskip("vllm")
pytest.importorskip("vllm_omni")


def test_exact_stable_runtime_is_installed() -> None:
    assert version("vllm") == "0.28.0"
    assert version("vllm-omni") == "0.28.0"
    assert version("transformers") == "5.12.1"
    assert version("kernels") == "0.14.1"
    assert version("diffusers") == "0.40.0"


def test_real_stable_output_uses_flat_completion_fields() -> None:
    from vllm.outputs import CompletionOutput
    from vllm_omni.outputs import OmniRequestOutput

    from unirl.rollout.engine.vllm_omni.utils.tracks import _extract_completion, decoded_text_from_ar

    completion = CompletionOutput(
        index=0,
        text="answer",
        token_ids=[1, 2],
        cumulative_logprob=None,
        logprobs=None,
        finish_reason="stop",
    )
    output = OmniRequestOutput(
        request_id="0",
        stage_id=0,
        final_output_type="text",
        outputs=[completion],
    )

    assert decoded_text_from_ar([[output]]).texts == ["answer"]
    assert _extract_completion(output) == ([1, 2], None)


@pytest.mark.parametrize(
    "module_name",
    [
        "unirl.rollout.engine.vllm_omni.pipelines.sd3.pipeline",
        "unirl.rollout.engine.vllm_omni.pipelines.qwen_image.pipeline",
        "unirl.rollout.engine.vllm_omni.pipelines.hv15.pipeline",
        "unirl.rollout.engine.vllm_omni.pipelines.hi3.pipeline",
        "unirl.rollout.engine.vllm_omni.pipelines.bagel.pipeline",
        "unirl.rollout.engine.vllm_omni.worker.dit_extension",
        "unirl.rollout.engine.vllm_omni.worker.ar_extension",
    ],
)
def test_custom_pipeline_and_worker_modules_import(module_name: str) -> None:
    importlib.import_module(module_name)


def test_capture_plugin_patches_both_formatter_bindings() -> None:
    from vllm_omni.diffusion import diffusion_engine, output_formatter

    from unirl.rollout.engine.vllm_omni.plugin import register_capture_flush

    register_capture_flush()
    patched = output_formatter.format_diffusion_outputs
    assert patched is diffusion_engine.format_diffusion_outputs

    register_capture_flush()
    assert output_formatter.format_diffusion_outputs is patched


def test_diffusion_lora_loader_delegates_non_tensor_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager

    from unirl.rollout.engine.vllm_omni.patches.runtime import patch_dit_lora_loader

    request = object()
    expected = object()
    calls = []

    def original(manager, received):
        calls.append((manager, received))
        return expected

    monkeypatch.setattr(DiffusionLoRAManager, "_load_adapter", original)
    patch_dit_lora_loader()

    manager = object()
    assert DiffusionLoRAManager._load_adapter(manager, request) is expected
    assert calls == [(manager, request)]


def test_hi3_expert_mapping_tuple_is_unwrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm.model_executor import utils as vllm_utils

    from unirl.rollout.engine.vllm_omni.patches import compat_hi3_lora

    mapping = [("experts.0", "weight", 0, "w")]
    monkeypatch.setattr(vllm_utils, "get_moe_expert_mapping", lambda _model: (mapping, {"old": "new"}))
    monkeypatch.setattr(compat_hi3_lora, "_INSTALLED", False)

    compat_hi3_lora.install()

    assert vllm_utils.get_moe_expert_mapping(object()) == mapping


def test_diffusion_parameter_probe_reaches_wrapped_transformer() -> None:
    import torch

    from unirl.rollout.engine.vllm_omni.worker.ipc_receive_mixin import BucketedIPCReceiveMixin

    worker = object.__new__(BucketedIPCReceiveMixin)
    worker.worker = torch.nn.Module()
    transformer = torch.nn.Linear(4, 3)
    worker.model_runner = type("Runner", (), {"pipeline": type("Pipeline", (), {"transformer": transformer})()})()

    descriptions = worker._diffrl_describe_params()

    assert descriptions["weight"] == ((3, 4), "torch.float32")
