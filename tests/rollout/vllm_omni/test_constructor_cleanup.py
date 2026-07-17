from __future__ import annotations

import multiprocessing as mp
import os
from types import SimpleNamespace

import pytest

import unirl.rollout.engine.vllm_omni.backends.native as native_module
import unirl.rollout.engine.vllm_omni.engine as engine_module
import unirl.rollout.engine.vllm_omni.patches as patches_module


def test_backend_boot_hides_expandable_allocator_from_omni_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocator_env_seen_by_omni = {}

    class _Omni:
        def __init__(self, **_kwargs) -> None:
            self.stage_configs = []
            allocator_env_seen_by_omni.update(
                {name: os.environ.get(name) for name in native_module._ALLOCATOR_ENV_VARS}
            )

    monkeypatch.setattr(patches_module, "install", lambda: None)
    monkeypatch.setattr(mp, "set_start_method", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        native_module,
        "_import_omni_runtime",
        lambda: {"Omni": _Omni},
    )
    monkeypatch.setattr(native_module, "_resolve_stage_yaml", lambda *_args: "/fake/stages.yaml")
    monkeypatch.setenv("DIFFRL_OMNI_BOOT_SERIALIZE", "0")
    cuda_conf = (
        "max_split_size_mb:64,expandable_segments:True,"
        "garbage_collection_threshold:0.95,per_process_memory_fraction:0.90"
    )
    alloc_conf = "expandable_segments:true"
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", cuda_conf)
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", alloc_conf)

    native_module.VLLMOmniBackend.boot(
        {
            "model_path": "/fake/model",
            "stage_yaml": "stages.yaml",
            "stage_yaml_source": "local",
            "needs_driver_tokenizer": False,
        }
    )

    assert allocator_env_seen_by_omni == {
        "PYTORCH_CUDA_ALLOC_CONF": (
            "max_split_size_mb:64,garbage_collection_threshold:0.95,per_process_memory_fraction:0.90"
        ),
        "PYTORCH_ALLOC_CONF": None,
    }
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == cuda_conf
    assert os.environ["PYTORCH_ALLOC_CONF"] == alloc_conf


def test_omni_allocator_scope_restores_environment_after_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    cuda_conf = "expandable_segments:True,max_split_size_mb:64"
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", cuda_conf)

    with pytest.raises(RuntimeError, match="injected boot failure"):
        with native_module._without_expandable_segments_for_omni():
            assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:64"
            raise RuntimeError("injected boot failure")

    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == cuda_conf


def test_backend_boot_closes_omni_when_post_boot_setup_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    omni_instances = []

    class _Omni:
        def __init__(self, **_kwargs) -> None:
            self.stage_configs = []
            self.close_calls = 0
            omni_instances.append(self)

        def close(self) -> None:
            self.close_calls += 1

    class _Tokenizer:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            raise RuntimeError("injected tokenizer failure")

    monkeypatch.setattr(patches_module, "install", lambda: None)
    monkeypatch.setattr(mp, "set_start_method", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        native_module,
        "_import_omni_runtime",
        lambda: {"Omni": _Omni, "AutoTokenizer": _Tokenizer},
    )
    monkeypatch.setattr(native_module, "_resolve_stage_yaml", lambda *_args: "/fake/stages.yaml")
    monkeypatch.setenv("DIFFRL_OMNI_BOOT_SERIALIZE", "0")

    with pytest.raises(RuntimeError, match="injected tokenizer failure"):
        native_module.VLLMOmniBackend.boot(
            {
                "model_path": "/fake/model",
                "stage_yaml": "stages.yaml",
                "stage_yaml_source": "local",
                "needs_driver_tokenizer": True,
            }
        )

    assert len(omni_instances) == 1
    assert omni_instances[0].close_calls == 1


def test_rollout_constructor_shuts_down_booted_backend_on_setup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Adapter:
        lora_copy_transport = False

        def boot_kwargs(self):
            return {}

        def schedule_policy(self):
            raise RuntimeError("injected schedule failure")

    class _Config:
        modality = "test"

        def server_intent(self, **_kwargs):
            return {}

    class _Backend:
        def __init__(self) -> None:
            self.shutdown_calls = 0

        def shutdown(self) -> None:
            self.shutdown_calls += 1

    backend = _Backend()
    monkeypatch.setattr(engine_module, "get_adapter", lambda _modality: lambda *_args, **_kwargs: _Adapter())
    monkeypatch.setattr(
        engine_module,
        "VLLMOmniBackend",
        SimpleNamespace(boot=lambda _intent: backend),
    )

    with pytest.raises(RuntimeError, match="injected schedule failure"):
        engine_module.VLLMOmniRolloutEngine(
            _Config(),
            ports=SimpleNamespace(master_port=12345),
        )

    assert backend.shutdown_calls == 1
