from __future__ import annotations

import multiprocessing as mp
from types import SimpleNamespace

import pytest

import unirl.rollout.engine.vllm_omni.backends.native as native_module
import unirl.rollout.engine.vllm_omni.engine as engine_module
import unirl.rollout.engine.vllm_omni.patches as patches_module


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
