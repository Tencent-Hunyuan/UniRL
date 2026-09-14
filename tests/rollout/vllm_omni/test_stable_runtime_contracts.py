from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from unirl.rollout.engine.vllm_omni import patches, plugin
from unirl.rollout.engine.vllm_omni.backends import native
from unirl.rollout.engine.vllm_omni.backends.native import VLLMOmniBackend
from unirl.rollout.engine.vllm_omni.utils.tracks import _extract_completion, decoded_text_from_ar


class _Task:
    def __init__(self, *, task_id: str, level: int | None = None, tags=None) -> None:
        self.task_id = task_id
        self.level = level
        self.tags = tags


class _Engine:
    num_stages = 2

    def __init__(self) -> None:
        self.calls: list[tuple[int, str, tuple, dict | None]] = []

    def get_stage_metadata(self, stage_id: int):
        return SimpleNamespace(stage_type="llm" if stage_id == 0 else "diffusion")

    def collective_rpc(self, *, method: str, args=(), kwargs=None, stage_ids=None):
        stage_id = stage_ids[0]
        self.calls.append((stage_id, method, args, kwargs))
        if method in {"handle_sleep_task", "handle_wake_task"}:
            task = args[0]
            return [
                {
                    "status": "SUCCESS",
                    "stage_id": stage_id,
                    "task_id": task.task_id,
                    "rank": 0,
                }
            ]
        return [True]


def _backend(engine: _Engine) -> VLLMOmniBackend:
    runtime = {
        "OmniSleepTask": _Task,
        "OmniWakeTask": _Task,
    }
    return VLLMOmniBackend(
        SimpleNamespace(engine=engine),
        runtime,
        tokenizer=None,
        tp_per_stage={0: 1, 1: 1},
    )


def test_stable_outputs_are_read_directly() -> None:
    completion = SimpleNamespace(text="answer", token_ids=[1, 2], logprobs=None)
    output = SimpleNamespace(stage_id=0, final_output_type="text", outputs=[completion])

    texts = decoded_text_from_ar([[output]])
    tokens, logprobs = _extract_completion(output)

    assert texts.texts == ["answer"]
    assert tokens == [1, 2]
    assert logprobs is None


def test_sleep_and_wake_route_by_stage_type(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    engine = _Engine()
    backend = _backend(engine)

    backend.sleep_task()
    backend.wake_task()

    assert [(stage, method) for stage, method, _args, _kwargs in engine.calls] == [
        (0, "sleep"),
        (1, "handle_sleep_task"),
        (0, "wake_up"),
        (1, "handle_wake_task"),
    ]
    assert engine.calls[0][2] == (1, "abort")
    assert engine.calls[2][3] == {"tags": None}


def test_collective_rpc_rejects_unsupported_stage() -> None:
    backend = _backend(_Engine())
    with pytest.raises(RuntimeError, match="not installed"):
        backend._require_rpc_success(
            "update_weights",
            1,
            [{"supported": False, "error": "worker extension is not installed"}],
        )


def test_boot_passes_stable_deploy_config_and_reads_engine_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    class Omni:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)
            self.engine = SimpleNamespace(
                num_stages=1,
                stage_configs=[
                    {
                        "stage_id": 0,
                        "engine_args": {"tensor_parallel_size": 2},
                    }
                ],
            )

    monkeypatch.setattr(patches, "install", lambda: None)
    monkeypatch.setattr(plugin, "register_unirl_runtime", lambda: None)
    monkeypatch.setattr(
        native,
        "_import_omni_runtime",
        lambda: {
            "Omni": Omni,
            "OmniSleepTask": _Task,
            "OmniWakeTask": _Task,
        },
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    backend = VLLMOmniBackend.boot(
        {
            "model_path": "unused",
            "deploy_config": "sd35_t2i_rl.yaml",
            "enable_sleep_mode": False,
            "ports": None,
        }
    )

    assert captured["model"] == "unused"
    assert captured["deploy_config"].endswith("deploy_configs/sd35_t2i_rl.yaml")
    assert "stage_configs_path" not in captured
    assert backend.tp_per_stage() == {0: 2}
