from __future__ import annotations

import json
import sys

import pytest

try:
    import torch
    from safetensors.torch import load_file
except ModuleNotFoundError:
    torch = None
    load_file = None

pytestmark = pytest.mark.skipif(torch is None, reason="torch is not installed")


class _FakeModel:
    def state_dict(self):
        return {
            "base.weight": torch.ones(2, 2),
            "layer.q_proj.lora_A.default.weight": torch.ones(1, 2),
            "layer.q_proj.lora_B.default.weight": torch.ones(2, 1),
            "layer.q_proj.lora_A.old.weight": torch.full((1, 2), 2.0),
        }


class _FakeCpuDTensor:
    def __init__(self):
        self.device = torch.device("cpu")
        self.moved_to_cuda = False

    def cuda(self):
        self.moved_to_cuda = True
        self.device = torch.device("cuda")
        return self

    def full_tensor(self):
        assert self.moved_to_cuda
        return torch.ones(1, 2)


def test_gather_lora_state_dict_filters_without_full_model_gather():
    from unirl.train.fsdp_utils import gather_lora_state_dict

    state = gather_lora_state_dict(_FakeModel())

    assert set(state) == {
        "layer.q_proj.lora_A.default.weight",
        "layer.q_proj.lora_B.default.weight",
        "layer.q_proj.lora_A.old.weight",
    }
    assert all(value.device.type == "cpu" for value in state.values())


def test_gather_lora_state_dict_moves_cpu_dtensor_to_cuda_before_collective(monkeypatch):
    from unirl.train.fsdp_utils import gather_lora_state_dict

    fake_dtensor = _FakeCpuDTensor()

    class Model:
        def state_dict(self):
            return {"layer.q_proj.lora_A.default.weight": fake_dtensor}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    state = gather_lora_state_dict(Model())

    assert fake_dtensor.moved_to_cuda
    assert set(state) == {"layer.q_proj.lora_A.default.weight"}


def test_export_adapter_state_dict_strips_adapter_name_and_adds_peft_prefix():
    from unirl.tools import export_adapter

    converted = export_adapter.export_adapter_state_dict(
        {
            "layer.q_proj.lora_A.default.weight": torch.ones(1, 2),
            "layer.q_proj.lora_B.default.weight": torch.ones(2, 1),
            "layer.q_proj.lora_A.old.weight": torch.zeros(1, 2),
        },
        adapter="default",
    )

    assert set(converted) == {
        "base_model.model.layer.q_proj.lora_A.weight",
        "base_model.model.layer.q_proj.lora_B.weight",
    }


def test_export_adapter_main_writes_peft_artifact(tmp_path, monkeypatch):
    from unirl.tools import export_adapter

    checkpoint = {
        "policy_state_dict": {
            "layer.q_proj.lora_A.default.weight": torch.ones(1, 2),
            "layer.q_proj.lora_B.default.weight": torch.ones(2, 1),
        },
        "lora_config": {
            "rank": 1,
            "alpha": 4,
            "target_modules": ["q_proj"],
            "dropout": 0.0,
            "bias": "none",
            "task_type": "FEATURE_EXTRACTION",
        },
    }
    checkpoint_path = tmp_path / "checkpoint.pt"
    output = tmp_path / "adapter"
    torch.save(checkpoint, checkpoint_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_adapter",
            "--checkpoint",
            str(checkpoint_path),
            "--base",
            "example/base",
            "--output",
            str(output),
        ],
    )

    export_adapter.main()

    weights = load_file(output / "adapter_model.safetensors")
    assert set(weights) == {
        "base_model.model.layer.q_proj.lora_A.weight",
        "base_model.model.layer.q_proj.lora_B.weight",
    }
    with open(output / "adapter_config.json") as f:
        config = json.load(f)
    assert config["base_model_name_or_path"] == "example/base"
    assert config["r"] == 1


def test_export_adapter_errors_for_missing_adapter():
    from unirl.tools import export_adapter

    with pytest.raises(SystemExit):
        export_adapter.export_adapter_state_dict(
            {"layer.q_proj.lora_A.default.weight": torch.ones(1, 2)},
            adapter="old",
        )
