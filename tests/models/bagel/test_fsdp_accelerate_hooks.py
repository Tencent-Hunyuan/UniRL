from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from unirl.train.backend.base import LrSchedulerConfig, OptimizerConfig
from unirl.train.backend.fsdp.backend import FSDPBackend, _prepare_cpu_offload_model
from unirl.train.configs import FSDPConfig


def test_fsdp_backend_removes_accelerate_hooks_before_wrap(monkeypatch: pytest.MonkeyPatch) -> None:
    hooks = pytest.importorskip("accelerate.hooks")
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU())
    hooks.add_hook_to_module(model, hooks.AlignDevicesHook(execution_device=torch.device("cpu")))
    hooks.add_hook_to_module(model[0], hooks.AlignDevicesHook(execution_device=torch.device("cpu")))
    assert any(hasattr(module, "_hf_hook") for module in model.modules())

    bundle = SimpleNamespace(trainable_module=lambda: model)
    wrapped = False

    def fake_wrap(candidate: torch.nn.Module, **_: object) -> None:
        nonlocal wrapped
        wrapped = True
        assert candidate is model
        assert not any(hasattr(module, "_hf_hook") for module in candidate.modules())
        assert not any(hasattr(module, "_old_forward") for module in candidate.modules())

    backend_module = __import__("unirl.train.backend.fsdp.backend", fromlist=["unused"])
    monkeypatch.setattr(backend_module, "fsdp_wrap", fake_wrap)
    monkeypatch.setattr(backend_module, "load_trainable_weights", lambda *args, **kwargs: None)
    monkeypatch.setattr(backend_module, "apply_deferred_ops", lambda *args, **kwargs: None)
    monkeypatch.setattr(FSDPBackend, "_inject_structural", lambda *args, **kwargs: None)
    monkeypatch.setattr(FSDPBackend, "_finalize_construction", lambda *args, **kwargs: None)

    FSDPBackend(
        bundle=bundle,
        block_class_names=("Linear",),
        fsdp_cfg=FSDPConfig(),
        optimizer_cfg=OptimizerConfig(
            learning_rate=1e-4,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            weight_decay=0.0,
        ),
        scheduler_cfg=LrSchedulerConfig(type="constant", warmup_steps=0, total_steps=1),
        device=torch.device("cpu"),
    )

    assert wrapped
    torch.testing.assert_close(model(torch.ones(1, 2)), model._modules["0"](torch.ones(1, 2)).relu())


def test_cpu_offload_preparation_moves_eager_model_and_preserves_meta_model() -> None:
    class TrackingLinear(torch.nn.Linear):
        def __init__(self) -> None:
            super().__init__(2, 2)
            self.to_devices: list[torch.device] = []

        def to(self, *args, **kwargs):
            device = kwargs.get("device", args[0] if args else None)
            self.to_devices.append(torch.device(device))
            return super().to(*args, **kwargs)

    eager = TrackingLinear()
    _prepare_cpu_offload_model(eager)
    assert eager.to_devices == [torch.device("cpu")]
    assert {parameter.device.type for parameter in eager.parameters()} == {"cpu"}

    with torch.device("meta"):
        meta = torch.nn.Linear(2, 2)
    _prepare_cpu_offload_model(meta)
    assert {parameter.device.type for parameter in meta.parameters()} == {"meta"}


def test_cpu_offload_preparation_rejects_partially_meta_model() -> None:
    model = torch.nn.Module()
    model.register_parameter("materialized", torch.nn.Parameter(torch.ones(1)))
    model.register_parameter("deferred", torch.nn.Parameter(torch.empty(1, device="meta")))

    with pytest.raises(RuntimeError, match="partially-meta"):
        _prepare_cpu_offload_model(model)
