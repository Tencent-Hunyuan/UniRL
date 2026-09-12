from __future__ import annotations

import threading
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch import nn

from unirl.distributed.weight_sync.full.nccl import NCCLWeightSync
from unirl.rollout.engine.native_sd3.config import NativeSD3EngineConfig
from unirl.rollout.engine.native_sd3.engine import NativeSD3RolloutEngine
from unirl.rollout.engine.native_sd3.quantization import (
    FP8Controller,
    RoutedTransformer,
    convert_transformer_for_fp8,
)
from unirl.types.sampling import DiffusionSamplingParams


class _TinyTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.register_buffer("position_table", torch.arange(4))
        self.register_buffer("scratch", torch.arange(2), persistent=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.proj(values)


class _FakeTELinear(nn.Linear):
    def __init__(self, in_features, out_features, *, bias, params_dtype, device):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=params_dtype)


def test_disabled_fp8_path_preserves_parameter_names_and_bf16_forward() -> None:
    config = NativeSD3EngineConfig(fp8_enabled=False, compile_model=False)
    controller = FP8Controller(config)
    model = _TinyTransformer().to(torch.bfloat16)
    expected = model(torch.ones(2, 4, dtype=torch.bfloat16))

    targets, report = convert_transformer_for_fp8(model, config=config, controller=controller)
    routed = RoutedTransformer(model, controller)
    with controller.rollout(mode="bf16", total_steps=1):
        actual = routed(torch.ones(2, 4, dtype=torch.bfloat16))

    assert report.replaced == ()
    assert set(targets) == {"proj.weight", "proj.bias", "position_table"}
    assert targets["position_table"] is model.position_table
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="full-weight export materializes tensors on CUDA")
def test_full_weight_export_matches_native_parameter_targets() -> None:
    config = NativeSD3EngineConfig(fp8_enabled=False, compile_model=False)
    model = _TinyTransformer().to(torch.bfloat16)
    targets, _ = convert_transformer_for_fp8(model, config=config, controller=FP8Controller(config))
    backend = SimpleNamespace(
        model=model,
        rollout_adapter_name="default",
        expert_weight_export_transform=lambda: None,
    )
    sync = NCCLWeightSync(backend=backend)
    assert {name for name, _ in sync._iter_full_tensors()} == set(targets)


def test_fp8_conversion_freezes_aligned_te_replacements() -> None:
    config = NativeSD3EngineConfig(fp8_enabled=False, fp8_min_dim=0, compile_model=False)
    config.fp8_enabled = True
    controller = FP8Controller.__new__(FP8Controller)
    controller.enabled = True
    controller.te = SimpleNamespace(Linear=_FakeTELinear)
    model = nn.Sequential(nn.Linear(16, 16, dtype=torch.bfloat16))
    _, report = convert_transformer_for_fp8(model, config=config, controller=controller)
    assert report.replaced == ("0",)
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_disabled_engine_rejects_fp8_request() -> None:
    controller = FP8Controller(NativeSD3EngineConfig(fp8_enabled=False, compile_model=False))
    try:
        with controller.rollout(mode="fp8", total_steps=1):
            pass
    except RuntimeError as exc:
        assert "fp8_enabled=false" in str(exc)
    else:
        raise AssertionError("expected disabled FP8 controller to reject an FP8 request")
    capabilities = NativeSD3EngineConfig.resolve_rollout_capabilities({"fp8_enabled": False})
    assert capabilities.rollout_precisions == frozenset({"bf16"})


def test_weight_bucket_is_fully_validated_before_copy() -> None:
    engine = NativeSD3RolloutEngine.__new__(NativeSD3RolloutEngine)
    first = torch.zeros(2, 2)
    second = torch.zeros(2, 2)
    engine._parameter_targets = {"first": first, "second": second}

    try:
        engine._resolve_weight_targets(
            [
                ("transformer.first", torch.ones(2, 2)),
                ("transformer.second", torch.ones(3, 2)),
            ]
        )
    except ValueError as exc:
        assert "shape mismatch" in str(exc)
    else:
        raise AssertionError("expected invalid bucket shape to fail")
    assert torch.equal(first, torch.zeros_like(first))


def test_weight_publication_blocks_generation_until_finished() -> None:
    engine = NativeSD3RolloutEngine.__new__(NativeSD3RolloutEngine)
    engine._weight_groups = {"weights": object()}
    engine._weight_group_timeouts = {"weights": 30.0}
    engine._failed_weight_groups = set()
    engine._generate_lock = threading.Lock()
    engine._weights_valid = True
    engine._parameter_targets = {"weight": torch.zeros(1)}
    engine._controller = type("_Controller", (), {"mark_weights_dirty": lambda self: None})()

    engine.begin_weights_update(group_name="weights")
    assert not engine._weights_valid
    engine._publication_names.add("weight")
    engine.finish_weights_update(group_name="weights")
    assert engine._weights_valid


def test_incomplete_and_duplicate_weight_publications_are_rejected() -> None:
    engine = NativeSD3RolloutEngine.__new__(NativeSD3RolloutEngine)
    engine._weight_groups = {"weights": object()}
    engine._weight_group_timeouts = {"weights": 30.0}
    engine._failed_weight_groups = set()
    engine._generate_lock = threading.Lock()
    engine._weights_valid = True
    engine._parameter_targets = {"first": torch.zeros(1), "second": torch.zeros(1)}
    engine._controller = type("_Controller", (), {"mark_weights_dirty": lambda self: None})()

    engine.begin_weights_update(group_name="weights")
    engine._publication_names.add("first")
    with pytest.raises(RuntimeError, match="publication is incomplete"):
        engine.finish_weights_update(group_name="weights")
    assert not engine._weights_valid

    with pytest.raises(ValueError, match="duplicate parameter"):
        engine._resolve_weight_targets(
            [("transformer.first", torch.ones(1))],
            previously_received={"first"},
        )


def test_weight_bucket_is_allocated_and_armed_before_sender_broadcast(monkeypatch) -> None:
    engine = NativeSD3RolloutEngine.__new__(NativeSD3RolloutEngine)
    engine.device = torch.device("cpu")
    engine._weight_groups = {"weights": object()}
    engine._weight_group_timeouts = {"weights": 30.0}
    engine._failed_weight_groups = set()
    engine._generate_lock = threading.Lock()
    engine._weights_valid = True
    engine._parameter_targets = {"weight": torch.zeros(2, 2)}
    engine._controller = type("_Controller", (), {"mark_weights_dirty": lambda self: None})()
    engine.begin_weights_update(group_name="weights")
    broadcasts = []

    class _Work:
        def wait(self, *, timeout):
            return True

    def _arm(tensor, **kwargs):
        tensor.fill_(2.0)
        broadcasts.append(tensor)
        return _Work()

    monkeypatch.setattr(
        dist,
        "broadcast",
        _arm,
    )

    with pytest.raises(KeyError, match="no target"):
        engine.prepare_weights_update(
            names=["transformer.missing"],
            dtypes=["torch.float32"],
            shapes=[[2, 2]],
            group_name="weights",
        )
    engine.prepare_weights_update(
        names=["transformer.weight"],
        dtypes=["torch.float32"],
        shapes=[[2, 2]],
        group_name="weights",
    )
    assert engine._prepared_bucket is not None
    assert len(broadcasts) == 1
    engine.update_weights_from_distributed(
        names=["transformer.weight"],
        dtypes=["torch.float32"],
        shapes=[[2, 2]],
        group_name="weights",
    )
    assert torch.equal(engine._parameter_targets["weight"], torch.full((2, 2), 2.0))


def test_failed_armed_receive_permanently_rejects_group_reuse(monkeypatch) -> None:
    engine = NativeSD3RolloutEngine.__new__(NativeSD3RolloutEngine)
    engine.device = torch.device("cpu")
    engine._weight_groups = {"weights": object()}
    engine._weight_group_timeouts = {"weights": 30.0}
    engine._failed_weight_groups = set()
    engine._generate_lock = threading.Lock()
    engine._weights_valid = True
    engine._parameter_targets = {"weight": torch.zeros(1)}
    engine._controller = type("_Controller", (), {"mark_weights_dirty": lambda self: None})()
    engine.begin_weights_update(group_name="weights")
    work = SimpleNamespace(wait=lambda **kwargs: False)
    monkeypatch.setattr(dist, "broadcast", lambda tensor, **kwargs: work)
    engine.prepare_weights_update(
        names=["transformer.weight"],
        dtypes=["torch.float32"],
        shapes=[[1]],
        group_name="weights",
    )
    with pytest.raises(TimeoutError, match="armed weight receive"):
        engine.update_weights_from_distributed(
            names=["transformer.weight"],
            dtypes=["torch.float32"],
            shapes=[[1]],
            group_name="weights",
        )
    assert "weights" in engine._failed_weight_groups
    with pytest.raises(RuntimeError, match="failed and cannot be reused"):
        engine.begin_weights_update(group_name="weights")


def test_failed_fp8_forward_keeps_cache_dirty_for_retry() -> None:
    controller = FP8Controller.__new__(FP8Controller)
    controller.config = SimpleNamespace(bf16_prefix_steps=0, bf16_suffix_steps=0)
    controller.enabled = True
    controller._mode = "fp8"
    controller._step = 0
    controller._total_steps = 1
    controller._weights_dirty = True
    controller._first_microbatch = False
    controller._fp8_active = False
    controller.te = SimpleNamespace(autocast=lambda **kwargs: nullcontext())
    controller.recipe = object()

    with pytest.raises(RuntimeError, match="synthetic"):
        with controller.transformer_forward():
            raise RuntimeError("synthetic")
    assert controller._weights_dirty
    assert controller._step == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"forward_batch_size": 1.5},
        {"fp8_enabled": "false"},
        {"bf16_prefix_steps": 0.5},
    ],
)
def test_native_sd3_config_rejects_coercible_types(kwargs: dict) -> None:
    with pytest.raises(TypeError):
        NativeSD3EngineConfig(**kwargs)


@pytest.mark.parametrize("value", [True, 1.5, "224"])
def test_reward_image_size_requires_a_positive_integer(value: object) -> None:
    with pytest.raises(ValueError):
        DiffusionSamplingParams(reward_image_size=value)
