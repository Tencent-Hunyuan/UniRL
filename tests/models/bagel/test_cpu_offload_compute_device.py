from __future__ import annotations

import gc
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from unirl.algorithms.bagel_flow_unigrpo import BagelFlowUniGRPO
from unirl.algorithms.base import AlgorithmStepResult
from unirl.models.bagel.ar import BagelARStage
from unirl.models.bagel.conditions import BagelARConditions
from unirl.models.bagel.diffusion import BagelDiffusionStage
from unirl.models.bagel.rl_ops import prefill_prompt_text
from unirl.models.types.replay_result import ReplayResult
from unirl.types.segments import TextSegment, make_image_segment


class _BoundaryTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.context_weight = nn.Parameter(torch.tensor(0.4))
        self.policy_weight = nn.Parameter(torch.tensor(0.7))


class _BoundaryStage:
    detach_forward_kwargs = staticmethod(BagelDiffusionStage.detach_forward_kwargs)

    def __init__(self, transformer: _BoundaryTransformer) -> None:
        self.model = SimpleNamespace(device=torch.device("cpu"), transformer=transformer)
        self.context_builds = 0
        self.replay_calls = 0
        self.predict_calls = 0
        self.replay_contexts: list[torch.Tensor] = []
        self.predict_contexts: list[torch.Tensor] = []
        self.last_velocity: torch.Tensor | None = None

    def build_forward_kwargs(self, conditions, *, params, device):
        del params, device
        self.context_builds += 1
        if "forward_kwargs" in conditions:
            return conditions["forward_kwargs"]
        scale = float(conditions.get("context_scale", 1.0))
        return {"context": self.model.transformer.context_weight * scale}

    def _velocity(self, forward_kwargs, segment) -> torch.Tensor:
        x_t = segment.latents_at(0)[0]
        velocity = self.model.transformer.policy_weight * x_t + forward_kwargs["context"]
        self.last_velocity = velocity
        return velocity

    @staticmethod
    def _result(velocity: torch.Tensor) -> ReplayResult:
        return ReplayResult(
            log_probs=velocity.mean().reshape(1, 1),
            prev_sample_means=velocity.reshape(1, 1, *velocity.shape),
        )

    def replay(self, conditions, *, segment, params, step_indices):
        del params, step_indices
        forward_kwargs = self.build_forward_kwargs(conditions, params=None, device=torch.device("cpu"))
        velocity = self._velocity(forward_kwargs, segment)
        return self._result(velocity)

    def replay_from_forward_kwargs_with_velocities(self, forward_kwargs, *, segment, params, step_indices):
        del params, step_indices
        self.replay_calls += 1
        self.replay_contexts.append(forward_kwargs["context"])
        velocity = self._velocity(forward_kwargs, segment)
        return self._result(velocity), [velocity]

    def predict_velocity_at(self, forward_kwargs, *, sample, sigma, params):
        del sigma, params
        self.predict_calls += 1
        self.predict_contexts.append(forward_kwargs["context"])
        return self.model.transformer.policy_weight * sample + forward_kwargs["context"]


def _boundary_segment() -> object:
    return make_image_segment(
        latents=torch.tensor([[[[0.3]], [[0.2]]]], dtype=torch.float32),
        sigmas=torch.tensor([0.8, 0.4], dtype=torch.float32),
        indices=torch.tensor([0, 1], dtype=torch.long),
        sde_indices=torch.tensor([0], dtype=torch.long),
        # On-policy for context_scale=1.5: 0.7 * 0.3 + 0.4 * 1.5 = 0.81.
        # Keeping the ratio inside the clip window makes the gradient-boundary
        # assertion exercise the context path instead of a clipped constant.
        sde_logp=torch.tensor([[0.81]], dtype=torch.float32),
        sde_means=torch.tensor([[[[0.81]]]], dtype=torch.float32),
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"context_gradient_mode": "unknown"}, "must be one of"),
        (
            {"context_gradient_mode": "stage_boundary", "ratio_norm": False},
            "requires ratio_norm=True",
        ),
        (
            {
                "context_gradient_mode": "stage_boundary",
                "ratio_norm": True,
                "reuse_ratio_context_for_mse": True,
            },
            "already shares one detached context",
        ),
        (
            {"lazy_first_update_anchor": True, "ratio_norm": True},
            "requires context_gradient_mode='stage_boundary'",
        ),
        (
            {
                "lazy_first_update_anchor": True,
                "ratio_norm": True,
                "context_gradient_mode": "stage_boundary",
                "old_logp_source": "rollout",
            },
            "old_logp_source='replay'",
        ),
        (
            {"stage_prepared_replay_to_cpu": True},
            "requires context_gradient_mode='stage_boundary'",
        ),
    ],
)
def test_stage_boundary_configuration_validation(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        BagelFlowUniGRPO(params=object(), stage=object(), **kwargs)


def test_prepared_replay_cpu_staging_defaults_off() -> None:
    algorithm = BagelFlowUniGRPO(params=object(), stage=object())

    assert algorithm.stage_prepared_replay_to_cpu is False


def test_ar_replay_uses_bundle_compute_device_when_fsdp_shards_are_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution_device = torch.device("meta")
    bundle = SimpleNamespace(
        device=execution_device,
        transformer=nn.Linear(2, 2),
        new_token_ids={"bos_token_id": 1},
    )
    stage = object.__new__(BagelARStage)
    stage.model = bundle
    stage.replay_mode = "train"
    stage.logprob_dtype = torch.float32
    captured: dict[str, torch.device] = {}

    def replay_train(_conditions, **kwargs):
        captured["device"] = kwargs["device"]
        return [torch.tensor([0.0])]

    monkeypatch.setattr(stage, "_replay_train", replay_train)
    conditions = BagelARConditions.for_sample(splits=[{"kind": "text", "ids": torch.tensor([2, 3], dtype=torch.long)}])
    segment = TextSegment.pack(tokens=[torch.tensor([4])], log_probs=[torch.tensor([0.0])])

    stage.replay(conditions, segment=segment)

    assert next(bundle.transformer.parameters()).device.type == "cpu"
    assert captured["device"] == execution_device


def test_unigrpo_mse_uses_bundle_compute_device_when_fsdp_shards_are_on_cpu() -> None:
    execution_device = torch.device("meta")

    class StopAfterDeviceCapture(Exception):
        pass

    class FakeStage:
        def __init__(self) -> None:
            self.model = SimpleNamespace(
                device=execution_device,
                transformer=nn.Linear(2, 2),
            )
            self.captured_device = None
            self.context_grad_enabled = None

        def build_forward_kwargs(self, _conditions, *, params, device):
            del params
            self.captured_device = device
            self.context_grad_enabled = torch.is_grad_enabled()
            raise StopAfterDeviceCapture

    stage = FakeStage()
    algorithm = BagelFlowUniGRPO(
        params=object(),
        stage=stage,
        mse_weight=1.0,
        ratio_norm=True,
    )
    algorithm._ratio_norm_surrogate = lambda **_kwargs: AlgorithmStepResult(
        loss=0.0,
        metrics={},
        num_steps_or_tokens=1,
        has_backward=True,
    )
    segment = make_image_segment(
        sigmas=torch.tensor([1.0, 0.0]),
        sde_indices=torch.tensor([0], dtype=torch.long),
    )

    with pytest.raises(StopAfterDeviceCapture):
        algorithm.compute_loss_and_backward(
            conditions={},
            segment=segment,
            advantages=torch.ones(1),
            training_progress=0.0,
            loss_scale=1.0,
        )

    assert next(stage.model.transformer.parameters()).device.type == "cpu"
    assert stage.captured_device == execution_device
    assert stage.context_grad_enabled is False


def test_stage_boundary_preserves_forward_and_loss_but_stops_context_gradient() -> None:
    def run(mode: str):
        transformer = _BoundaryTransformer()
        stage = _BoundaryStage(transformer)
        algorithm = BagelFlowUniGRPO(
            params=SimpleNamespace(eta=0.8),
            stage=stage,
            mse_weight=0.0,
            ratio_norm=True,
            context_gradient_mode=mode,
        )
        result = algorithm.compute_loss_and_backward(
            conditions={"context_scale": 1.5},
            segment=_boundary_segment(),
            advantages=torch.ones(1),
            training_progress=0.25,
            loss_scale=1.0,
        )
        return result, stage, transformer

    full_result, full_stage, full_transformer = run("full")
    boundary_result, boundary_stage, boundary_transformer = run("stage_boundary")

    assert torch.equal(full_stage.last_velocity, boundary_stage.last_velocity)
    assert boundary_result.loss == pytest.approx(full_result.loss, rel=0.0, abs=0.0)
    assert boundary_result.metrics == full_result.metrics
    assert torch.equal(boundary_transformer.policy_weight.grad, full_transformer.policy_weight.grad)
    assert full_transformer.context_weight.grad is not None
    assert float(full_transformer.context_weight.grad.abs()) > 0.0
    assert boundary_transformer.context_weight.grad is None
    assert boundary_stage.context_builds == 1
    assert boundary_stage.replay_calls == 1


def test_stage_boundary_fused_ratio_mse_matches_separate_backward() -> None:
    params = SimpleNamespace(eta=0.8)
    segment = _boundary_segment()
    advantages = torch.ones(1)
    loss_scale = 0.5
    mse_weight = 0.3
    reference_velocity = torch.tensor([[0.15]], dtype=torch.float32)

    fused_transformer = _BoundaryTransformer()
    fused_stage = _BoundaryStage(fused_transformer)
    fused = BagelFlowUniGRPO(
        params=params,
        stage=fused_stage,
        mse_weight=mse_weight,
        ratio_norm=True,
        context_gradient_mode="stage_boundary",
    )
    fused_context = {"context": fused_transformer.context_weight.detach() * 1.5}
    fused_result = fused._stage_boundary_loss_and_backward(
        segment=segment,
        advantages=advantages,
        training_progress=0.25,
        loss_scale=loss_scale,
        forward_kwargs=fused_context,
        target_steps=[0],
        reference_velocities=[reference_velocity],
    )

    separate_transformer = _BoundaryTransformer()
    separate_stage = _BoundaryStage(separate_transformer)
    separate = BagelFlowUniGRPO(
        params=params,
        stage=separate_stage,
        mse_weight=mse_weight,
        ratio_norm=True,
        context_gradient_mode="stage_boundary",
    )
    separate_context = {"context": separate_transformer.context_weight.detach() * 1.5}
    replay, _ = separate_stage.replay_from_forward_kwargs_with_velocities(
        separate_context,
        segment=segment,
        params=params,
        step_indices=[0],
    )
    policy_loss, _ = separate._ratio_norm_loss(
        replay=replay,
        segment=segment,
        advantages=advantages,
        training_progress=0.25,
        target_steps=[0],
    )
    (policy_loss * loss_scale).backward()
    second_velocity = separate_stage.predict_velocity_at(
        separate_context,
        sample=segment.latents_at(0)[0],
        sigma=segment.sigmas[0],
        params=params,
    )
    separate_mse = ((second_velocity - reference_velocity) ** 2).mean()
    (mse_weight * separate_mse * loss_scale).backward()
    separate_total = float(policy_loss.detach()) + mse_weight * float(separate_mse.detach())

    assert fused_result.loss == pytest.approx(separate_total, rel=1.0e-6, abs=1.0e-7)
    assert torch.allclose(fused_transformer.policy_weight.grad, separate_transformer.policy_weight.grad)
    assert fused_transformer.context_weight.grad is None
    assert separate_transformer.context_weight.grad is None
    assert fused_stage.replay_calls == separate_stage.replay_calls == 1
    assert fused_stage.predict_calls == 0
    assert separate_stage.predict_calls == 1


def test_stage_boundary_detaches_all_retained_forward_tree_leaves() -> None:
    class Cache:
        def __init__(self) -> None:
            self.key_cache = {0: None}
            self.value_cache = {0: None}

        def fork(self):
            cache = type(self)()
            cache.key_cache = self.key_cache.copy()
            cache.value_cache = self.value_cache.copy()
            return cache

    transformer = _BoundaryTransformer()
    stage = _BoundaryStage(transformer)
    algorithm = BagelFlowUniGRPO(
        params=SimpleNamespace(eta=0.8),
        stage=stage,
        mse_weight=0.3,
        ratio_norm=True,
        context_gradient_mode="stage_boundary",
    )
    attached_context = transformer.context_weight * 1.5
    attached_cache_value = transformer.context_weight.square().reshape(1)
    cache = Cache()
    cache.key_cache[0] = attached_cache_value
    cache.value_cache[0] = attached_cache_value
    forward_kwargs = {
        "context": attached_context,
        "past_key_values": cache,
        "cfg_img_past_key_values": cache,
    }
    segment = _boundary_segment()
    micro_batches = [({"forward_kwargs": forward_kwargs}, segment, torch.ones(1))]

    algorithm.prepare_update_batch(
        micro_batches=micro_batches,
        training_progress=0.25,
        loss_scale=1.0,
    )
    retained = algorithm._prepared_mse_batches[0].forward_kwargs

    assert retained is not forward_kwargs
    assert retained["context"].data_ptr() == attached_context.data_ptr()
    assert not retained["context"].requires_grad
    assert retained["past_key_values"] is retained["cfg_img_past_key_values"]
    assert retained["past_key_values"] is not cache
    assert retained["past_key_values"].key_cache[0].data_ptr() == attached_cache_value.data_ptr()
    assert not retained["past_key_values"].key_cache[0].requires_grad
    assert attached_context.requires_grad
    assert attached_cache_value.requires_grad

    result = algorithm.compute_loss_and_backward(
        conditions=micro_batches[0][0],
        segment=segment,
        advantages=micro_batches[0][2],
        training_progress=0.25,
        loss_scale=1.0,
    )

    assert result.has_backward
    assert transformer.context_weight.grad is None
    assert transformer.policy_weight.grad is not None


def test_prepared_replay_cpu_staging_hydrates_only_the_consumed_micro() -> None:
    class Cache:
        def __init__(self) -> None:
            self.key_cache = {0: None}
            self.value_cache = {0: None}

        def fork(self):
            cache = type(self)()
            cache.key_cache = self.key_cache.copy()
            cache.value_cache = self.value_cache.copy()
            return cache

    transformer = _BoundaryTransformer()
    stage = _BoundaryStage(transformer)
    algorithm = BagelFlowUniGRPO(
        params=SimpleNamespace(eta=0.8),
        stage=stage,
        mse_weight=0.3,
        ratio_norm=True,
        context_gradient_mode="stage_boundary",
        stage_prepared_replay_to_cpu=True,
    )
    source_kwargs = []
    micro_batches = []

    def ordered_segment():
        return make_image_segment(
            latents=torch.tensor([[[[0.3]], [[0.2]]]], dtype=torch.float32),
            sigmas=torch.tensor([0.8, 0.4], dtype=torch.float32),
            indices=torch.tensor([0, 1], dtype=torch.long),
            sde_indices=torch.tensor([1, 0], dtype=torch.long),
            sde_logp=torch.tensor([[0.0, 0.0]], dtype=torch.float32),
            sde_means=torch.tensor([[[[0.0]], [[0.0]]]], dtype=torch.float32),
        )

    for index in range(2):
        context = (transformer.context_weight * float(index + 1)).reshape(1)
        cache_value = (transformer.context_weight * float(index + 3)).reshape(1)
        cache = Cache()
        cache.key_cache[0] = cache_value
        cache.value_cache[0] = cache_value
        kwargs = {
            "context": context,
            "past_key_values": cache,
            "cfg_img_past_key_values": cache,
        }
        source_kwargs.append(kwargs)
        micro_batches.append(({"forward_kwargs": kwargs}, ordered_segment(), torch.ones(1)))

    algorithm.prepare_update_batch(
        micro_batches=micro_batches,
        training_progress=0.25,
        loss_scale=0.5,
    )
    queue = algorithm._prepared_mse_batches
    assert queue is not None
    assert all(entry is not None and entry.staged_on_cpu for entry in queue)
    current_staged = queue[0]
    future_staged = queue[1]
    assert current_staged is not None
    assert future_staged is not None
    current_staged_ptr = current_staged.forward_kwargs["context"].data_ptr()
    future_staged_ptr = future_staged.forward_kwargs["context"].data_ptr()

    current = algorithm._take_prepared_mse([1, 0])

    assert current is not None
    assert current.staged_on_cpu is False
    assert current.forward_kwargs["context"].device.type == "cpu"
    assert current.forward_kwargs["context"].dtype == source_kwargs[0]["context"].dtype
    assert current.forward_kwargs["context"].data_ptr() != current_staged_ptr
    assert not current.forward_kwargs["context"].requires_grad
    assert current.forward_kwargs["past_key_values"] is current.forward_kwargs["cfg_img_past_key_values"]
    assert [float(velocity) for velocity in current.reference_velocities] == pytest.approx(
        [0.54, 0.61], rel=0.0, abs=3.0e-4
    )
    assert algorithm._prepared_mse_batches == [future_staged]
    assert future_staged.forward_kwargs["context"].data_ptr() == future_staged_ptr
    assert future_staged.staged_on_cpu

    algorithm.finish_update_batch(succeeded=False)
    assert algorithm._prepared_mse_batches is None


def test_prepared_replay_cpu_staging_releases_queue_after_hydration_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transformer = _BoundaryTransformer()
    stage = _BoundaryStage(transformer)
    algorithm = BagelFlowUniGRPO(
        params=SimpleNamespace(eta=0.8),
        stage=stage,
        mse_weight=0.3,
        ratio_norm=True,
        context_gradient_mode="stage_boundary",
        stage_prepared_replay_to_cpu=True,
    )
    micro_batches = [
        ({"context_scale": 1.0}, _boundary_segment(), torch.ones(1)),
        ({"context_scale": 2.0}, _boundary_segment(), torch.ones(1)),
    ]
    algorithm.prepare_update_batch(
        micro_batches=micro_batches,
        training_progress=0.25,
        loss_scale=0.5,
    )

    def fail_hydration(*_args, **_kwargs):
        raise RuntimeError("synthetic hydration failure")

    monkeypatch.setattr("unirl.algorithms.bagel_flow_unigrpo.move_replay_tree", fail_hydration)
    with pytest.raises(RuntimeError, match="synthetic hydration failure"):
        algorithm._take_prepared_mse([0])

    assert algorithm._prepared_mse_batches is not None
    assert len(algorithm._prepared_mse_batches) == 1
    algorithm.finish_update_batch(succeeded=False)
    assert algorithm._prepared_mse_batches is None


def test_prepared_replay_cpu_staging_restores_reference_weights_after_d2h_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transformer = _BoundaryTransformer()
    stage = _BoundaryStage(transformer)
    algorithm = BagelFlowUniGRPO(
        params=SimpleNamespace(eta=0.8),
        stage=stage,
        mse_weight=0.3,
        ratio_norm=True,
        context_gradient_mode="stage_boundary",
        stage_prepared_replay_to_cpu=True,
    )
    with algorithm._reference_weights(transformer):
        pass
    with torch.no_grad():
        transformer.context_weight.fill_(1.2)
        transformer.policy_weight.fill_(1.4)
    live_weights = {name: parameter.detach().clone() for name, parameter in transformer.named_parameters()}

    def fail_staging(*_args, **_kwargs):
        raise RuntimeError("synthetic D2H failure")

    monkeypatch.setattr("unirl.algorithms.bagel_flow_unigrpo.move_replay_tree", fail_staging)
    with pytest.raises(RuntimeError, match="synthetic D2H failure"):
        algorithm.prepare_update_batch(
            micro_batches=[({"context_scale": 1.0}, _boundary_segment(), torch.ones(1))],
            training_progress=0.25,
            loss_scale=1.0,
        )

    assert all(torch.equal(parameter, live_weights[name]) for name, parameter in transformer.named_parameters())
    assert algorithm._prepared_mse_batches is None
    algorithm.finish_update_batch(succeeded=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_prepared_replay_cpu_staging_cuda_matches_resident_loss_and_gradient() -> None:
    device = torch.device("cuda:0")

    class CudaStage:
        detach_forward_kwargs = staticmethod(BagelDiffusionStage.detach_forward_kwargs)

        def __init__(self) -> None:
            transformer = _BoundaryTransformer().to(device)
            self.model = SimpleNamespace(device=device, transformer=transformer)

        def build_forward_kwargs(self, conditions, *, params, device):
            del params
            scale = float(conditions.get("context_scale", 1.0))
            return {"context": self.model.transformer.context_weight * scale + torch.zeros((), device=device)}

        def predict_velocity_at(self, forward_kwargs, *, sample, sigma, params):
            del sigma, params
            sample = sample.to(device)
            return self.model.transformer.policy_weight * sample + forward_kwargs["context"]

        def replay_from_forward_kwargs_with_velocities(self, forward_kwargs, *, segment, params, step_indices):
            del params, step_indices
            velocity = self.predict_velocity_at(
                forward_kwargs,
                sample=segment.latents_at(0)[0],
                sigma=segment.sigmas[0],
                params=None,
            )
            replay = ReplayResult(
                log_probs=velocity.mean().reshape(1, 1),
                prev_sample_means=velocity.reshape(1, 1, *velocity.shape),
            )
            return replay, [velocity]

    def run(*, staged: bool):
        stage = CudaStage()
        algorithm = BagelFlowUniGRPO(
            params=SimpleNamespace(eta=0.8),
            stage=stage,
            mse_weight=0.3,
            ratio_norm=True,
            clip_range=0.1,
            context_gradient_mode="stage_boundary",
            stage_prepared_replay_to_cpu=staged,
        )
        segment = _boundary_segment()
        conditions = {"context_scale": 1.5}
        algorithm.prepare_update_batch(
            micro_batches=[(conditions, segment, torch.ones(1))],
            training_progress=0.25,
            loss_scale=1.0,
        )
        result = algorithm.compute_loss_and_backward(
            conditions=conditions,
            segment=segment,
            advantages=torch.ones(1),
            training_progress=0.25,
            loss_scale=1.0,
        )
        algorithm.finish_update_batch(succeeded=True)
        gradients = {
            name: parameter.grad.detach().cpu().clone()
            for name, parameter in stage.model.transformer.named_parameters()
            if parameter.grad is not None
        }
        return result, gradients

    resident_result, resident_gradients = run(staged=False)
    staged_result, staged_gradients = run(staged=True)

    assert staged_result.loss == pytest.approx(resident_result.loss, rel=0.0, abs=0.0)
    assert staged_result.metrics == resident_result.metrics
    assert staged_gradients.keys() == resident_gradients.keys()
    assert all(torch.equal(staged_gradients[name], resident_gradients[name]) for name in resident_gradients)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_prepared_replay_cpu_staging_bounds_the_cuda_queue_to_one_micro() -> None:
    device = torch.device("cuda:0")
    tensor_elements = 1 << 20
    num_micros = 4

    class LargeStage:
        detach_forward_kwargs = staticmethod(BagelDiffusionStage.detach_forward_kwargs)

        def __init__(self) -> None:
            self.model = SimpleNamespace(device=device, transformer=_BoundaryTransformer().to(device))

        def build_forward_kwargs(self, conditions, *, params, device):
            del params
            return {
                "context": torch.full(
                    (tensor_elements,),
                    float(conditions["value"]),
                    dtype=torch.bfloat16,
                    device=device,
                )
            }

        def predict_velocity_at(self, forward_kwargs, *, sample, sigma, params):
            del sample, sigma, params
            weight = self.model.transformer.policy_weight.to(dtype=forward_kwargs["context"].dtype)
            return forward_kwargs["context"] + weight

    def make_algorithm(*, staged: bool):
        stage = LargeStage()
        algorithm = BagelFlowUniGRPO(
            params=SimpleNamespace(eta=0.8),
            stage=stage,
            mse_weight=0.3,
            ratio_norm=True,
            context_gradient_mode="stage_boundary",
            stage_prepared_replay_to_cpu=staged,
        )
        return stage, algorithm

    micro_batches = [({"value": index + 1}, _boundary_segment(), torch.ones(1)) for index in range(num_micros)]

    torch.cuda.empty_cache()
    resident_stage, resident = make_algorithm(staged=False)
    resident_baseline = torch.cuda.memory_allocated(device)
    resident.prepare_update_batch(micro_batches=micro_batches, training_progress=0.25, loss_scale=0.25)
    resident_queue_bytes = torch.cuda.memory_allocated(device) - resident_baseline
    resident.finish_update_batch(succeeded=False)
    del resident, resident_stage
    gc.collect()
    torch.cuda.empty_cache()

    staged_stage, staged = make_algorithm(staged=True)
    staged_baseline = torch.cuda.memory_allocated(device)
    staged.prepare_update_batch(micro_batches=micro_batches, training_progress=0.25, loss_scale=0.25)
    staged_queue_bytes = torch.cuda.memory_allocated(device) - staged_baseline
    assert staged._prepared_mse_batches is not None
    assert all(
        entry is not None
        and entry.staged_on_cpu
        and entry.forward_kwargs["context"].device.type == "cpu"
        and all(velocity.device.type == "cpu" for velocity in entry.reference_velocities)
        for entry in staged._prepared_mse_batches
    )

    idle_allocations = []
    for _ in range(num_micros):
        current = staged._take_prepared_mse([0])
        assert current is not None
        assert current.forward_kwargs["context"].device == device
        del current
        torch.cuda.synchronize(device)
        gc.collect()
        idle_allocations.append(torch.cuda.memory_allocated(device) - staged_baseline)
    staged.finish_update_batch(succeeded=True)

    bytes_per_tensor = tensor_elements * torch.tensor([], dtype=torch.bfloat16).element_size()
    assert resident_queue_bytes >= (num_micros * 2 - 1) * bytes_per_tensor
    assert staged_queue_bytes < bytes_per_tensor
    assert max(idle_allocations) < bytes_per_tensor
    assert resident_queue_bytes - staged_queue_bytes >= (num_micros * 2 - 2) * bytes_per_tensor
    del staged, staged_stage
    gc.collect()
    torch.cuda.empty_cache()


def test_stage_boundary_rejects_broadcastable_velocity_shapes() -> None:
    transformer = _BoundaryTransformer()
    stage = _BoundaryStage(transformer)
    algorithm = BagelFlowUniGRPO(
        params=SimpleNamespace(eta=0.8),
        stage=stage,
        mse_weight=0.3,
        ratio_norm=True,
        context_gradient_mode="stage_boundary",
    )

    with pytest.raises(
        RuntimeError,
        match=r"shape mismatch at SDE step 0: policy=\(1, 1\), reference=\(1,\)",
    ):
        algorithm._stage_boundary_loss_and_backward(
            segment=_boundary_segment(),
            advantages=torch.ones(1),
            training_progress=0.25,
            loss_scale=1.0,
            forward_kwargs={"context": transformer.context_weight.detach() * 1.5},
            target_steps=[0],
            reference_velocities=[torch.tensor([0.15], dtype=torch.float32)],
        )

    assert transformer.policy_weight.grad is None


def test_stage_boundary_reuses_prepared_context_and_refreshes_next_update() -> None:
    transformer = _BoundaryTransformer()
    stage = _BoundaryStage(transformer)
    algorithm = BagelFlowUniGRPO(
        params=SimpleNamespace(eta=0.8),
        stage=stage,
        mse_weight=0.3,
        ratio_norm=True,
        context_gradient_mode="stage_boundary",
    )

    built_context_values = []
    for update, context_value in enumerate((0.4, 0.9)):
        transformer.context_weight.grad = None
        transformer.policy_weight.grad = None
        with torch.no_grad():
            transformer.context_weight.fill_(context_value)
            transformer.policy_weight.fill_(0.7 + 0.2 * update)
        segment = _boundary_segment()
        micro_batches = [({"context_scale": 1.0}, segment, torch.ones(1))]

        algorithm.prepare_update_batch(
            micro_batches=micro_batches,
            training_progress=0.25,
            loss_scale=1.0,
        )
        prepared_context = algorithm._prepared_mse_batches[0].forward_kwargs["context"]
        built_context_values.append(float(prepared_context))
        algorithm.compute_loss_and_backward(
            conditions=micro_batches[0][0],
            segment=segment,
            advantages=micro_batches[0][2],
            training_progress=0.25,
            loss_scale=1.0,
        )
        algorithm.finish_update_batch(succeeded=True)

        assert stage.predict_contexts[-1] is prepared_context
        assert stage.replay_contexts[-1] is prepared_context
        assert transformer.context_weight.grad is None
        assert transformer.policy_weight.grad is not None

    assert built_context_values == pytest.approx([0.4, 0.9])
    assert stage.context_builds == 2
    assert stage.replay_calls == 2
    assert stage.predict_calls == 2  # one frozen-reference velocity per update


def test_lazy_first_update_anchor_matches_eager_anchor_and_removes_one_replay() -> None:
    def run(*, lazy: bool, stage_prepared_replay_to_cpu: bool = False):
        transformer = _BoundaryTransformer()
        stage = _BoundaryStage(transformer)
        algorithm = BagelFlowUniGRPO(
            params=SimpleNamespace(eta=0.8),
            stage=stage,
            old_logp_source="replay",
            mse_weight=0.3,
            ratio_norm=True,
            clip_range=0.1,
            context_gradient_mode="stage_boundary",
            lazy_first_update_anchor=lazy,
            stage_prepared_replay_to_cpu=stage_prepared_replay_to_cpu,
        )
        conditions = {"context_scale": 1.5}
        segments = [_boundary_segment(), _boundary_segment()]
        updates = [[(conditions, segments[0])], [(conditions, segments[1])]]
        if lazy:
            # Native Omni emits a rollout log-prob but no trainer-runtime mean.
            # The prepared anchor contract must not read either placeholder.
            for segment in segments:
                segment.sde_logp = torch.tensor([[-99.0]])
                segment.sde_means = None
            algorithm.prepare_anchor_batch(updates=updates)
        else:
            for update in updates:
                for update_conditions, segment in update:
                    algorithm.prepare_segment(conditions=update_conditions, segment=segment)

        results = []
        gradients = []
        for update_index, ((update_conditions, segment),) in enumerate(updates):
            algorithm.prepare_update_batch(
                micro_batches=[(update_conditions, segment, torch.ones(1))],
                training_progress=0.25,
                loss_scale=1.0,
                update_index=update_index,
            )
            result = algorithm.compute_loss_and_backward(
                conditions=update_conditions,
                segment=segment,
                advantages=torch.ones(1),
                training_progress=0.25,
                loss_scale=1.0,
            )
            algorithm.finish_update_batch(succeeded=True)
            results.append(result)
            gradients.append(transformer.policy_weight.grad.detach().clone())
            with torch.no_grad():
                transformer.policy_weight.add_(transformer.policy_weight.grad, alpha=-0.01)
            transformer.policy_weight.grad = None

        if lazy:
            algorithm.finish_anchor_batch(succeeded=True)
        return stage, results, gradients

    eager_stage, eager_results, eager_gradients = run(lazy=False)
    lazy_stage, lazy_results, lazy_gradients = run(lazy=True)
    staged_lazy_stage, staged_lazy_results, staged_lazy_gradients = run(
        lazy=True,
        stage_prepared_replay_to_cpu=True,
    )

    assert eager_stage.context_builds == 4  # two eager anchors + two current contexts
    assert lazy_stage.context_builds == 3  # update-1 anchor + two current contexts
    assert [result.loss for result in lazy_results] == pytest.approx(
        [result.loss for result in eager_results], rel=0.0, abs=0.0
    )
    assert [result.metrics for result in lazy_results] == [result.metrics for result in eager_results]
    assert all(torch.equal(lazy_grad, eager_grad) for lazy_grad, eager_grad in zip(lazy_gradients, eager_gradients))
    assert lazy_results[0].metrics["ratio_mean"] == pytest.approx(1.0)
    assert lazy_results[0].metrics["rn_delta_mu_sq_mean"] == pytest.approx(0.0)
    assert staged_lazy_stage.context_builds == lazy_stage.context_builds
    assert [result.loss for result in staged_lazy_results] == pytest.approx(
        [result.loss for result in lazy_results], rel=0.0, abs=0.0
    )
    assert [result.metrics for result in staged_lazy_results] == [result.metrics for result in lazy_results]
    assert all(
        torch.equal(staged_grad, lazy_grad) for staged_grad, lazy_grad in zip(staged_lazy_gradients, lazy_gradients)
    )


def test_stage_boundary_direct_call_builds_one_shared_context() -> None:
    transformer = _BoundaryTransformer()
    stage = _BoundaryStage(transformer)
    algorithm = BagelFlowUniGRPO(
        params=SimpleNamespace(eta=0.8),
        stage=stage,
        mse_weight=0.3,
        ratio_norm=True,
        context_gradient_mode="stage_boundary",
    )

    result = algorithm.compute_loss_and_backward(
        conditions={"context_scale": 1.25},
        segment=_boundary_segment(),
        advantages=torch.ones(1),
        training_progress=0.25,
        loss_scale=1.0,
    )

    assert result.has_backward
    assert stage.context_builds == 1
    assert stage.replay_calls == 1
    assert stage.predict_calls == 1  # reference only; policy velocity came from replay
    assert stage.predict_contexts[0] is stage.replay_contexts[0]


def test_unigrpo_full_ft_reference_survives_checkpoint_resume(tmp_path: Path) -> None:
    def make_algorithm(transformer: nn.Module) -> BagelFlowUniGRPO:
        stage = SimpleNamespace(model=SimpleNamespace(transformer=transformer))
        algorithm = BagelFlowUniGRPO(params=object(), stage=stage, mse_weight=1.0)
        algorithm.rank_info = SimpleNamespace(rank=0, world_size=1)
        return algorithm

    original = nn.Linear(2, 2)
    with torch.no_grad():
        original.weight.fill_(1.0)
        original.bias.fill_(2.0)
    algorithm = make_algorithm(original)
    with algorithm._reference_weights(original):
        pass

    with torch.no_grad():
        original.weight.fill_(3.0)
        original.bias.fill_(4.0)
    tuned_state = {name: tensor.detach().clone() for name, tensor in original.state_dict().items()}
    algorithm.save_reference_checkpoint(str(tmp_path))

    resumed_model = nn.Linear(2, 2)
    resumed_model.load_state_dict(tuned_state)
    resumed = make_algorithm(resumed_model)
    resumed.load_reference_checkpoint(str(tmp_path))

    with resumed._reference_weights(resumed_model):
        assert torch.equal(resumed_model.weight, torch.ones_like(resumed_model.weight))
        assert torch.equal(resumed_model.bias, torch.full_like(resumed_model.bias, 2.0))
    assert torch.equal(resumed_model.weight, torch.full_like(resumed_model.weight, 3.0))
    assert torch.equal(resumed_model.bias, torch.full_like(resumed_model.bias, 4.0))


def test_unigrpo_rejects_bad_reference_without_mutating_live_weights() -> None:
    transformer = nn.Linear(2, 2)
    stage = SimpleNamespace(model=SimpleNamespace(transformer=transformer))
    algorithm = BagelFlowUniGRPO(params=object(), stage=stage, mse_weight=1.0)
    with algorithm._reference_weights(transformer):
        pass

    with torch.no_grad():
        transformer.weight.fill_(3.0)
        transformer.bias.fill_(4.0)
    algorithm._ref_snapshot["bias"] = torch.zeros(3, dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="incompatible with the live parameter"):
        with algorithm._reference_weights(transformer):
            pass

    assert torch.equal(transformer.weight, torch.full_like(transformer.weight, 3.0))
    assert torch.equal(transformer.bias, torch.full_like(transformer.bias, 4.0))


def test_unigrpo_full_ft_reference_swap_is_hoisted_once_per_optimizer_update() -> None:
    class FakeStage:
        def __init__(self) -> None:
            transformer = nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                transformer.weight.fill_(1.0)
            self.model = SimpleNamespace(device=torch.device("cpu"), transformer=transformer)
            self.context_weights: list[float] = []
            self.velocity_calls = 0

        def build_forward_kwargs(self, conditions, *, params, device):
            del params, device
            weight = self.model.transformer.weight.detach().clone()
            self.context_weights.append(float(weight.item()))
            return {"context_weight": weight, "sample_id": conditions["sample_id"]}

        def predict_velocity_at(self, forward_kwargs, *, sample, sigma, params):
            del sigma, params
            self.velocity_calls += 1
            # The detached context term makes it observable that update 2 was
            # prepared after update 1's weight change.
            return self.model.transformer(sample) + 0.1 * forward_kwargs["context_weight"]

    def segment(value: float):
        return make_image_segment(
            latents=torch.tensor([[[[value]], [[value + 0.5]]]], dtype=torch.float32),
            sigmas=torch.tensor([1.0, 0.0]),
            indices=torch.tensor([0, 1], dtype=torch.long),
            sde_indices=torch.tensor([0], dtype=torch.long),
        )

    stage = FakeStage()
    algorithm = BagelFlowUniGRPO(
        params=object(),
        stage=stage,
        mse_weight=1.0,
        ratio_norm=True,
    )
    algorithm._ratio_norm_surrogate = lambda **_kwargs: AlgorithmStepResult(
        loss=0.0,
        metrics={},
        num_steps_or_tokens=1,
        has_backward=True,
    )

    reference_swaps = 0
    original_reference_weights = algorithm._reference_weights

    @contextmanager
    def counted_reference_weights(transformer):
        nonlocal reference_swaps
        reference_swaps += 1
        with original_reference_weights(transformer):
            yield

    algorithm._reference_weights = counted_reference_weights

    updates = [
        [
            ({"sample_id": 0}, segment(1.0), torch.ones(1)),
            ({"sample_id": 1}, segment(2.0), torch.ones(1)),
        ],
        [
            ({"sample_id": 2}, segment(3.0), torch.ones(1)),
            ({"sample_id": 3}, segment(4.0), torch.ones(1)),
        ],
    ]
    for update_index, micro_batches in enumerate(updates):
        if update_index == 1:
            # Simulate the first optimizer update before preparing the second.
            stage.model.transformer.weight.grad = None
            with torch.no_grad():
                stage.model.transformer.weight.fill_(2.0)
        algorithm.prepare_update_batch(micro_batches=micro_batches, training_progress=0.0, loss_scale=0.5)
        assert stage.context_weights[-2:] == [float(update_index + 1)] * 2
        for conditions, micro_segment, advantages in micro_batches:
            algorithm.compute_loss_and_backward(
                conditions=conditions,
                segment=micro_segment,
                advantages=advantages,
                training_progress=0.0,
                loss_scale=0.5,
            )

    assert reference_swaps == 2
    assert stage.velocity_calls == 8  # four detached v_ref + four grad-enabled v_theta
    assert algorithm._prepared_mse_batches is None
    assert stage.model.transformer.weight.grad is not None


def test_unigrpo_reuses_ratio_context_without_changing_loss_or_gradient() -> None:
    class FakeStage:
        def __init__(self) -> None:
            transformer = nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                transformer.weight.fill_(1.0)
            self.model = SimpleNamespace(device=torch.device("cpu"), transformer=transformer)
            self.context_builds = 0
            self.velocity_calls = 0

        def build_forward_kwargs(self, conditions, *, params, device):
            del params, device
            self.context_builds += 1
            return {
                "context_weight": self.model.transformer.weight.detach().clone(),
                "sample_id": conditions["sample_id"],
            }

        def predict_velocity_at(self, forward_kwargs, *, sample, sigma, params):
            del sigma, params
            self.velocity_calls += 1
            return self.model.transformer(sample) + 0.1 * forward_kwargs["context_weight"]

    def segment(value: float):
        return make_image_segment(
            latents=torch.tensor([[[[value]], [[value + 0.5]]]], dtype=torch.float32),
            sigmas=torch.tensor([1.0, 0.0]),
            indices=torch.tensor([0, 1], dtype=torch.long),
            sde_indices=torch.tensor([0], dtype=torch.long),
        )

    def run(*, reuse: bool):
        stage = FakeStage()
        algorithm = BagelFlowUniGRPO(
            params=object(),
            stage=stage,
            mse_weight=1.0,
            ratio_norm=True,
            reuse_ratio_context_for_mse=reuse,
        )
        # Capture the immutable base at weight=1, then emulate a tuned policy.
        with algorithm._reference_weights(stage.model.transformer):
            pass
        with torch.no_grad():
            stage.model.transformer.weight.fill_(2.0)

        surrogate_calls = 0

        def surrogate(*, conditions, loss_scale, **_kwargs):
            nonlocal surrogate_calls
            surrogate_calls += 1
            context = stage.model.transformer.weight * float(conditions["sample_id"] + 1)
            loss = context.square().mean()
            (loss * loss_scale).backward()
            return AlgorithmStepResult(
                loss=float(loss.detach()),
                metrics={"policy_loss": float(loss.detach())},
                num_steps_or_tokens=1,
                has_backward=True,
            )

        if reuse:

            def surrogate_with_context(**kwargs):
                result = surrogate(**kwargs)
                forward_kwargs = {
                    "context_weight": stage.model.transformer.weight.detach().clone(),
                    "sample_id": kwargs["conditions"]["sample_id"],
                }
                return result, forward_kwargs

            algorithm._ratio_norm_surrogate_with_context = surrogate_with_context
        else:
            algorithm._ratio_norm_surrogate = surrogate

        micro_batches = [
            ({"sample_id": 0}, segment(1.0), torch.ones(1)),
            ({"sample_id": 1}, segment(2.0), torch.ones(1)),
        ]
        algorithm.prepare_update_batch(
            micro_batches=micro_batches,
            training_progress=0.25,
            loss_scale=0.5,
        )
        results = [
            algorithm.compute_loss_and_backward(
                conditions=conditions,
                segment=micro_segment,
                advantages=advantages,
                training_progress=0.25,
                loss_scale=0.5,
            )
            for conditions, micro_segment, advantages in micro_batches
        ]
        return stage, algorithm, results, surrogate_calls, stage.model.transformer.weight.grad.detach().clone()

    legacy_stage, legacy_algorithm, legacy_results, legacy_surrogates, legacy_grad = run(reuse=False)
    reused_stage, reused_algorithm, reused_results, reused_surrogates, reused_grad = run(reuse=True)

    assert legacy_surrogates == reused_surrogates == 2
    assert legacy_stage.context_builds == 2
    assert reused_stage.context_builds == 0
    assert legacy_stage.velocity_calls == reused_stage.velocity_calls == 4
    assert [result.loss for result in reused_results] == pytest.approx([result.loss for result in legacy_results])
    assert torch.equal(reused_grad, legacy_grad)
    assert legacy_algorithm._prepared_mse_batches is None
    assert reused_algorithm._prepared_mse_batches is None


def test_prompt_prefill_moves_vendor_cpu_tensors_to_compute_device() -> None:
    execution_device = torch.device("meta")
    seen: dict[str, torch.device] = {}

    class FakeBagel:
        def prepare_prompts(self, **_kwargs):
            return (
                {
                    "packed_text_ids": torch.tensor([1, 2]),
                    "packed_text_indexes": torch.tensor([0, 1]),
                    "key_values_lens": torch.tensor([0]),
                },
                [2],
                [2],
            )

        def forward_cache_update_text(self, _past, **generation_input):
            seen.update({name: tensor.device for name, tensor in generation_input.items()})
            return "advanced-cache"

    context = prefill_prompt_text(
        FakeBagel(),
        {"kv_lens": [0], "ropes": [0], "past_key_values": "empty-cache"},
        prompt="hello",
        tokenizer=object(),
        new_token_ids={"bos_token_id": 1, "eos_token_id": 2},
        device=execution_device,
    )

    assert set(seen.values()) == {execution_device}
    assert context == {"kv_lens": [2], "ropes": [2], "past_key_values": "advanced-cache"}
