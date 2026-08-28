"""Focused CPU tests for SenseNova-U1.5 geometry and flow conventions."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from unirl.models.sensenova_u1.conditions import SenseNovaU1Conditions
from unirl.models.sensenova_u1.config import SenseNovaU1PipelineConfig
from unirl.models.sensenova_u1.diffusion import (
    SenseNovaU1DiffusionParams,
    SenseNovaU1DiffusionStep,
    resolve_noise_scale,
)
from unirl.models.sensenova_u1.pipeline import SenseNovaU1Pipeline
from unirl.models.sensenova_u1.pixels import packed_pixel_shape, patchify_pixels, unpatchify_pixels
from unirl.sde.kernels import FlowSDEStrategy
from unirl.sde.runtime import FlowMatchSchedulePolicy


def test_pixel_pack_roundtrip() -> None:
    pixels = torch.arange(3 * 64 * 96, dtype=torch.float32).reshape(1, 3, 64, 96)
    packed = patchify_pixels(pixels, patch_size=32)

    assert packed.shape == (1, *packed_pixel_shape((64, 96), patch_size=32))
    torch.testing.assert_close(
        unpatchify_pixels(packed, image_shape=(64, 96), patch_size=32),
        pixels,
    )


def test_driver_noise_uses_upstream_nchw_layout() -> None:
    sampling = SimpleNamespace(height=512, width=768)

    assert SenseNovaU1Pipeline.latent_shape(model_config=None, sampling_spec=sampling) == (3, 512, 768)


def test_data_time_velocity_is_negated_for_sigma_solver() -> None:
    class ConstantVelocityStep(SenseNovaU1DiffusionStep):
        def predict_velocity(self, *args, sample, **kwargs):
            return torch.full_like(sample, 2.0)

    state = torch.zeros(1, 2, 3, dtype=torch.bfloat16)
    next_state, log_prob, _ = ConstantVelocityStep().step_with_logp(
        None,
        None,
        strategy=FlowSDEStrategy(),
        sample=state,
        sigma=torch.tensor(1.0),
        sigma_next=torch.tensor(0.5),
        params=None,
        eta=0.0,
    )

    # Upstream integrates dx/dt=2 over dt=0.5. The framework integrates over
    # decreasing sigma, so its noise prediction must be dx/dsigma=-2.
    torch.testing.assert_close(next_state, torch.ones_like(state))
    assert next_state.dtype == torch.bfloat16
    assert log_prob is None


def test_resolution_dependent_noise_scale() -> None:
    model = SimpleNamespace(
        patch_size=16,
        downsample_ratio=0.5,
        noise_scale=1.0,
        noise_scale_mode="resolution",
        noise_scale_base_image_seq_len=64,
        noise_scale_max_value=16.0,
    )

    assert resolve_noise_scale(model, (256, 256)) == 1.0
    assert resolve_noise_scale(model, (512, 512)) == 2.0
    assert resolve_noise_scale(model, (1024, 1024)) == 4.0


def test_shifted_schedule_matches_upstream_data_time() -> None:
    steps = 20
    shift = 3.0
    sigmas = FlowMatchSchedulePolicy.static_only(shift).compute_sigma(
        num_inference_steps=steps,
        height=512,
        width=512,
    )
    upstream_time = torch.linspace(0.0, 1.0, steps + 1)
    upstream_sigma = 1.0 - upstream_time
    upstream_sigma = shift * upstream_sigma / (1.0 + (shift - 1.0) * upstream_sigma)

    torch.testing.assert_close(sigmas, upstream_sigma)


def test_stochastic_transition_uses_unit_noise_coordinates() -> None:
    class ZeroVelocityStep(SenseNovaU1DiffusionStep):
        def predict_velocity(self, *args, sample, **kwargs):
            return torch.zeros_like(sample)

    model = SimpleNamespace(
        patch_size=16,
        downsample_ratio=0.5,
        noise_scale=1.0,
        noise_scale_mode="resolution",
        noise_scale_base_image_seq_len=64,
        noise_scale_max_value=16.0,
    )
    bundle = SimpleNamespace(model=model)
    conditions = SimpleNamespace(image_shapes=[(512, 512)])
    strategy = FlowSDEStrategy()
    sample = torch.full((1, 2, 3), 2.0)
    previous = torch.full_like(sample, 1.5)
    sigma = torch.tensor(0.8)
    sigma_next = torch.tensor(0.6)

    actual, actual_logp, actual_mean = ZeroVelocityStep().step_with_logp(
        bundle,
        conditions,
        strategy=strategy,
        sample=sample,
        sigma=sigma,
        sigma_next=sigma_next,
        params=None,
        prev_sample=previous,
        sigma_max=0.7,
        eta=0.5,
    )
    expected, expected_logp, expected_mean = strategy.denoise(
        noise_pred=torch.zeros_like(sample),
        sample=sample / 2.0,
        sigma=sigma,
        sigma_next=sigma_next,
        prev_sample=previous / 2.0,
        sigma_max=0.7,
        eta=0.5,
    )

    torch.testing.assert_close(actual, expected * 2.0)
    torch.testing.assert_close(actual_mean, expected_mean * 2.0)
    torch.testing.assert_close(actual_logp, expected_logp)


def test_condition_cache_moves_with_indexes() -> None:
    cache = SimpleNamespace(
        layers=[
            SimpleNamespace(
                keys=torch.ones(1),
                values=torch.zeros(1),
            )
        ]
    )
    indexes = torch.zeros(3, 1, dtype=torch.long)
    conditions = SenseNovaU1Conditions(
        prompts=["prompt", "prompt"],
        condition_caches=[cache, cache],
        uncondition_caches=[None, None],
        condition_image_indexes=[indexes, indexes],
        uncondition_image_indexes=[None, None],
        image_shapes=[(32, 32), (32, 32)],
    )

    moved = conditions.to_device("meta")

    assert moved.condition_caches[0].layers[0].keys.device.type == "meta"
    assert moved.condition_caches[0].layers[0].values.device.type == "meta"
    assert moved.condition_caches[0] is moved.condition_caches[1]
    assert moved.condition_image_indexes[0].device.type == "meta"
    assert cache.layers[0].keys.device.type == "cpu"


def test_sensenova_defaults_match_official_precision() -> None:
    config = SenseNovaU1PipelineConfig(pretrained_model_ckpt_path="unused")
    params = SenseNovaU1DiffusionParams()

    assert config.trajectory_precision == "bf16"
    assert params.trajectory_precision == "bf16"
    assert params.logprob_precision == "fp32"


def test_cfg_combinations_match_upstream_formulas() -> None:
    step = SenseNovaU1DiffusionStep()
    condition = torch.tensor([[[2.0, 0.0], [0.0, 1.0]]])
    uncondition = torch.tensor([[[0.5, 0.5], [0.5, 0.5]]])
    classical = uncondition + 4.0 * (condition - uncondition)

    none = step._apply_cfg(
        condition,
        uncondition,
        guidance=4.0,
        cfg_norm="none",
        step_index=1,
    )
    global_norm = step._apply_cfg(
        condition,
        uncondition,
        guidance=4.0,
        cfg_norm="global",
        step_index=1,
    )
    channel_norm = step._apply_cfg(
        condition,
        uncondition,
        guidance=4.0,
        cfg_norm="channel",
        step_index=1,
    )
    zero_first = step._apply_cfg(
        condition,
        uncondition,
        guidance=4.0,
        cfg_norm="cfg_zero_star",
        step_index=0,
    )
    expected_global = classical * (
        torch.norm(condition, dim=(1, 2), keepdim=True) / (torch.norm(classical, dim=(1, 2), keepdim=True) + 1e-8)
    ).clamp(0.0, 1.0)
    expected_channel = classical * (
        torch.norm(condition, dim=-1, keepdim=True) / (torch.norm(classical, dim=-1, keepdim=True) + 1e-8)
    ).clamp(0.0, 1.0)

    torch.testing.assert_close(none, classical)
    torch.testing.assert_close(global_norm, expected_global)
    torch.testing.assert_close(channel_norm, expected_channel)
    torch.testing.assert_close(zero_first, torch.zeros_like(condition))
