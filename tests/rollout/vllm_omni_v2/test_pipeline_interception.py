"""The shared RL-pipeline interception mechanics — CPU, no vllm-omni.

The pipeline subclasses themselves only run inside the GPU worker (they
import vllm-omni at module level), but the mechanics they delegate to are
deliberately runtime-free (``pipelines/_shared/interception.py``) — wire
objects are ``SimpleNamespace`` fakes, same idiom as ``conftest.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from unirl.rollout.engine.vllm_omni_v2.pipelines._shared.flow_match_sde_scheduler import (
    FlowMatchSDEDiscreteScheduler,
)
from unirl.rollout.engine.vllm_omni_v2.pipelines._shared.interception import (
    detach_cpu,
    detach_cpu_pair,
    drain_trajectory_into,
    inject_latents,
    resolve_request_noise,
    stamp_custom_output,
)


def make_noise_req(request_id: str = "1_uuid", **extra_args: Any) -> SimpleNamespace:
    return SimpleNamespace(request_id=request_id, sampling_params=SimpleNamespace(extra_args=extra_args))


# --------------------------------------------------------------------------- #
# resolve_request_noise — both x_T transports + the failure modes
# --------------------------------------------------------------------------- #


def test_resolve_noise_none_without_transport():
    assert resolve_request_noise(make_noise_req(), caller="t") is None


def test_resolve_noise_picks_batch_row_by_request_id():
    batch = torch.arange(24, dtype=torch.float32).reshape(3, 2, 2, 2)
    out = resolve_request_noise(make_noise_req("1_uuid", initial_noise_batch=batch), caller="t")
    # [1, ...] slice (batch dim kept), cloned (mutating it must not touch the batch).
    assert out.shape == (1, 2, 2, 2)
    assert torch.equal(out[0], batch[1])
    out += 1.0
    assert torch.equal(batch[1], torch.arange(8, 16, dtype=torch.float32).reshape(2, 2, 2))


def test_resolve_noise_recipe_row_is_deterministic():
    req = make_noise_req(
        "0_uuid",
        init_noise_group_ids=["g7", "g8"],
        init_noise_seed=5,
        init_noise_latent_shape=[4, 2, 2],
    )
    a = resolve_request_noise(req, caller="t")
    b = resolve_request_noise(req, caller="t")
    assert a.shape == (1, 4, 2, 2)
    assert torch.equal(a, b)  # byte-identical regeneration is the recipe contract
    # A different gid row draws different noise.
    other = resolve_request_noise(
        make_noise_req(
            "1_uuid", init_noise_group_ids=["g7", "g8"], init_noise_seed=5, init_noise_latent_shape=[4, 2, 2]
        ),
        caller="t",
    )
    assert not torch.equal(a, other)


def test_resolve_noise_unparsable_request_id_raises():
    with pytest.raises(RuntimeError, match="cannot parse batch index"):
        resolve_request_noise(make_noise_req("nope", initial_noise_batch=torch.zeros(1, 2)), caller="t")


def test_resolve_noise_out_of_bounds_raises():
    with pytest.raises(IndexError, match="out of bounds"):
        resolve_request_noise(make_noise_req("5_uuid", initial_noise_batch=torch.zeros(2, 2)), caller="t")
    with pytest.raises(IndexError, match="init_noise_group_ids"):
        resolve_request_noise(
            make_noise_req("5_uuid", init_noise_group_ids=["g0"], init_noise_seed=0, init_noise_latent_shape=[2]),
            caller="t",
        )


# --------------------------------------------------------------------------- #
# inject_latents — the prepare_latents positional surgery
# --------------------------------------------------------------------------- #


def test_inject_latents_replaces_positional_slot():
    noise = torch.zeros(1, 4, 2, 2)
    # sd3/hv15 layout: (b, x, h, w, dtype@4, device@5, generator@6, latents@7)
    args = (1, 4, 2, 2, torch.float64, torch.device("cpu"), None, None)
    new_args, new_kwargs = inject_latents(args, {}, noise)
    assert new_args[7] is not None and new_args[:7] == args[:7]
    assert new_args[7].dtype == torch.float64  # moved to the call site's dtype
    assert new_kwargs == {}  # no double-bind via kwargs


def test_inject_latents_kwargs_fallback_for_partial_calls():
    noise = torch.zeros(1, 4)
    args = (1, 4)  # fewer than 8 positionals
    new_args, new_kwargs = inject_latents(args, {"dtype": torch.float16}, noise)
    assert new_args == args
    assert torch.equal(new_kwargs["latents"], noise.to(torch.float16))
    assert new_kwargs["latents"].dtype == torch.float16


# --------------------------------------------------------------------------- #
# harvest mechanics
# --------------------------------------------------------------------------- #


def _fake_out() -> SimpleNamespace:
    return SimpleNamespace(
        custom_output=None, trajectory_latents=None, trajectory_timesteps=None, trajectory_log_probs=None
    )


def test_stamp_custom_output_creates_and_preserves():
    out = _fake_out()
    stamp_custom_output(out, "a", 1)
    stamp_custom_output(out, "b", 2)
    assert out.custom_output == {"a": 1, "b": 2}


def test_drain_trajectory_into_stamps_wire_fields():
    latents, sigmas, log_probs = torch.zeros(1, 3, 4), torch.linspace(1, 0, 3), torch.zeros(1, 2)
    scheduler = SimpleNamespace(
        drain_trajectory=lambda: (latents, sigmas, torch.zeros(3), log_probs),
        last_sde_step_indices=[0, 1],
    )
    out = _fake_out()
    drain_trajectory_into(out, scheduler)
    assert torch.equal(out.trajectory_latents, latents)
    assert torch.equal(out.trajectory_timesteps, sigmas)  # the true [0,1] σ schedule
    assert torch.equal(out.trajectory_log_probs, log_probs)
    assert out.custom_output["sde_step_indices"] == [0, 1]


def test_drain_trajectory_into_noop_when_empty():
    out = _fake_out()
    drain_trajectory_into(out, SimpleNamespace(drain_trajectory=lambda: None, last_sde_step_indices=[]))
    assert out.trajectory_latents is None and out.custom_output is None


# --------------------------------------------------------------------------- #
# detach helpers + scheduler.arm
# --------------------------------------------------------------------------- #


def test_detach_cpu_passthrough_and_pair():
    t = torch.ones(2, requires_grad=True) * 2
    moved = detach_cpu(t)
    assert moved.requires_grad is False and moved.device.type == "cpu"
    assert detach_cpu(None) is None and detach_cpu("x") == "x"
    cos, sin = detach_cpu_pair((t, t))
    assert cos.requires_grad is False and sin.requires_grad is False
    assert detach_cpu_pair(None) is None


def test_scheduler_arm_round_trip():
    sched = FlowMatchSDEDiscreteScheduler(eta=1.0)
    sched.arm(eta=0.7, sde_indices=[3, 1])
    assert sched._eta == 0.7
    assert sched._sde_indices_set == frozenset({1, 3})
    sched.arm(eta=0.0, sde_indices=None)  # disarm: pure ODE, capture still on
    assert sched._eta == 0.0 and sched._sde_indices_set is None
    with pytest.raises(ValueError, match="must be >= 0"):
        sched.arm(eta=-1.0, sde_indices=None)
