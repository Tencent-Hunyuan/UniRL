"""FlowMatchSchedulePolicy σ-schedule tests — the ``shift_terminal`` seam.

Qwen-Image's ``scheduler_config.json`` ships ``shift_terminal: 0.02`` (the
terminal stretch diffusers applies after the dynamic mu shift); SD3/Flux ship
``null``. The policy is the driver-side σ SSOT — rollout (external sigmas,
verbatim via the ``_patches`` set_timesteps gate) and replay both consume what
it produces, so it must match the official diffusers schedule exactly, and
``shift_terminal=None`` must stay byte-identical to the pre-field behavior.

CPU-only: torch + numpy + diffusers (core deps), no sglang/GPU.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from unirl.models.qwen_image.config import _qwen_image_dynamic_overrides
from unirl.sde.runtime import FlowMatchSchedulePolicy, get_sigma_schedule

#: Qwen/Qwen-Image scheduler_config.json (verified from HF Hub).
_QWEN_SCHED_JSON = {
    "_class_name": "FlowMatchEulerDiscreteScheduler",
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "max_image_seq_len": 8192,
    "max_shift": 0.9,
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": 0.02,
    "time_shift_type": "exponential",
    "use_dynamic_shifting": True,
}


def _qwen_policy() -> FlowMatchSchedulePolicy:
    return FlowMatchSchedulePolicy(
        shift=1.0,
        use_dynamic_shifting=True,
        base_shift=0.5,
        max_shift=0.9,
        base_image_seq_len=256,
        max_image_seq_len=8192,
        time_shift_type="exponential",
        shift_terminal=0.02,
        vae_scale_factor=8,
        patch_size=2,
    )


def _diffusers_reference(num_steps: int, mu: float, *, shift_terminal) -> torch.Tensor:
    """The official schedule: base grid → mu shift → (optional) terminal stretch."""
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        use_dynamic_shifting=True,
        time_shift_type="exponential",
        shift_terminal=shift_terminal,
    )
    base = np.linspace(1.0, 1.0 / num_steps, num_steps)
    scheduler.set_timesteps(num_inference_steps=num_steps, sigmas=base, mu=mu)
    return scheduler.sigmas


def test_qwen_policy_matches_diffusers_reference_with_terminal_stretch():
    policy = _qwen_policy()
    num_steps, height, width = 12, 384, 384
    got = policy.compute_sigma(num_inference_steps=num_steps, height=height, width=width)

    image_seq_len = (384 // 8 // 2) * (384 // 8 // 2)
    mu = policy.compute_mu(image_seq_len, num_steps)
    want = _diffusers_reference(num_steps, mu, shift_terminal=0.02)

    torch.testing.assert_close(got, want, atol=0.0, rtol=0.0)
    # The stretch's defining property: the last interior σ lands on 0.02.
    assert float(got[-2]) == pytest.approx(0.02, abs=1e-6)
    assert float(got[0]) == pytest.approx(1.0, abs=1e-6)
    assert float(got[-1]) == 0.0  # terminal zero


def test_none_shift_terminal_is_byte_identical_to_prior_behavior():
    # SD3/Flux/Klein declare no shift_terminal — the new field must be inert.
    policy = _qwen_policy()
    policy.shift_terminal = None
    num_steps = 12
    got = policy.compute_sigma(num_inference_steps=num_steps, height=384, width=384)

    image_seq_len = (384 // 8 // 2) * (384 // 8 // 2)
    mu = policy.compute_mu(image_seq_len, num_steps)
    want = _diffusers_reference(num_steps, mu, shift_terminal=None)

    torch.testing.assert_close(got, want, atol=0.0, rtol=0.0)
    assert float(got[-2]) != pytest.approx(0.02, abs=1e-6)  # no stretch applied


def test_static_branch_rejects_shift_terminal():
    with pytest.raises(ValueError, match="shift_terminal"):
        get_sigma_schedule(8, 3.0, shift_terminal=0.02)  # no mu → static branch


def test_from_pretrained_reads_shift_terminal_from_scheduler_json(tmp_path):
    (tmp_path / "scheduler").mkdir()
    (tmp_path / "scheduler" / "scheduler_config.json").write_text(json.dumps(_QWEN_SCHED_JSON))

    policy = FlowMatchSchedulePolicy.from_pretrained(tmp_path, shift=1.0)
    assert policy.use_dynamic_shifting is True
    assert policy.shift_terminal == pytest.approx(0.02)
    assert policy.max_shift == pytest.approx(0.9)
    assert policy.max_image_seq_len == 8192


def test_from_pretrained_null_shift_terminal_normalizes_to_none(tmp_path):
    sched = dict(_QWEN_SCHED_JSON, shift_terminal=None)  # SD3/Flux-style null
    (tmp_path / "scheduler").mkdir()
    (tmp_path / "scheduler" / "scheduler_config.json").write_text(json.dumps(sched))

    policy = FlowMatchSchedulePolicy.from_pretrained(tmp_path, shift=1.0)
    assert policy.shift_terminal is None


def test_qwen_dynamic_overrides_carry_real_scheduler_numbers():
    # The HF-repo-id fallback must agree with the real scheduler_config.json
    # (this dict previously carried Flux's calculate_shift defaults: 1.15/4096).
    overrides = _qwen_image_dynamic_overrides()
    assert overrides["max_shift"] == pytest.approx(0.9)
    assert overrides["max_image_seq_len"] == 8192
    assert overrides["shift_terminal"] == pytest.approx(0.02)

    policy = FlowMatchSchedulePolicy._dynamic_from_overrides(1.0, overrides, "Qwen/Qwen-Image")
    assert policy.shift_terminal == pytest.approx(0.02)
    assert policy.max_shift == pytest.approx(0.9)
