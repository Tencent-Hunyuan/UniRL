"""Regression: external (driver-pinned) sigmas pass through ``set_timesteps`` verbatim.

The ``_patches`` suite re-hosts the fork's "external sigmas are final" contract
on stock upstream sglang. Three schedule-mutation paths must be neutralized for
external sigmas: the static shift, the dynamic mu shift, and — the Qwen-Image
case — the terminal stretch (``shift_terminal: 0.02``; gated upstream only on
the config value, not on external sigmas). Internally-computed schedules must
keep all three (the patch must not over-neutralize the serving default).

CPU-only; skipped when sglang[diffusion] is not installed (mac dev hosts).
"""

from __future__ import annotations

import pytest

sched_mod = pytest.importorskip(
    "sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete",
    reason="sglang[diffusion] not installed (linux-only dependency)",
)
torch = pytest.importorskip("torch")

from unirl.rollout.engine.sglang._patches.patch_set_timesteps import (  # noqa: E402
    patch_set_timesteps,
)

#: Interior-T schedule (terminal 0 is appended by ``set_timesteps`` itself).
#: Dyadic rationals — exactly representable in float32, so verbatim
#: passthrough can be asserted bit-exactly.
_SIGMAS = [1.0, 0.875, 0.625, 0.25, 0.0625]


def _qwen_scheduler():
    """Mirror ``Qwen/Qwen-Image`` scheduler_config.json (``shift_terminal: 0.02``)."""
    return sched_mod.FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=1.0,
        use_dynamic_shifting=True,
        base_shift=0.5,
        max_shift=0.9,
        base_image_seq_len=256,
        max_image_seq_len=8192,
        shift_terminal=0.02,
        time_shift_type="exponential",
    )


def _sd3_scheduler():
    """Static-shift config, ``shift_terminal`` null — the path that always worked."""
    return sched_mod.FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)


@pytest.fixture(autouse=True)
def _patched():
    patch_set_timesteps()  # idempotent (sentinel-guarded)


def test_external_sigmas_skip_terminal_stretch_qwen_config():
    s = _qwen_scheduler()
    s.set_timesteps(sigmas=list(_SIGMAS), mu=0.8)
    got = s.sigmas[: len(_SIGMAS)]
    torch.testing.assert_close(got, torch.tensor(_SIGMAS, dtype=torch.float32), atol=0.0, rtol=0.0)
    assert float(s.sigmas[-1]) == 0.0  # terminal zero still appended


def test_external_sigmas_skip_static_shift_sd3_config():
    s = _sd3_scheduler()
    s.set_timesteps(sigmas=list(_SIGMAS))
    got = s.sigmas[: len(_SIGMAS)]
    torch.testing.assert_close(got, torch.tensor(_SIGMAS, dtype=torch.float32), atol=0.0, rtol=0.0)


def test_internal_schedule_keeps_terminal_stretch():
    # No external sigmas → the stock path must still stretch to shift_terminal
    # (stretch maps the last interior sigma to exactly shift_terminal).
    s = _qwen_scheduler()
    s.set_timesteps(num_inference_steps=8, mu=0.8)
    interior_last = float(s.sigmas[-2])  # last nonzero sigma; terminal 0 follows
    assert interior_last == pytest.approx(0.02, abs=1e-6)


def test_stretch_shadow_restored_after_call():
    s = _qwen_scheduler()
    s.set_timesteps(sigmas=list(_SIGMAS), mu=0.8)
    # The identity instance-binding must not leak past the call.
    assert "stretch_shift_to_terminal" not in s.__dict__
