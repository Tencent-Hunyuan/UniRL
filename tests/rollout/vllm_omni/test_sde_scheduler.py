"""Shared SDE scheduler: externally pinned σ are FINAL values.

Diffusers' ``set_timesteps`` mutates passed-in sigmas in up to three
places — the dynamic-shift branch, the static-shift branch, and the
``shift_terminal`` whole-schedule stretch. The shared scheduler must
neutralize all three so the worker echoes exactly what the engine
pinned (the σ-echo gate compares them at atol=1e-5).

The shift_terminal case is the LIN-382 qwen smoke regression: the
Qwen-Image-2512 checkpoint ships ``shift_terminal: 0.02``, which
stretched pinned [.., 0.3584] to [.., 0.0200] on the worker.
"""

from __future__ import annotations

import torch

from unirl.rollout.engine.vllm_omni.pipelines._shared.flow_match_sde_scheduler import (
    FlowMatchSDEDiscreteScheduler,
)

# The Qwen-Image-2512 scheduler_config.json shape (the regression source).
_QWEN_2512_CONFIG = {
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "use_dynamic_shifting": True,
    "base_shift": 0.5,
    "max_shift": 0.9,
    "base_image_seq_len": 256,
    "max_image_seq_len": 8192,
    "shift_terminal": 0.02,
    "time_shift_type": "exponential",
}

# SD3.5-medium-style config (static shift, no terminal stretch).
_SD35_CONFIG = {
    "num_train_timesteps": 1000,
    "shift": 3.0,
    "use_dynamic_shifting": False,
}

_PINNED = [0.8341, 0.6262, 0.3584]  # T-length, terminal stripped (wire shape)


def _assert_verbatim(scheduler) -> None:
    got = scheduler.sigmas.to(torch.float64)
    want = torch.tensor(_PINNED + [0.0], dtype=torch.float64)
    assert torch.allclose(got, want, atol=1e-6), f"sigmas mutated: {got.tolist()} != {want.tolist()}"


def test_injected_sigmas_survive_shift_terminal():
    s = FlowMatchSDEDiscreteScheduler.from_config(dict(_QWEN_2512_CONFIG))
    s.set_timesteps(sigmas=list(_PINNED), mu=0.516, device="cpu")
    _assert_verbatim(s)
    # finally-restore: the config is unchanged for any non-injected caller.
    assert s.config.shift_terminal == 0.02
    assert s.config.use_dynamic_shifting is True


def test_injected_sigmas_survive_static_shift():
    s = FlowMatchSDEDiscreteScheduler.from_config(dict(_SD35_CONFIG))
    s.set_timesteps(sigmas=list(_PINNED), device="cpu")
    _assert_verbatim(s)
    # finally-restore: the CONFIG is untouched (``s.shift`` is a property
    # over the instance attr ``_shift``, which diffusers' from_config never
    # seeds from config.shift — asserting it would test diffusers, not us).
    assert float(s.config.shift) == 3.0
