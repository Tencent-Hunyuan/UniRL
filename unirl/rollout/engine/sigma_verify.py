"""Shared σ-schedule round-trip verifier for rollout engines.

Single helper used by every rollout engine adapter (sglang, vllm-omni)
to assert the worker echoed back the exact σ schedule the engine sent.

Contract
--------

The engine adapter (main side):
    1. Pins ``RolloutReq.sigmas`` via
       :func:`unirl.sde.runtime.ensure_req_sigmas` (which applies
       the engine's :class:`FlowMatchSchedulePolicy` to the per-request
       ``(T, H, W)`` triple).
    2. Forwards ``req.sigmas`` to the worker (SGLang's
       ``DiffusionSamplingParams.sigmas`` / vllm-omni's
       ``OmniDiffusionSamplingParams.sigmas``).
    3. Worker calls ``scheduler.set_timesteps(sigmas=...)`` so the loop
       uses the schedule verbatim.
    4. Worker echoes the actual scheduler-stored σ back via
       ``trajectory_timesteps``.

This module's :func:`verify_engine_used_sigmas` enforces step 4 ==
step 1: any drift surfaces as a ``RuntimeError`` at the rollout→trainer
boundary instead of silently de-syncing the GRPO log-prob ratio.

Scale normalization
-------------------

Our σ live in ``[0, 1]`` (FlowMatch normalized). Some sglang builds emit
the *un-normalized* form ``sigma * num_train_timesteps`` (e.g.
``[1000, 750, 0]`` for SD3 / FLUX) directly out of
``multimodal_gen.denoising`` instead of normalized ``[1, 0.75, 0]``.
Same schedule, different unit.

Detection is **dynamic** (ported from main-repo commit ``43642ac1``
"fix(sglang): tolerate scaled trajectory_timesteps"): any value with
absolute magnitude > 10 cannot be a normalized σ (those are in
``[0, 1]``), so we compute ``scale = round(actual_max / expected_max)``
and divide by it. Hardcoding ``/ 1000`` would have broken on any model
that ships a non-1000 ``num_train_timesteps`` (e.g. some research
variants use 500 or 4000); the dynamic ratio handles all of them.

Any non-integer ratio means a *genuine* schedule drift — ``round()``
collapses it to the nearest integer scale, but the subsequent
``allclose`` is the actual guard: it surfaces drift regardless of
which scale layer it lives in.
"""

from __future__ import annotations

from typing import Any, Optional

import torch


def _to_cpu_float32(t: Any) -> torch.Tensor:
    """Coerce ``t`` (Tensor / array-like) → CPU float32 tensor.

    Defensive: detach() to break autograd refs, .cpu() to free worker
    device pointers, .float() to compare against our reference uniformly.
    """
    if torch.is_tensor(t):
        return t.detach().cpu().to(torch.float32)
    return torch.as_tensor(t, dtype=torch.float32)


def verify_engine_used_sigmas(
    actual: Any,
    *,
    expected: Optional[torch.Tensor],
    engine_name: str,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> None:
    """Assert ``actual`` (worker-echoed σ) matches ``expected`` (engine-sent σ).

    Args:
        actual: The σ schedule the worker actually used and echoed back —
            ``trajectory_timesteps`` on the per-result / DiffusionOutput.
            May be ``None`` for legacy results that don't surface it;
            we then raise rather than silently pass (silent agreement
            on σ is unsafe).
        expected: ``RolloutReq.sigmas`` (engine pinned). ``None`` skips
            the check — legacy callers that bypass
            :func:`unirl.sde.runtime.ensure_req_sigmas` keep their
            pre-existing behavior.
        engine_name: Used in error messages to point at the right wire
            (``"sglang"`` / ``"vllm-omni"`` etc.).
        atol, rtol: ``torch.allclose`` tolerances.

    Raises:
        RuntimeError: on shape mismatch, missing tensor, or value drift.
    """
    if expected is None:
        return
    if actual is None:
        raise RuntimeError(
            f"{engine_name}: worker did not echo trajectory_timesteps. "
            f"Cannot verify the σ schedule the engine pinned on req.sigmas "
            f"was actually used. Either upgrade the worker to emit "
            f"trajectory_timesteps or pin a build that does — silent "
            f"agreement on σ is not safe (GRPO log-prob ratio drifts away "
            f"from 1.0)."
        )
    actual_t = _to_cpu_float32(actual)
    expected_f32 = expected.detach().to(torch.float32).cpu()

    # Shape mismatch is a definitive wiring break, regardless of scale.
    if actual_t.shape != expected_f32.shape:
        raise RuntimeError(
            f"{engine_name}: trajectory_timesteps shape mismatch — got "
            f"{tuple(actual_t.shape)}, sent {tuple(expected_f32.shape)}. "
            f"Worker did not use the σ schedule pinned on req.sigmas. "
            f"Verify the engine→worker wiring threads req.sigmas into "
            f"scheduler.set_timesteps(sigmas=...) without modification."
        )

    # Dynamic scale normalization (ported from main-repo commit 43642ac1).
    # When the worker echoes raw timesteps `sigma * num_train_timesteps`
    # (some sglang builds), fold them back to the [0, 1] scale by the
    # integer ratio of max abs values. See module docstring.
    actual_max = float(actual_t.abs().max().item()) if actual_t.numel() > 0 else 0.0
    expected_max = float(expected_f32.abs().max().item()) if expected_f32.numel() > 0 else 0.0
    if actual_max > 10.0 and expected_max > 0:
        scale = round(actual_max / expected_max)
        if scale > 0:
            actual_t = actual_t / float(scale)

    if not torch.allclose(actual_t, expected_f32, atol=atol, rtol=rtol):
        max_diff = (actual_t - expected_f32).abs().max().item()
        raise RuntimeError(
            f"{engine_name}: trajectory_timesteps value mismatch — max abs "
            f"diff={max_diff:.3e} (atol={atol:g}, rtol={rtol:g}). "
            f"Engine sent (head): {expected_f32.tolist()[:5]}; "
            f"worker returned (head): {actual_t.tolist()[:5]}. "
            f"Worker did NOT use the σ we sent — verify "
            f"scheduler.set_timesteps(sigmas=...) is actually called on "
            f"req.sampling_params.sigmas in the worker pipeline."
        )


__all__ = ["verify_engine_used_sigmas"]
