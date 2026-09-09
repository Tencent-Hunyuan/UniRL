"""Shared σ round-trip verifier — our σ live in ``[0, 1]``; the scale vs a 1000-based echo is detected."""

from __future__ import annotations

from typing import Any, Optional

import torch


def _to_cpu_float32(t: Any) -> torch.Tensor:
    """Coerce ``t`` (Tensor / array-like) → CPU float32 tensor."""
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
    """Assert ``actual`` (worker-echoed σ) matches ``expected`` (engine-sent σ)."""
    if expected is None:
        return
    if actual is None:
        raise RuntimeError(
            f"{engine_name}: worker did not echo trajectory_timesteps. "
            f"Cannot verify the σ schedule pinned on the diffusion Part "
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
            f"Worker did not use the σ schedule pinned on the diffusion Part. "
            f"Verify the engine→worker wiring threads sampling_params.sigmas into "
            f"scheduler.set_timesteps(sigmas=...) without modification."
        )

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
