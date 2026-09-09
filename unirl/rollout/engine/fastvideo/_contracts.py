"""Extend FastVideo's RL contract and transition math to UniRL's kernels; contract in README.md."""

from __future__ import annotations

import functools
import math
from contextvars import ContextVar
from dataclasses import dataclass, fields
from typing import Any, Optional

import torch

_CONTEXT: ContextVar[Optional["_TransitionContext"]] = ContextVar("unirl_fastvideo_transition", default=None)


@dataclass(frozen=True)
class _TransitionContext:
    """Per-request transition data the stock loop cannot pass through its own signature."""

    eta: float
    sde_type: Any


def patch_contracts() -> None:
    """Add UniRL's resolved SDE-window fields to ``ForwardBatch.RLData`` idempotently."""
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

    original = ForwardBatch.RLData
    existing = {field.name for field in fields(original)}
    if {"sde_step_indices", "sde_type"} <= existing:
        original._unirl_fastvideo_contract = True
        return

    missing_base = {"enabled", "collect_log_probs", "store_trajectory", "keep_trajectory_on_cpu"} - existing
    if missing_base:
        raise RuntimeError(f"FastVideo RLData is incompatible; missing base fields {sorted(missing_base)}")

    @dataclass
    class UniRLFastVideoRLData(original):  # type: ignore[valid-type,misc]
        """Stock FastVideo RLData plus UniRL's resolved transition contract."""

        sde_step_indices: Optional[list[int]] = None
        sde_type: Any = "dance"

    UniRLFastVideoRLData.__name__ = "RLData"
    UniRLFastVideoRLData.__qualname__ = "ForwardBatch.RLData"
    UniRLFastVideoRLData.__module__ = original.__module__
    UniRLFastVideoRLData._unirl_fastvideo_contract = True
    ForwardBatch.RLData = UniRLFastVideoRLData


def _unirl_std_dev_t(sde_type: str, sigma: torch.Tensor, sigma_max: torch.Tensor, eta: float) -> torch.Tensor:
    """Return the UniRL kernel's diffusion coefficient; must match ``unirl/sde/kernels.py``."""
    if sde_type == "dance":
        return torch.full_like(sigma, float(eta))
    if sde_type == "flow":
        return torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))) * float(eta)
    raise ValueError(f"FastVideo UniRL transition supports sde_type 'flow' or 'dance'; got {sde_type!r}")


def _sde_step_with_logprob(
    scheduler,
    model_output: torch.Tensor,
    timestep: Any,
    sample: torch.Tensor,
    prev_sample: Optional[torch.Tensor] = None,
    generator: Any = None,
    deterministic: bool = False,
    return_pixel_log_prob: bool = False,
    return_dt_and_std_dev_t: bool = False,
    eta: Optional[float] = None,
    sde_type: Any = None,
):
    """Apply the same Gaussian transition UniRL's trainer replays; stock upstream uses a different law."""
    if return_pixel_log_prob:
        raise NotImplementedError("FastVideo UniRL patch does not expose pixel-level log probabilities")
    if prev_sample is not None and generator is not None:
        raise ValueError("prev_sample and generator are mutually exclusive")

    context = _CONTEXT.get()
    resolved_eta = float(eta if eta is not None else (context.eta if context is not None else 0.0))
    raw_type = sde_type if sde_type is not None else (context.sde_type if context is not None else "dance")
    kernel = str(getattr(raw_type, "sde_type", raw_type)).strip().lower()

    if isinstance(timestep, torch.Tensor):
        values = timestep.reshape(-1).tolist() if timestep.ndim else [timestep.item()]
    else:
        values = [timestep]
    indices = [int(scheduler.index_for_timestep(v)) for v in values]

    sigmas = scheduler.sigmas.to(device=sample.device, dtype=sample.dtype)
    view = (-1, *([1] * (sample.ndim - 1)))
    sigma = sigmas[indices].reshape(view)
    sigma_next = sigmas[[i + 1 for i in indices]].reshape(view)
    dt = sigma_next - sigma
    sqrt_dt = torch.sqrt(-dt)

    if deterministic:
        result = sample + dt * model_output
        zeros = torch.zeros(sample.shape[0], device=sample.device, dtype=sample.dtype)
        std = torch.zeros_like(sigma)
        return (result, zeros, result, std, sqrt_dt) if return_dt_and_std_dev_t else (result, zeros, result, std)

    std_dev_t = _unirl_std_dev_t(kernel, sigma, sigmas[1].reshape(1, *([1] * (sample.ndim - 1))), resolved_eta)
    prev_sample_mean = (
        sample * (1 + std_dev_t.square() / (2 * sigma) * dt)
        + model_output * (1 + std_dev_t.square() * (1 - sigma) / (2 * sigma)) * dt
    )

    if prev_sample is None:
        from fastvideo.pipelines.stages.denoising import randn_tensor

        noise = randn_tensor(
            model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype
        )
        prev_sample = prev_sample_mean + std_dev_t * sqrt_dt * noise

    std_var = std_dev_t * sqrt_dt
    log_prob = (
        -((prev_sample.detach() - prev_sample_mean).square()) / (2 * std_var.square())
        - torch.log(std_var)
        - 0.5 * math.log(2 * math.pi)
    )
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    if return_dt_and_std_dev_t:
        return prev_sample, log_prob, prev_sample_mean, std_dev_t, sqrt_dt
    return prev_sample, log_prob, prev_sample_mean, std_var


_sde_step_with_logprob._unirl_fastvideo_sde = True  # type: ignore[attr-defined]


def patch_transition() -> None:
    """Install UniRL's transition kernel and pass request-local eta/sde_type into the stock loop."""
    from fastvideo.pipelines.stages import denoising

    if not getattr(denoising.sde_step_with_logprob, "_unirl_fastvideo_sde", False):
        denoising.sde_step_with_logprob = _sde_step_with_logprob

    original_forward = denoising.DenoisingStage.forward
    if getattr(original_forward, "_unirl_fastvideo_denoising", False):
        return

    @functools.wraps(original_forward)
    def forward(self, batch, fastvideo_args):
        rl_data = batch.rl_data if batch.rl_data is not None and batch.rl_data.enabled else None
        if rl_data is None:
            return original_forward(self, batch, fastvideo_args)
        collect = bool(rl_data.collect_log_probs)
        # The stock loop gates its SDE branch on this flag; force it so replay mode also
        # walks the UniRL kernel, then drop the log-probs the caller did not ask for.
        rl_data.collect_log_probs = True
        token = _CONTEXT.set(_TransitionContext(eta=float(batch.eta), sde_type=getattr(rl_data, "sde_type", "dance")))
        try:
            result = original_forward(self, batch, fastvideo_args)
            if not collect and result.rl_data is not None:
                result.rl_data.log_probs = None
            return result
        finally:
            _CONTEXT.reset(token)
            rl_data.collect_log_probs = collect

    forward._unirl_fastvideo_denoising = True
    denoising.DenoisingStage.forward = forward


__all__ = ["patch_contracts", "patch_transition"]
