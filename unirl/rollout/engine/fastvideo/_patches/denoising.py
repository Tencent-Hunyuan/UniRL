"""UniRL-owned Flow/Dance transition math for FastVideo RL rollouts."""

from __future__ import annotations

import math
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from typing import Any

import torch


@dataclass(frozen=True)
class _TransitionContext:
    eta: float
    sde_type: str
    sde_step_indices: frozenset[int] | None
    collect_kl: bool
    timestep_dtype: str | None
    generator: Any


_CONTEXT: ContextVar[_TransitionContext | None] = ContextVar("unirl_fastvideo_transition", default=None)


def _step_indices(scheduler: Any, timestep: float | torch.Tensor) -> list[int]:
    if isinstance(timestep, torch.Tensor):
        values = timestep.reshape(-1).tolist()
    else:
        values = [timestep]
    return [int(scheduler.index_for_timestep(value)) for value in values]


def _sde_step_with_logprob(
    scheduler: Any,
    model_output: torch.Tensor,
    timestep: float | torch.Tensor,
    sample: torch.Tensor,
    prev_sample: torch.Tensor | None = None,
    generator: torch.Generator | list[torch.Generator] | None = None,
    deterministic: bool = False,
    return_pixel_log_prob: bool = False,
    return_dt_and_std_dev_t: bool = False,
    eta: float | None = None,
    sde_type: str | None = None,
) -> tuple[torch.Tensor, ...]:
    """Apply the same Gaussian transition used by UniRL trainer replay."""

    if return_pixel_log_prob:
        raise NotImplementedError("FastVideo UniRL patch does not expose pixel-level log probabilities")
    if prev_sample is not None and generator is not None:
        raise ValueError("prev_sample and generator are mutually exclusive")

    context = _CONTEXT.get()
    if generator is None and prev_sample is None and context is not None:
        generator = context.generator
    resolved_eta = float(eta if eta is not None else (context.eta if context is not None else 0.3))
    kernel = str(sde_type or (context.sde_type if context is not None else "dance")).strip().lower()
    indices = _step_indices(scheduler, timestep)

    sigmas = scheduler.sigmas.to(device=sample.device, dtype=sample.dtype)
    sigma = sigmas[indices].reshape(-1, *([1] * (sample.ndim - 1)))
    sigma_next = sigmas[[index + 1 for index in indices]].reshape(-1, *([1] * (sample.ndim - 1)))
    dsigma = sigma_next - sigma
    delta_t = sigma - sigma_next

    is_sde_step = (
        context is None
        or context.sde_step_indices is None
        or all(index in context.sde_step_indices for index in indices)
    )
    if deterministic or not is_sde_step:
        if context is not None and context.collect_kl and not is_sde_step:
            raise RuntimeError("FastVideo collect_kl is unsupported on deterministic tail transitions")
        deterministic_sample = sample + dsigma * model_output
        zeros = torch.zeros(sample.shape[0], device=sample.device, dtype=sample.dtype)
        transition_std = torch.zeros_like(sigma)
        sqrt_dt = torch.sqrt(torch.clamp(delta_t, min=0))
        result = (deterministic_sample, zeros, deterministic_sample, transition_std)
        return (*result, sqrt_dt) if return_dt_and_std_dev_t else result

    if kernel == "flow":
        sigma_max = sigmas[1].reshape(1, *([1] * (sample.ndim - 1)))
        sigma_for_denom = torch.where(sigma == 1, sigma_max, sigma)
        diffusion_coeff = torch.sqrt(sigma / (1 - sigma_for_denom)) * resolved_eta
    elif kernel == "dance":
        diffusion_coeff = torch.full_like(sigma, resolved_eta)
    else:
        raise ValueError(f"FastVideo UniRL patch supports sde_type 'flow' or 'dance'; got {kernel!r}")

    prev_sample_mean = (
        sample * (1 + diffusion_coeff.square() / (2 * sigma) * dsigma)
        + model_output * (1 + diffusion_coeff.square() * (1 - sigma) / (2 * sigma)) * dsigma
    )
    sqrt_dt = torch.sqrt(delta_t)
    transition_std = diffusion_coeff * sqrt_dt

    if prev_sample is None:
        from fastvideo.pipelines.stages.denoising import randn_tensor

        variance_noise = randn_tensor(
            model_output.shape,
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )
        prev_sample = prev_sample_mean + transition_std * variance_noise

    log_prob = (
        -((prev_sample.detach() - prev_sample_mean).square()) / (2 * transition_std.square())
        - torch.log(transition_std + 1e-8)
        - math.log(math.sqrt(2 * math.pi))
    )
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    result = (prev_sample, log_prob, prev_sample_mean, transition_std)
    return (*result, sqrt_dt) if return_dt_and_std_dev_t else result


setattr(_sde_step_with_logprob, "_unirl_fastvideo_sde", True)


def patch_denoising() -> None:
    """Install transition math and pass request-local SDE context to stock loops."""

    from fastvideo.pipelines.stages import denoising

    denoising.sde_step_with_logprob = _sde_step_with_logprob

    original_forward = denoising.DenoisingStage.forward
    if getattr(original_forward, "_unirl_fastvideo_denoising", False):
        return

    @wraps(original_forward)
    def forward(self, batch, fastvideo_args):
        rl_data = batch.rl_data if batch.rl_data is not None and batch.rl_data.enabled else None
        if rl_data is None:
            return original_forward(self, batch, fastvideo_args)

        raw_indices = getattr(rl_data, "sde_step_indices", None)
        indices = None if raw_indices is None else frozenset(int(index) for index in raw_indices)
        collect_log_probs = bool(rl_data.collect_log_probs)
        guidance_scale = batch.guidance_scale
        guidance_scale_2 = getattr(batch, "guidance_scale_2", None)
        if getattr(fastvideo_args, "_unirl_cfg_combine_dtype", None) == "float32":
            device = batch.latents.device
            batch.guidance_scale = torch.as_tensor(guidance_scale, dtype=torch.float32, device=device)
            if guidance_scale_2 is not None:
                batch.guidance_scale_2 = torch.as_tensor(guidance_scale_2, dtype=torch.float32, device=device)
        # The pinned stock loop gates the SDE transition itself on this flag.
        # Force that branch for replay too, then discard the unrequested logp.
        rl_data.collect_log_probs = True
        token = _CONTEXT.set(
            _TransitionContext(
                eta=float(batch.eta),
                sde_type=str(getattr(rl_data, "sde_type", "dance")),
                sde_step_indices=indices,
                collect_kl=bool(getattr(rl_data, "collect_kl", False)),
                timestep_dtype=getattr(fastvideo_args, "_unirl_timestep_dtype", None),
                generator=batch.generator,
            )
        )
        try:
            result = original_forward(self, batch, fastvideo_args)
            if not collect_log_probs and result.rl_data is not None:
                result.rl_data.log_probs = None
            return result
        finally:
            rl_data.collect_log_probs = collect_log_probs
            batch.guidance_scale = guidance_scale
            if guidance_scale_2 is not None:
                batch.guidance_scale_2 = guidance_scale_2
            _CONTEXT.reset(token)

    setattr(forward, "_unirl_fastvideo_denoising", True)
    denoising.DenoisingStage.forward = forward

    # Wan consumes integer diffusion timesteps during UniRL replay. Keep this
    # request-local and adapter-controlled instead of using a process-global
    # environment variable in FastVideo's shared DenoisingStage.
    from fastvideo.models.dits.wanvideo import WanTransformer3DModel

    transformer_forward = WanTransformer3DModel.forward
    if not getattr(transformer_forward, "_unirl_fastvideo_timestep", False):

        @wraps(transformer_forward)
        def wan_forward(self, *args, **kwargs):
            context = _CONTEXT.get()
            if context is not None and context.timestep_dtype == "long":
                args = list(args)
                if "timestep" in kwargs and torch.is_tensor(kwargs["timestep"]):
                    kwargs["timestep"] = kwargs["timestep"].to(torch.long)
                elif len(args) >= 3 and torch.is_tensor(args[2]):
                    args[2] = args[2].to(torch.long)
                args = tuple(args)
            return transformer_forward(self, *args, **kwargs)

        setattr(wan_forward, "_unirl_fastvideo_timestep", True)
        WanTransformer3DModel.forward = wan_forward


__all__ = ["patch_denoising"]
