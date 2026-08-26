"""Shared interception mechanics for the worker-side RL pipelines."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from unirl.rollout.engine.vllm_omni.pipelines._shared.flow_match_sde_scheduler import (
    FlowMatchSDEDiscreteScheduler,
)
from unirl.types.noise_recipe import NoiseRecipe


def detach_cpu(t: Any) -> Any:
    """Detach + move to CPU for IPC transport. ``None``/non-tensor passthrough."""
    if isinstance(t, torch.Tensor):
        return t.detach().to("cpu")
    return t


def detach_cpu_pair(p: Any) -> Any:
    """``(cos, sin)`` rope-cache pair handler. Pass-through otherwise."""
    if isinstance(p, tuple) and len(p) == 2:
        return (detach_cpu(p[0]), detach_cpu(p[1]))
    return p


def stamp_custom_output(out: Any, key: str, value: Any) -> None:
    """Write onto ``DiffusionOutput.custom_output``."""
    if out.custom_output is None:
        out.custom_output = {}
    out.custom_output[key] = value


def drain_trajectory_into(out: Any, scheduler: Any) -> None:
    """Harvest the SDE scheduler's recordings; ``trajectory_timesteps`` is the true ``[0, 1]`` sigma schedule."""
    traj = scheduler.drain_trajectory()
    if traj is None:
        return
    latents, sigmas, _timesteps, log_probs = traj
    out.trajectory_latents = latents
    out.trajectory_timesteps = sigmas
    out.trajectory_log_probs = log_probs
    stamp_custom_output(out, "sde_step_indices", scheduler.last_sde_step_indices)


def _grouped_span(idx: int, spp: int) -> tuple[int, int]:
    spp = int(spp or 1)
    if spp < 1:
        raise ValueError(f"_grouped_span: spp must be >= 1, got {spp}")
    start = int(idx) * spp
    return start, start + spp


def resolve_request_noise(req: Any, *, caller: str) -> Optional[torch.Tensor]:
    """This request's driver x_T, sliced from ``[B, ...]`` keeping its leading ``[num_outputs_per_prompt, ...]``."""
    extra = getattr(req.sampling_params, "extra_args", None) or {}
    noise_batch = extra.get("initial_noise_batch")
    recipe_gids = extra.get("init_noise_group_ids")
    if noise_batch is None and not recipe_gids:
        return None

    rid = str(getattr(req, "request_id", "") or "")
    try:
        idx = int(rid.split("_", 1)[0])
    except ValueError:
        raise RuntimeError(
            f"{caller}: cannot parse batch index from request_id={rid!r}. Expected Omni's ``f'{{i}}_{{uuid}}'`` shape."
        )

    spp = int(getattr(req.sampling_params, "num_outputs_per_prompt", 1) or 1)
    start, end = _grouped_span(idx, spp)

    if noise_batch is not None:
        if start < 0 or end > int(noise_batch.shape[0]):
            raise IndexError(
                f"{caller}: grouped slice [{start}:{end}) out of bounds for "
                f"noise_batch.shape[0]={int(noise_batch.shape[0])}."
            )
        return noise_batch[start:end].clone()

    if start < 0 or end > len(recipe_gids):
        raise IndexError(
            f"{caller}: grouped slice [{start}:{end}) out of bounds for init_noise_group_ids len={len(recipe_gids)}."
        )
    return NoiseRecipe(
        noise_group_ids=[str(g) for g in recipe_gids[start:end]],
        base_seed=int(extra.get("init_noise_seed", 0)),
        latent_shape=tuple(extra["init_noise_latent_shape"]),
    ).resolve()


def inject_latents(
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    noise: torch.Tensor,
    *,
    dtype_idx: int = 4,
    device_idx: int = 5,
    latents_idx: int = 7,
) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """Slot a pre-computed x_T into a ``prepare_latents`` call site."""
    dtype = args[dtype_idx] if len(args) > dtype_idx else kwargs.get("dtype")
    device = args[device_idx] if len(args) > device_idx else kwargs.get("device")
    if dtype is not None:
        noise = noise.to(dtype=dtype)
    if device is not None:
        noise = noise.to(device=device)
    if len(args) >= latents_idx + 1:
        args = (*args[:latents_idx], noise, *args[latents_idx + 1 :])
    else:
        kwargs = {**kwargs, "latents": noise}
    return args, kwargs


def make_sde_scheduler(upstream_config: Any, *, eta: float = 0.0) -> FlowMatchSDEDiscreteScheduler:
    """Build the trajectory-capturing scheduler from the upstream scheduler's config — the sd3/hv15 install path."""
    return FlowMatchSDEDiscreteScheduler.from_config(upstream_config, eta=float(eta))


__all__ = [
    "detach_cpu",
    "detach_cpu_pair",
    "drain_trajectory_into",
    "_grouped_span",
    "inject_latents",
    "make_sde_scheduler",
    "resolve_request_noise",
    "stamp_custom_output",
]
