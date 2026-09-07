"""Shared interception mechanics for the worker-side RL pipelines."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
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


#: Metadata group for unirl captures; vllm-omni validates only its own groups, so this cannot collide.
CAPTURE_GROUP = "unirl"
#: Private bag on ``DiffusionOutput``; flushed into ``metadata[CAPTURE_GROUP]`` after postprocess.
CAPTURE_ATTR = "_unirl_captures"


def single_request(req: Any, *, caller: str) -> Any:
    """Unwrap the one request; the GPU batch is ``num_outputs_per_prompt`` inside it, not request batching."""
    requests = getattr(req, "requests", None)
    if requests is None:
        return req
    if len(requests) != 1:
        raise RuntimeError(
            f"{caller}: expected a single-request batch (supports_request_batch=False), got {len(requests)}. "
            "Set max_num_seqs=1 on this stage."
        )
    return requests[0]


def _captures(out: Any) -> Dict[str, Any]:
    bag = getattr(out, CAPTURE_ATTR, None)
    if not isinstance(bag, dict):
        bag = {}
        setattr(out, CAPTURE_ATTR, bag)
    return bag


def stamp_capture(out: Any, key: str, value: Any) -> None:
    """Record an RL capture on the output object; postprocess never sees this bag."""
    _captures(out)[key] = value


def set_payload(out: Any, value: Any) -> None:
    """Replace the generated media; captures live on the output object, not in ``output``."""
    out.output = value


def read_captures(result: Any) -> Dict[str, Any]:
    """Driver-side inverse of :func:`stamp_capture` after the formatter flush."""
    mm = getattr(result, "multimodal_output", None) or {}
    if not isinstance(mm, dict):
        return {}
    metadata = mm.get("metadata") or {}
    if not isinstance(metadata, dict):
        return {}
    captures = metadata.get(CAPTURE_GROUP) or {}
    return captures if isinstance(captures, dict) else {}


def drain_trajectory_into(out: Any, scheduler: Any) -> None:
    """Write the SDE recordings onto ``DiffusionOutput.trajectory_*`` — the formatter's fallback channel."""
    traj = scheduler.drain_trajectory()
    if traj is None:
        return
    latents, sigmas, _timesteps, log_probs = traj
    out.trajectory_latents = latents
    out.trajectory_timesteps = sigmas
    out.trajectory_log_probs = log_probs
    stamp_capture(out, "sde_step_indices", scheduler.last_sde_step_indices)


def _adopt_payload_trajectory(out: Any, traj: Any) -> None:
    """Lift upstream payload trajectory onto ``trajectory_*`` so unwrap does not drop it."""
    if isinstance(traj, Mapping):
        if (latents := traj.get("latents")) is not None:
            out.trajectory_latents = latents
        if (timesteps := traj.get("timesteps")) is not None:
            out.trajectory_timesteps = timesteps
        if (log_probs := traj.get("log_probs")) is not None:
            out.trajectory_log_probs = log_probs
        if (decoded := traj.get("decoded")) is not None:
            out.trajectory_decoded = decoded
        return
    if torch.is_tensor(traj):
        out.trajectory_latents = traj


def finalize_output(out: Any) -> None:
    """Keep captures and SDE trajectory; unwrap media so postprocess sees a tensor."""
    envelope = getattr(out, "output", None)
    if not (isinstance(envelope, dict) and isinstance(envelope.get("payload"), dict)):
        return
    payload = envelope["payload"]
    extra = envelope.get("metadata")
    if isinstance(extra, dict) and extra:
        bag = _captures(out)
        for key, value in extra.items():
            if key not in bag:
                bag[key] = value
            elif isinstance(bag[key], dict) and isinstance(value, dict):
                bag[key] = {**value, **bag[key]}
    has_sde_traj = getattr(out, "trajectory_latents", None) is not None
    if has_sde_traj:
        payload.pop("trajectory", None)
    elif "image" in payload or "video" in payload:
        _adopt_payload_trajectory(out, payload.get("trajectory"))
    if "image" in payload:
        out.output = payload["image"]
    elif "video" in payload:
        out.output = payload["video"]


def resolve_request_noise(req: Any, *, caller: str) -> Optional[torch.Tensor]:
    """This request's driver x_T, sliced from ``[B, ...]`` by Omni's ``f'{i}_{uuid}'`` id."""
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
        ) from None

    spp = int(getattr(req.sampling_params, "num_outputs_per_prompt", 1) or 1)
    start, end = idx * spp, (idx + 1) * spp
    n = int(noise_batch.shape[0]) if noise_batch is not None else len(recipe_gids)
    if spp < 1 or not 0 <= start < end <= n:
        raise IndexError(f"{caller}: grouped slice [{start}:{end}) out of bounds for length {n} (spp={spp}).")
    if noise_batch is not None:
        return noise_batch[start:end].clone()
    return NoiseRecipe(
        noise_group_ids=[str(g) for g in recipe_gids[start:end]],
        base_seed=int(extra.get("init_noise_seed", 0)),
        latent_shape=tuple(extra["init_noise_latent_shape"]),
    ).resolve()


def inject_latents(
    target: Any,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    noise: torch.Tensor,
) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """Slot a pre-computed x_T into a ``prepare_latents`` call site."""
    bound = inspect.signature(target).bind_partial(*args, **kwargs)
    if (dtype := bound.arguments.get("dtype")) is not None:
        noise = noise.to(dtype=dtype)
    if (device := bound.arguments.get("device")) is not None:
        noise = noise.to(device=device)
    bound.arguments["latents"] = noise
    return bound.args, bound.kwargs


def flush_captures_into_postprocess(diffusion_output: Any, postprocess_output: Any) -> Any:
    """Copy the private capture bag into formatter metadata."""
    from dataclasses import replace

    captures = getattr(diffusion_output, CAPTURE_ATTR, None)
    if not (isinstance(captures, dict) and captures):
        return postprocess_output
    metadata = dict(postprocess_output.metadata)
    existing = metadata.get(CAPTURE_GROUP)
    metadata[CAPTURE_GROUP] = {**(existing if isinstance(existing, dict) else {}), **captures}
    return replace(postprocess_output, metadata=metadata)


def make_sde_scheduler(upstream_config: Any, *, eta: float = 0.0) -> FlowMatchSDEDiscreteScheduler:
    """Build the trajectory-capturing scheduler from the upstream scheduler's config — the sd3/hv15 install path."""
    return FlowMatchSDEDiscreteScheduler.from_config(upstream_config, eta=float(eta))


__all__ = [
    "CAPTURE_ATTR",
    "CAPTURE_GROUP",
    "detach_cpu",
    "detach_cpu_pair",
    "drain_trajectory_into",
    "finalize_output",
    "flush_captures_into_postprocess",
    "inject_latents",
    "make_sde_scheduler",
    "read_captures",
    "resolve_request_noise",
    "set_payload",
    "single_request",
    "stamp_capture",
]
