"""Driver-authoritative x_T packing for the single-stage DiT request builders."""

from __future__ import annotations

from typing import Any, Dict

from unirl.types.sample import Part
from unirl.types.segments.latent import LatentSegment


def pack_initial_noise_extra_args(
    extra_args: Dict[str, Any],
    gen_part: Part,
    diff_params: Any,
    *,
    caller: str,
) -> None:
    """Pack the per-sample x_T — a ``[B, C, H, W]`` ``initial_noise_batch`` tensor or a recipe — into ``extra_args``."""
    n_samples = len(gen_part.sample_ids)
    if diff_params.disable_driver_xt:
        return
    seg = gen_part.segment
    initial_latents = seg.initial_latents if isinstance(seg, LatentSegment) else None
    if initial_latents is not None:
        if int(initial_latents.shape[0]) != n_samples:
            raise RuntimeError(
                f"{caller}: initial_latents.shape[0]={int(initial_latents.shape[0])} "
                f"!= diffusion sample count {n_samples} after sharding."
            )
        extra_args["initial_noise_batch"] = initial_latents
    elif diff_params.init_noise_latent_shape:
        explicit_keys = list(gen_part.init_noise_group_ids)
        if explicit_keys:
            if len(explicit_keys) != n_samples:
                raise RuntimeError(
                    f"{caller}: init_noise_group_ids count {len(explicit_keys)} "
                    f"!= diffusion sample count {n_samples} after sharding."
                )
            keys = explicit_keys
        else:
            keys = gen_part.group_ids if diff_params.init_same_noise else list(gen_part.sample_ids)
        extra_args["init_noise_group_ids"] = keys
        extra_args["init_noise_latent_shape"] = list(diff_params.init_noise_latent_shape)
        extra_args["init_noise_seed"] = diff_params.seed if diff_params.seed is not None else 0


__all__ = ["pack_initial_noise_extra_args"]
