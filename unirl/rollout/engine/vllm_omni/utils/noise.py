"""Driver-authoritative x_T packing for the single-stage DiT request builders."""

from __future__ import annotations

from typing import Any, Dict

from unirl.types.sample import Part


def pack_initial_noise_extra_args(
    extra_args: Dict[str, Any],
    gen_part: Part,
    diff_params: Any,
    *,
    caller: str,
) -> None:
    """Pack the per-sample x_T — a ``[B, C, H, W]`` ``initial_noise_batch`` tensor or a recipe — into ``extra_args``."""
    n_samples = len(gen_part.sample_ids)
    if bool(getattr(diff_params, "disable_driver_xt", False)):
        return
    seg = gen_part.segment
    initial_latents = getattr(seg, "initial_latents", None) if seg is not None else None
    if initial_latents is not None:
        if int(initial_latents.shape[0]) != n_samples:
            raise RuntimeError(
                f"{caller}: initial_latents.shape[0]={int(initial_latents.shape[0])} "
                f"!= diffusion sample count {n_samples} after sharding."
            )
        extra_args["initial_noise_batch"] = initial_latents
    elif diff_params.init_noise_latent_shape:
        share = bool(getattr(diff_params, "init_same_noise", False))
        keys = gen_part.group_ids if share else list(gen_part.sample_ids)
        extra_args["init_noise_group_ids"] = [str(k) for k in keys]
        extra_args["init_noise_latent_shape"] = [int(x) for x in diff_params.init_noise_latent_shape]
        extra_args["init_noise_seed"] = int(diff_params.seed) if getattr(diff_params, "seed", None) is not None else 0


__all__ = ["pack_initial_noise_extra_args"]
