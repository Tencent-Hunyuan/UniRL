"""Driver-authoritative x_T packing for the single-stage DiT request builders.

Shared verbatim by the ``sd35_t2i`` and ``t2v`` adapters (the HI3 shapes pack
their recipe-only variant inline — their DiT latent shape is AR-dynamic, so a
materialized tensor is rejected there).
"""

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
    """Pack the per-sample x_T (tensor or recipe) into ``extra_args`` in place.

    - ``gen_part.segment.initial_latents`` (the img2img x_T shell, a CONCAT
      field so it is sliced correctly under multi-actor sharding) → a single
      ``[B, C, H, W]`` ``initial_noise_batch`` tensor; the worker pipeline's
      ``prepare_latents`` slices its row by request index.
    - else ``DiffusionSamplingParams.init_noise_latent_shape`` (the trainer's
      x_T-recipe opt-in) → the x_T RECIPE; the worker regenerates each gid's
      noise on CPU-fp32. The noise key is derived from the gen Part's lineage
      (OD-2): the parent (group) id under ``init_same_noise`` so siblings share
      x_T, else the per-sample id.
    - ``disable_driver_xt=True`` → pack neither transport; the worker owns x_T.

    Batch-dim mismatches indicate an upstream slicing bug — fail fast here
    instead of silently mis-slicing inside the worker.
    """
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
