"""Driver-authoritative x_T packing for the single-stage DiT request builders.

Shared verbatim by the ``sd35_t2i`` and ``t2v`` adapters (the HI3 shapes pack
their recipe-only variant inline — their DiT latent shape is AR-dynamic, so a
materialized tensor is rejected there).
"""

from __future__ import annotations

from typing import Any, Dict

from unirl.types.rollout_req import RolloutReq


def pack_initial_noise_extra_args(
    extra_args: Dict[str, Any],
    req: RolloutReq,
    diff_params: Any,
    *,
    n_prompts: int,
    caller: str,
) -> None:
    """Pack the per-sample x_T (tensor or recipe) into ``extra_args`` in place.

    - ``request_conditions['initial_latents']`` → a single ``[B, C, H, W]``
      ``initial_noise_batch`` tensor; the worker pipeline's ``prepare_latents``
      slices its row by request index. Sourced from a CONCAT field so it's
      sliced correctly under multi-actor sharding.
    - else ``req.init_noise_group_ids`` (+ ``init_noise_latent_shape``) → the
      x_T RECIPE; the worker regenerates each gid's noise on CPU-fp32.

    Batch-dim mismatches indicate an upstream slicing bug — fail fast here
    instead of silently mis-slicing inside the worker.
    """
    initial_latent_cond = (req.request_conditions or {}).get("initial_latents")
    if initial_latent_cond is not None:
        initial_noise = initial_latent_cond.latents
        if int(initial_noise.shape[0]) != n_prompts:
            raise RuntimeError(
                f"{caller}: initial_latents.shape[0]={int(initial_noise.shape[0])} "
                f"!= prompt count {n_prompts} after sharding."
            )
        # Tensor stays on whatever device the caller left it (typically CPU);
        # the worker pipeline does the device move inside ``prepare_latents``.
        extra_args["initial_noise_batch"] = initial_noise
    elif req.init_noise_group_ids and req.init_noise_latent_shape:
        if len(req.init_noise_group_ids) != n_prompts:
            raise RuntimeError(
                f"{caller}: init_noise_group_ids len {len(req.init_noise_group_ids)} "
                f"!= prompt count {n_prompts} after sharding."
            )
        extra_args["init_noise_group_ids"] = [str(g) for g in req.init_noise_group_ids]
        extra_args["init_noise_latent_shape"] = [int(x) for x in req.init_noise_latent_shape]
        extra_args["init_noise_seed"] = int(diff_params.seed) if getattr(diff_params, "seed", None) is not None else 0


__all__ = ["pack_initial_noise_extra_args"]
