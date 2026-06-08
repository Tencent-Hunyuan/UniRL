"""``NoiseRecipe`` — the normalized, engine-agnostic x_T recipe.

Every diffusion model gets a driver-authoritative initial latent (x_T) from the
SAME four-element recipe — ``(noise_group_ids, base_seed, latent_shape,
initial_latents)`` — regardless of which request class carries it (the driver's
:class:`~unirl.types.rollout_req.RolloutReq`, or a worker-side engine
request whose recipe rode in via ``extra_args``). The recipe is the lightweight
payload the driver ships (per-sample id strings + seed; NO noise tensor on the
wire); each engine reconstructs a ``NoiseRecipe`` from its own request type and
calls :meth:`resolve`.

Resolution precedence (one place, all engines):
  1. ``initial_latents`` present  → use it verbatim. This carries genuine latent
     DATA (img2img / i2v first-frame), which cannot be regenerated from a seed
     and so must be shipped as a tensor.
  2. ``noise_group_ids`` + ``latent_shape`` present → regenerate the
     byte-identical x_T via :func:`unirl.sde.noise.regen_initial_noise`
     (CPU-fp32 canonical). Same recipe → same x_T on any engine.
  3. otherwise → ``None`` (engine draws its own; e.g. ``DISABLE_DRIVER_XT``).

There is no separate "seed-only / shape-unknown" path: a model whose latent
shape is only known mid-rollout (e.g. HI3's DiT grid depends on the AR stage)
simply constructs its ``NoiseRecipe`` LATER — at the engine point where the
shape resolves — filling ``latent_shape`` then. So "shape known at request time"
vs "shape resolved in-worker" is just *when the recipe is built*, not two
different resolution code paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Optional, Tuple

import torch


@dataclass
class NoiseRecipe:
    """Normalized x_T recipe consumed by every engine (see module docstring)."""

    noise_group_ids: List[str] = field(default_factory=list)
    base_seed: int = 0
    # None until the latent shape is known. Models with a request-time-known
    # shape fill it via ``from_rollout_req``; dynamic-shape models fill it at the
    # engine point where the shape resolves.
    latent_shape: Optional[Tuple[int, ...]] = None
    # Path 1: genuine latent DATA (img2img / i2v first-frame), shipped verbatim.
    initial_latents: Optional[torch.Tensor] = None

    def for_batch(self, batch_size: int, *, latent_shape: Optional[Tuple[int, ...]] = None) -> "NoiseRecipe":
        """Specialize this (per-sample) recipe to a concrete ``batch_size``-row
        engine call, returning a NEW recipe.

        - Aligns the per-sample ``noise_group_ids`` to the batch: slice when we
          have enough, else cycle. (No-op when they already match.)
        - Optionally fills ``latent_shape`` — for engines whose shape is only
          known mid-rollout (e.g. HI3's DiT grid, resolved post-AR): build the
          recipe with ``latent_shape=None`` and pass the resolved shape here.

        Pure transform (``dataclasses.replace``); call ``.resolve()`` on the
        result to get the tensor.
        """
        gids = self.noise_group_ids
        if gids and len(gids) != batch_size:
            gids = gids[:batch_size] if len(gids) >= batch_size else [gids[i % len(gids)] for i in range(batch_size)]
        return replace(
            self,
            noise_group_ids=gids,
            latent_shape=latent_shape if latent_shape is not None else self.latent_shape,
        )

    def resolve(
        self,
        *,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> Optional[torch.Tensor]:
        """Produce x_T, or ``None`` to defer to the engine's own RNG.

        Pure resolution — Path 1 (``initial_latents`` tensor) / Path 2
        (``noise_group_ids`` + ``latent_shape`` → CPU-fp32 regen) / None. Batch
        alignment + late shape are a separate concern; see :meth:`for_batch`.
        """
        if self.initial_latents is not None:
            return self.initial_latents
        if not (self.noise_group_ids and self.latent_shape):
            return None
        # Local import avoids a module-level types→sde cycle.
        from unirl.sde.noise import regen_initial_noise

        return regen_initial_noise(
            noise_group_ids=[str(g) for g in self.noise_group_ids],
            base_seed=int(self.base_seed),
            latent_shape=tuple(self.latent_shape),
            device=device,
            dtype=dtype,
        )

    @classmethod
    def from_rollout_req(cls, req) -> "NoiseRecipe":
        """Build a recipe from a driver-side ``RolloutReq`` (request-time).

        Used by trainside/sglang model pipelines, where the latent shape is
        already known (fixed-shape models populate ``req.init_noise_latent_shape``).
        Duck-typed on the req's attributes so it doesn't import RolloutReq.
        """
        from unirl.types.sampling import get_diffusion_params

        cond = (getattr(req, "request_conditions", None) or {}).get("initial_latents")
        diffusion = get_diffusion_params(getattr(req, "sampling_params", None))
        seed = int(diffusion.seed) if diffusion is not None and getattr(diffusion, "seed", None) is not None else 0
        shape = getattr(req, "init_noise_latent_shape", None)
        return cls(
            noise_group_ids=list(getattr(req, "init_noise_group_ids", None) or []),
            base_seed=seed,
            latent_shape=tuple(shape) if shape else None,
            initial_latents=getattr(cond, "latents", None) if cond is not None else None,
        )


__all__ = ["NoiseRecipe"]
