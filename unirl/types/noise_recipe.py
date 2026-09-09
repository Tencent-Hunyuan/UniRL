"""``NoiseRecipe`` — the normalized, engine-agnostic x_T recipe."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Optional, Tuple

import torch


@dataclass
class NoiseRecipe:
    """Normalized x_T recipe consumed by every engine (see module docstring)."""

    noise_group_ids: List[str] = field(default_factory=list)
    base_seed: int = 0
    latent_shape: Optional[Tuple[int, ...]] = None
    initial_latents: Optional[torch.Tensor] = None

    def for_batch(self, batch_size: int, *, latent_shape: Optional[Tuple[int, ...]] = None) -> "NoiseRecipe":
        """Specialize this (per-sample) recipe to a concrete ``batch_size``-row engine call, returning a NEW recipe."""
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
        salt: str = "",
        latent_shape: Optional[Tuple[int, ...]] = None,
    ) -> Optional[torch.Tensor]:
        """Produce x_T, or ``None`` to defer to the engine's own RNG."""
        if not salt and self.initial_latents is not None:
            return self.initial_latents
        gids = [f"{g}::{salt}" for g in self.noise_group_ids] if salt else self.noise_group_ids
        shape = latent_shape if latent_shape is not None else self.latent_shape
        if not (gids and shape):
            return None
        from unirl.sde.noise import regen_initial_noise

        return regen_initial_noise(
            noise_group_ids=[str(g) for g in gids],
            base_seed=int(self.base_seed),
            latent_shape=tuple(shape),
            device=device,
            dtype=dtype,
        )

    @classmethod
    def from_sample(cls, sample) -> "NoiseRecipe":
        """Build a recipe from a request ``Sample`` (its gen frontier part)."""
        gen = sample.parts[-1]
        diffusion = gen.sampling_params
        if bool(getattr(diffusion, "disable_driver_xt", False)):
            return cls()
        seg = gen.segment
        explicit_keys = list(getattr(gen, "init_noise_group_ids", None) or [])
        share = bool(getattr(diffusion, "init_same_noise", False)) if diffusion is not None else False
        keys = explicit_keys or (gen.group_ids if share else list(gen.sample_ids))
        shape = getattr(diffusion, "init_noise_latent_shape", None) if diffusion is not None else None
        seed = int(diffusion.seed) if diffusion is not None and getattr(diffusion, "seed", None) is not None else 0
        return cls(
            noise_group_ids=[str(k) for k in keys],
            base_seed=seed,
            latent_shape=tuple(shape) if shape else None,
            initial_latents=getattr(seg, "initial_latents", None) if seg is not None else None,
        )


__all__ = ["NoiseRecipe"]
