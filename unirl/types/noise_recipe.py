"""``NoiseRecipe`` — normalized, engine-agnostic diffusion noise identity."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Optional, Tuple

import torch


@dataclass
class NoiseRecipe:
    """Driver-authored identities for initial and per-step diffusion noise."""

    noise_group_ids: List[str] = field(default_factory=list)
    denoise_seed_keys: List[str] = field(default_factory=list)
    base_seed: int = 0
    latent_shape: Optional[Tuple[int, ...]] = None
    initial_latents: Optional[torch.Tensor] = None

    def for_batch(self, batch_size: int, *, latent_shape: Optional[Tuple[int, ...]] = None) -> "NoiseRecipe":
        """Specialize this (per-sample) recipe to a concrete ``batch_size``-row engine call, returning a NEW recipe."""

        def align(keys: List[str]) -> List[str]:
            if not keys or len(keys) == batch_size:
                return list(keys)
            return (
                list(keys[:batch_size]) if len(keys) >= batch_size else [keys[i % len(keys)] for i in range(batch_size)]
            )

        return replace(
            self,
            noise_group_ids=align(self.noise_group_ids),
            denoise_seed_keys=align(self.denoise_seed_keys),
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
        seg = gen.segment
        disable_xt = bool(getattr(diffusion, "disable_driver_xt", False))
        explicit_xt_keys = list(getattr(gen, "init_noise_group_ids", None) or [])
        share = bool(getattr(diffusion, "init_same_noise", False)) if diffusion is not None else False
        xt_keys = [] if disable_xt else explicit_xt_keys or (gen.group_ids if share else list(gen.sample_ids))
        denoise_keys = list(getattr(gen, "denoise_seed_keys", None) or gen.sample_ids)
        shape = getattr(diffusion, "init_noise_latent_shape", None) if diffusion is not None else None
        seed = int(diffusion.seed) if diffusion is not None and getattr(diffusion, "seed", None) is not None else 0
        return cls(
            noise_group_ids=[str(k) for k in xt_keys],
            denoise_seed_keys=[str(k) for k in denoise_keys],
            base_seed=seed,
            latent_shape=tuple(shape) if shape else None,
            initial_latents=None if disable_xt or seg is None else getattr(seg, "initial_latents", None),
        )


__all__ = ["NoiseRecipe"]
