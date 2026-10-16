"""Request-local RNG helpers for reproducible HI3 trainside sampling."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import List, Optional, Sequence, Tuple

import torch

from unirl.types.noise_recipe import NoiseRecipe
from unirl.types.sample import Part, Sample
from unirl.types.sampling import DiffusionSamplingParams

_MAX_TORCH_SEED = (1 << 63) - 1


def derive_seed(base_seed: int, key: str) -> int:
    """Derive a stable Torch seed from a base seed and semantic key."""
    payload = f"{base_seed}::{key}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % (_MAX_TORCH_SEED + 1)


def make_cpu_generators(base_seed: int, keys: Sequence[str]) -> List[torch.Generator]:
    """Build independent CPU generators for the supplied semantic keys."""
    generators: List[torch.Generator] = []
    for key in keys:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(derive_seed(base_seed, key))
        generators.append(generator)
    return generators


def make_sde_step_generators(
    base_seed: int,
    sample_keys: Sequence[str],
    step_index: int,
) -> List[torch.Generator]:
    """Build one reproducible, sample-unique generator for an SDE step."""
    return make_cpu_generators(
        base_seed,
        [f"sde-step:{step_index}:{sample_key}" for sample_key in sample_keys],
    )


def prepare_seeded_sampling(
    sample: Sample,
    frontier: Part,
    params: DiffusionSamplingParams,
) -> Tuple[DiffusionSamplingParams, Optional[List[str]]]:
    """Prepare request-local RNG streams without touching global RNG state."""
    recipe_ids = NoiseRecipe.from_sample(sample).noise_group_ids
    if params.seed is None or not recipe_ids:
        return params, None
    seeded_params = replace(params, noise_group_ids=recipe_ids)
    sde_sample_keys = [
        f"{recipe_id}:sample:{sample_id}" for recipe_id, sample_id in zip(recipe_ids, frontier.sample_ids)
    ]
    return seeded_params, sde_sample_keys
