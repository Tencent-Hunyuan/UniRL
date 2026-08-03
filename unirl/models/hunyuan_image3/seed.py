"""Request-local RNG helpers for reproducible HI3 trainside sampling."""

from __future__ import annotations

import hashlib
from typing import List, Sequence

import torch

_MAX_TORCH_SEED = (1 << 63) - 1


def derive_seed(base_seed: int, key: str) -> int:
    """Derive a stable Torch seed from a base seed and semantic key."""
    payload = f"{int(base_seed)}::{str(key)}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % (_MAX_TORCH_SEED + 1)


def make_cpu_generators(base_seed: int, keys: Sequence[str]) -> List[torch.Generator]:
    """Build independent CPU generators for the supplied semantic keys."""
    generators: List[torch.Generator] = []
    for key in keys:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(derive_seed(base_seed, str(key)))
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
        [f"sde-step:{int(step_index)}:{sample_key}" for sample_key in sample_keys],
    )
