"""Noise primitives for the SDE / flow-match sampling loop."""

import hashlib
import json
from typing import Dict, List, Optional, Tuple

import torch

MAX_TORCH_SEED = (1 << 63) - 1


PROMPT_SEED_PREFIX = "prompt:"


def make_prompt_seed_group_id(prompt: str, sample_ordinal: int = 0) -> str:
    """Encode prompt content and a sibling-sample ordinal into an eval noise group id."""
    ordinal = int(sample_ordinal)
    if ordinal < 0:
        raise ValueError(f"sample_ordinal must be non-negative, got {ordinal}")
    payload = json.dumps(
        {"prompt": str(prompt), "sample": ordinal},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{PROMPT_SEED_PREFIX}{payload}"


def _derive_group_seed(base_seed: int, group_id: str) -> int:
    """Deterministic per-group seed for x_T generation."""
    gid = str(group_id)
    if gid.startswith(PROMPT_SEED_PREFIX):
        payload = gid[len(PROMPT_SEED_PREFIX) :]
        sample_ordinal = 0
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError:
            prompt = payload
        else:
            if not isinstance(decoded, dict) or not isinstance(decoded.get("prompt"), str):
                raise ValueError(f"invalid prompt-seed group id: {gid!r}")
            prompt = decoded["prompt"]
            sample_ordinal = int(decoded.get("sample", 0))
            if sample_ordinal < 0:
                raise ValueError(f"invalid prompt-seed sample ordinal: {sample_ordinal}")
        digest = hashlib.sha256(prompt.encode("utf-8")).digest()
        return (int(base_seed) + int.from_bytes(digest[:4], "big") + sample_ordinal) % (2**31)
    payload = f"{int(base_seed)}::{gid}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % (MAX_TORCH_SEED + 1)


def derive_denoise_step_seed(base_seed: int, step_index: int, sample_id: str) -> int:
    """Derive the cross-engine per-sample, per-step SDE-noise seed."""
    payload = (f"{int(base_seed)}::step::{int(step_index)}::sample::{str(sample_id)}").encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % MAX_TORCH_SEED


def make_denoise_step_generators(
    *,
    base_seed: int,
    step_index: int,
    sample_ids: List[str],
) -> List[torch.Generator]:
    """Build deterministic CPU generators for one SDE transition."""
    generators: List[torch.Generator] = []
    for sample_id in sample_ids:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            derive_denoise_step_seed(
                base_seed=int(base_seed),
                step_index=int(step_index),
                sample_id=str(sample_id),
            )
        )
        generators.append(generator)
    return generators


def generate_shared_noise(
    batch_size: int,
    latent_shape: Tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    noise_group_ids: Optional[List[str]] = None,
    base_seed: Optional[int] = None,
) -> torch.Tensor:
    """Generate initial noise where samples sharing the same noise_group_id receive identical noise."""
    if not isinstance(noise_group_ids, list) or len(noise_group_ids) != batch_size:
        raise ValueError(
            "generate_shared_noise requires explicit noise_group_ids aligned to batch_size. "
            f"Got batch_size={batch_size}, noise_group_ids_len="
            f"{len(noise_group_ids) if isinstance(noise_group_ids, list) else None}."
        )

    group_noise: Dict[str, torch.Tensor] = {}
    chunks: List[torch.Tensor] = []
    for raw_group_id in noise_group_ids:
        group_id = str(raw_group_id)
        noise = group_noise.get(group_id)
        if noise is None:
            if base_seed is None:
                noise = torch.randn(
                    *latent_shape,
                    device=device,
                    dtype=dtype,
                )
            else:
                group_generator = torch.Generator(device=device)
                group_generator.manual_seed(_derive_group_seed(base_seed, group_id))
                noise = torch.randn(
                    *latent_shape,
                    device=device,
                    dtype=dtype,
                    generator=group_generator,
                )
            group_noise[group_id] = noise
        chunks.append(noise)
    return torch.stack(chunks, dim=0)


def generate_latents(
    batch_size: int,
    latent_shape: Tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    init_same_noise: bool = False,
    samples_per_prompt: int = 1,
    noise_group_ids: Optional[List[str]] = None,
    base_seed: Optional[int] = None,
) -> torch.Tensor:
    """High-level function for generating initial latents."""
    if init_same_noise:
        assert base_seed is not None and noise_group_ids is not None, (
            "generate_latents requires both base_seed and noise_group_ids when init_same_noise=True."
        )
        return generate_shared_noise(
            batch_size=batch_size,
            latent_shape=latent_shape,
            device=device,
            dtype=dtype,
            noise_group_ids=noise_group_ids,
            base_seed=base_seed,
        )
    return torch.randn(
        batch_size,
        *latent_shape,
        device=device,
        dtype=dtype,
    )


def regen_initial_noise(
    noise_group_ids: List[str],
    base_seed: int,
    latent_shape: Tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Engine-side x_T regeneration from a driver-shipped RECIPE (gids + seed)."""
    xt_cpu_fp32 = generate_shared_noise(
        batch_size=len(noise_group_ids),
        latent_shape=tuple(latent_shape),
        device=torch.device("cpu"),
        dtype=torch.float32,
        noise_group_ids=list(noise_group_ids),
        base_seed=int(base_seed),
    )
    return xt_cpu_fp32.to(device=device, dtype=dtype)
