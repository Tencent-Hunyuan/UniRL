"""Noise primitives for the SDE / flow-match sampling loop.

Sibling of :mod:`unirl.sde.runtime` (σ schedule + dynamic-shift μ):
this module owns the *noise* side of the sampling loop — per-sample x_T
generation, per-group noise sharing, and the deterministic per-group seed
derivation (``_derive_group_seed``) that keys each sample's x_T.

The driver (``DiffusionTrainer._build_req``) ships only a deterministic x_T
RECIPE — per-sample ``init_noise_group_ids`` + ``init_noise_latent_shape`` on
the ``RolloutReq`` — and every engine regenerates the byte-identical x_T from
it via :func:`regen_initial_noise` (a CPU-fp32 wrapper over
:func:`generate_shared_noise`). The plain ``generate_latents`` fallback runs
only when neither a recipe nor ``request_conditions['initial_latents']`` is
present, i.e. the engine draws its own noise.
"""

import hashlib
from typing import Dict, List, Optional, Tuple

import torch

# Inclusive max for torch.Generator.manual_seed and torch initial_seed conventions.
MAX_TORCH_SEED = (1 << 63) - 1


def _derive_group_seed(base_seed: int, group_id: str) -> int:
    payload = f"{int(base_seed)}::{str(group_id)}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % (MAX_TORCH_SEED + 1)


def generate_shared_noise(
    batch_size: int,
    latent_shape: Tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    noise_group_ids: Optional[List[str]] = None,
    base_seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Generate initial noise where samples sharing the same noise_group_id
    receive identical noise.

    When ``base_seed`` is provided, each unique ``noise_group_id`` gets a
    deterministic seed via ``_derive_group_seed(base_seed, group_id)``.
    This is shard-safe: as long as ``base_seed`` (scalar) and per-sample
    ``noise_group_ids`` (sliced list) are preserved across GPU shards, the
    same group always produces the same noise.

    Used by:
    - ``init_same_noise=True``: noise_group_ids are per-group (shared within group)
    - ``init_same_noise=False``: noise_group_ids are per-sample (unique noise)

    Args:
        batch_size: Total number of samples in batch
        latent_shape: Shape of a single latent (C, H, W) or (C, T, H, W) for video
        device: Device for the tensor
        dtype: Data type for the tensor
        noise_group_ids: Explicit per-sample noise sharing groups aligned to the batch
        base_seed: Base seed for deterministic per-group noise derivation

    Returns:
        Noise tensor [batch_size, *latent_shape] with shared noise per explicit group
    """
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
    """
    High-level function for generating initial latents.

    When ``base_seed`` and ``noise_group_ids`` are both provided, noise is
    deterministically derived per unique ``noise_group_id`` via
    ``_derive_group_seed``.  The sharing-vs-uniqueness behaviour is
    controlled by the caller through the content of ``noise_group_ids``:

    - ``init_same_noise=True``: IDs are per-group → shared noise within group
    - ``init_same_noise=False``: IDs are per-sample → unique noise per sample

    When ``base_seed`` or ``noise_group_ids`` is absent, falls back to
    plain random noise.

    Args:
        batch_size: Total number of samples
        latent_shape: Shape of a single latent (C, H, W) or (C, T, H, W)
        device: Device for the tensor
        dtype: Data type for the tensor
        init_same_noise: Whether to share noise across samples for same prompt
        samples_per_prompt: Rollout geometry hint kept for sampler API compatibility
        noise_group_ids: Per-sample noise group identifiers
        base_seed: Base seed for deterministic noise derivation

    Returns:
        Latent tensor [batch_size, *latent_shape]
    """
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
    """Engine-side x_T regeneration from a driver-shipped RECIPE (gids + seed).

    Counterpart to the driver's :func:`generate_shared_noise` call: given the
    same ``(noise_group_ids, base_seed, latent_shape)`` the driver authored,
    every engine reproduces a BYTE-IDENTICAL x_T — so the driver is the single
    source of initial noise and all engines start each rollout from the same
    x_T (cross-engine-aligned and reproducible).

    Determinism rests on a PINNED generation environment: noise is always drawn
    on **CPU in fp32** with an explicit seeded ``torch.Generator`` (see
    :func:`generate_shared_noise`), then moved/cast to the engine's device/dtype
    as the LAST step. CPU randn is bit-stable across machines for a fixed torch
    version (cuda randn is NOT — it varies by GPU arch), so CPU-gen is what makes
    trainside / vllm / sglang agree to the byte. Verified across nodes+clusters
    on torch 2.11.0 (sha256 match). The cast to a lower-precision ``dtype`` is
    itself deterministic, so the result is reproducible end-to-end.
    """
    xt_cpu_fp32 = generate_shared_noise(
        batch_size=len(noise_group_ids),
        latent_shape=tuple(latent_shape),
        device=torch.device("cpu"),
        dtype=torch.float32,
        noise_group_ids=list(noise_group_ids),
        base_seed=int(base_seed),
    )
    return xt_cpu_fp32.to(device=device, dtype=dtype)
