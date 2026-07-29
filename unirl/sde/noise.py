"""Noise primitives for the SDE / flow-match sampling loop.

Sibling of :mod:`unirl.sde.runtime` (σ schedule + dynamic-shift μ):
this module owns the *noise* side of the sampling loop — per-sample x_T
generation, per-group noise sharing, and the deterministic per-group seed
derivation (``_derive_group_seed``) that keys each sample's x_T.

The driver ships only a deterministic x_T recipe on the request ``Sample``'s
diffusion generation Part: lineage-derived sample/group ids plus
``DiffusionSamplingParams.init_noise_latent_shape``. Every engine regenerates
the byte-identical x_T from it via :func:`regen_initial_noise` (a CPU-fp32
wrapper over :func:`generate_shared_noise`). The plain ``generate_latents``
fallback runs only when :class:`unirl.types.noise_recipe.NoiseRecipe` resolves
neither a recipe nor a Part-carried ``initial_latents`` tensor, so the engine
must draw its own noise.
"""

import hashlib
import json
from typing import Dict, List, Optional, Tuple

import torch

# Inclusive max for torch.Generator.manual_seed and torch initial_seed conventions.
MAX_TORCH_SEED = (1 << 63) - 1


# Group-id prefix selecting prompt-content seeding: x_T is keyed only on the prompt text
# (rank/step-independent), so the same prompt yields the same image across steps/checkpoints.
# Used by eval for reproducible, comparable generations.
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
    """Deterministic per-group seed for x_T generation.

    * prompt-seed group ids -> ``(base_seed + int(SHA256(text)[:4]) + sample_ordinal)
      % 2**31``: keyed on prompt content and sibling ordinal (rank/step-independent).
      Used for reproducible eval.
    * otherwise -> blake2b of ``base_seed::group_id`` (per-rollout/per-sample varying) for training.
    """
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
    # Match the SGLang rollout contract exactly.
    return int.from_bytes(digest, byteorder="big", signed=False) % MAX_TORCH_SEED


def make_denoise_step_generators(
    *,
    base_seed: int,
    step_index: int,
    sample_ids: List[str],
) -> List[torch.Generator]:
    """Build deterministic CPU generators for one SDE transition.

    CPU generation keeps the random values independent of GPU architecture and
    byte-identical between trainside and SGLang for the same seed tuple.
    """
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
