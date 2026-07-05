"""Shared DRaFT-K grad sampling for ReFL (direct differentiable-reward backprop).

Family-agnostic: drives the deterministic (eta=0), grad-windowed sampling loop
through the ``DiffusionStage`` protocol's ``predict_noise_at_step`` + the shared
``strategy.denoise`` (sde/kernels.py). Every diffusion family implements
``predict_noise_at_step`` (all packing/CFG/routing lives inside it), so this one
loop reproduces each family's ``diffuse`` at eta=0 — no per-family
``diffuse_draft_k`` needed.

- ``draft_k_sample`` — the loop; returns a 1-step ``LatentSegment`` the existing
  ``decode(grad=True)`` consumes unchanged.
- ``draft_generate`` — the family-agnostic ReFL forward (conditions → sample →
  grad-decode) that ``ReFLPolicy`` calls; selects the family purely via the
  ``Pipeline`` it's handed.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Optional

import torch

from unirl.sde.noise import generate_latents
from unirl.sde.runtime import get_sigma_schedule
from unirl.types.primitives import Images, Texts
from unirl.types.sampling import DiffusionSamplingParams
from unirl.types.segments import LatentSegment


def draft_k_sample(
    stage: Any,
    conditions: Any,
    *,
    schedule: torch.Tensor,
    params: DiffusionSamplingParams,
    draft_num_steps: int,
    initial_latents: torch.Tensor,
) -> LatentSegment:
    """Deterministic (eta=0) sampling with gradients through only the final
    ``draft_num_steps`` steps (DRaFT-K). Returns a 1-step ``LatentSegment`` whose
    clean latent ``x_0`` carries grad_fn into the stage's transformer params.

    ``draft_num_steps <= 0`` keeps grad through ALL steps; ``K>0`` runs the first
    ``T-K`` steps under ``torch.no_grad()`` (detached) and only the last ``K`` carry
    grad. Family-agnostic: uses ``stage.predict_noise_at_step`` (protocol) +
    ``stage.strategy.denoise`` (shared kernel). Caller owns ``.train()`` mode + the
    outer grad scope (the distributed ``enable_grad()`` context) and supplies
    ``initial_latents`` (noise shapes are per-family).
    """
    device = initial_latents.device
    T = int(params.num_inference_steps)
    if int(schedule.shape[0]) != T + 1:
        raise ValueError(f"draft_k_sample: schedule length {schedule.shape[0]} != T+1={T + 1}")
    schedule = schedule.to(device)
    stage.strategy.init_schedule(schedule)

    # cache_enabled=False: PEFT fp32 LoRA adapters lose parameter grads under
    # autocast's weight cache on the differentiable pass (see train_reward_dpgo.py).
    autocast_dtype = getattr(stage, "autocast_dtype", None)
    autocast_ctx = (
        torch.autocast("cuda", autocast_dtype, cache_enabled=False)
        if device.type == "cuda" and autocast_dtype in (torch.float16, torch.bfloat16)
        else nullcontext()
    )
    sigma_max = schedule[1].float() if int(schedule.shape[0]) > 1 else torch.tensor(0.99)

    # K<=0 → grad through all steps; K>0 → grad only through the final K.
    grad_start_index = max(0, T - draft_num_steps) if draft_num_steps > 0 else 0
    latents = initial_latents

    for i in range(T):
        sigma = schedule[i].to(device)
        sigma_next = schedule[i + 1].to(device)
        if i < grad_start_index:
            # Frozen prefix: deterministic, no autograd graph retained.
            with torch.no_grad(), autocast_ctx:
                noise_pred = stage.predict_noise_at_step(conditions, sample=latents, sigma=sigma, params=params)
                new_latents = stage.strategy.denoise(
                    noise_pred, latents, sigma, sigma_next, eta=0.0, sigma_max=sigma_max, step_index=i
                )[0]
            latents = new_latents.detach()
        else:
            # Grad window: backprop flows through these transformer forwards.
            if draft_num_steps > 0 and i == grad_start_index:
                latents = latents.detach().requires_grad_(True)
            with autocast_ctx:
                noise_pred = stage.predict_noise_at_step(conditions, sample=latents, sigma=sigma, params=params)
                new_latents = stage.strategy.denoise(
                    noise_pred, latents, sigma, sigma_next, eta=0.0, sigma_max=sigma_max, step_index=i
                )[0]
            latents = new_latents

    indices = torch.tensor([T], dtype=torch.long, device=device)
    return LatentSegment(latents=latents.unsqueeze(1), sigmas=schedule, indices=indices)


def draft_generate(
    pipeline: Any,
    *,
    model_config: Any,
    texts: Texts,
    params: DiffusionSamplingParams,
    draft_num_steps: int,
    negatives: Optional[Texts] = None,
    activation_checkpoint: bool = False,
) -> Images:
    """Family-agnostic ReFL forward: conditions → DRaFT-K sample → grad VAE decode.

    Selects the family purely via ``pipeline`` (its ``build_conditions`` /
    ``diffusion`` / ``vae_decode`` stages + ``latent_shape`` classmethod). The only
    output tensor is the image, carrying grad_fn into the policy weights.
    """
    diffusion = pipeline.diffusion
    device = pipeline.bundle.device

    # Text encoders are frozen — keep their (large, e.g. T5-XXL) forward graph out
    # of the DRaFT backward; the transformer still gets grad via the latents.
    with torch.no_grad():
        conditions = pipeline.build_conditions(texts, negatives=negatives, guidance_scale=float(params.guidance_scale))

    shift = float(getattr(model_config, "shift", 3.0))
    schedule = get_sigma_schedule(int(params.num_inference_steps), shift=shift, device=device)

    per_sample_shape = type(pipeline).latent_shape(model_config=model_config, sampling_spec=params)
    latents = generate_latents(
        batch_size=len(texts.texts),
        latent_shape=tuple(per_sample_shape),
        device=device,
        dtype=getattr(diffusion, "trajectory_dtype", torch.bfloat16),
        init_same_noise=bool(params.init_same_noise),
        samples_per_prompt=int(params.samples_per_prompt),
        noise_group_ids=params.noise_group_ids,
        base_seed=int(params.seed),
    )

    seg = draft_k_sample(
        diffusion,
        conditions,
        schedule=schedule,
        params=params,
        draft_num_steps=draft_num_steps,
        initial_latents=latents,
    )
    return pipeline.vae_decode.decode(seg, grad=True, activation_checkpoint=activation_checkpoint)


__all__ = ["draft_k_sample", "draft_generate"]
