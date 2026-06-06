"""Assemble ``RolloutResp`` track pieces (segment / decoded / conditions) from raw results.

Pure: operates on already-fetched wire data (SGLang ``GenerationResult`` objects)
and ``unirl.types`` — no SGLang import, no engine state. The model-specific
variation (e.g. Klein's packed-trajectory unpack, image vs video decoded) is an
overridable adapter method; these are the generic mechanics those methods call.

Ported from the old engine's ``response.py`` helpers, minus the model-family
branches (those move to adapter overrides).
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional, Sequence, Tuple

import torch

from unirl.config.require import require
from unirl.rollout.engine.sglang_diffusion.backends import RawResult
from unirl.rollout.engine.sigma_verify import verify_engine_used_sigmas
from unirl.types.conditions.text import TextEmbedCondition
from unirl.types.primitives import Images
from unirl.types.segments.latent import LatentSegment, make_image_segment
from unirl.types.trajectory_store import compute_trajectory_positions
from unirl.rollout.engine.sglang_diffusion.utils.tensors import (
    decode_sample,
    fuse_encoder_outputs,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Segment build
# ---------------------------------------------------------------------------


def collect_trajectory_latents(results: Sequence[RawResult]) -> torch.Tensor:
    """Concat per-result trajectory latents on the batch dim (detached, CPU)."""
    latents = []
    for r in results:
        require(r.trajectory_latents is not None, "SGLang result missing trajectory_latents")
        latents.append(r.trajectory_latents.detach().cpu())
    return torch.cat(latents, dim=0)


def derive_timestep_alignment(
    *,
    trajectories_tensor: torch.Tensor,
    expected_sigmas: torch.Tensor,
    results: Sequence[RawResult],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Validate the T+1 trajectory shape and verify SGLang used the σ we sent.

    ``expected_sigmas`` is the schedule the engine pinned on ``RolloutReq.sigmas``
    and forwarded to SGLang; SGLang echoes it back per result via
    ``trajectory_timesteps``. :func:`verify_engine_used_sigmas` asserts elementwise
    equality (fatal on drift) so rollout and trainer-side replay use numerically
    identical σ schedules.
    """
    traj_len = int(trajectories_tensor.shape[1])
    expected_len = int(expected_sigmas.shape[0])
    require(
        traj_len == expected_len,
        f"SGLang trajectory length {traj_len} != expected_sigmas length {expected_len}. "
        f"Modern SGLang prepends initial latents at "
        f"sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py so "
        f"trajectory carries T+1 latents; expected_sigmas (from req.sigmas) is T+1 "
        f"too. Upgrade SGLang or fix the sampler to emit a T+1 trajectory.",
    )
    expected_cpu = expected_sigmas.detach().to(torch.float32).cpu()
    step_indices = torch.arange(expected_len, dtype=torch.long)
    for i, result in enumerate(results):
        verify_engine_used_sigmas(
            result.trajectory_timesteps,
            expected=expected_cpu,
            engine_name=f"sglang (result {i})",
        )
    return expected_cpu, step_indices


def build_latent_segment(
    trajectories_tensor: torch.Tensor,
    *,
    results: Sequence[RawResult],
    expected_sigmas: torch.Tensor,
    num_steps: int,
    sde_indices: Optional[List[int]],
    emit_native_logprob: bool,
    segment_factory: Callable[..., LatentSegment] = make_image_segment,
) -> LatentSegment:
    """Pack an (already-unpacked) trajectory tensor into one batched ``LatentSegment``.

    ``segment_factory`` selects the modality (default image); a video adapter
    passes ``make_video_segment``. The caller owns the model-specific unpack of
    ``trajectories_tensor`` (e.g. Klein); this function is shape-agnostic past
    the T+1 invariant.
    """
    sigmas, step_indices = derive_timestep_alignment(
        trajectories_tensor=trajectories_tensor,
        expected_sigmas=expected_sigmas,
        results=results,
    )

    # Selective trim: when only a subset of trajectory positions is referenced by
    # the SDE step set, drop unused columns to save Ray IPC bandwidth.
    # ``compute_trajectory_positions`` returns only the (i, i+1) pairs for
    # SDE-gated steps; we always preserve the terminal position T so the clean
    # image latent (``seg.latents[:, -1]``) stays available for VAE decode.
    traj_len = int(trajectories_tensor.shape[1])
    indices_t: torch.Tensor = step_indices
    if sde_indices is not None and len(sde_indices) < num_steps:
        needed = set(compute_trajectory_positions(set(sde_indices), num_steps))
        needed.add(int(num_steps))
        keep_cols = sorted(p for p in needed if 0 <= p < traj_len)
        if keep_cols and len(keep_cols) < traj_len:
            trajectories_tensor = trajectories_tensor[:, keep_cols]
            indices_t = torch.tensor(keep_cols, dtype=torch.long)

    # sde_indices: always populated (trainer needs to know which steps to replay).
    # sde_logp: best-effort native emission; whether it is used or recomputed is
    # the training layer's call (``algorithm.old_logp_source``), not an engine flag.
    sde_indices_t: Optional[torch.Tensor] = (
        torch.tensor(list(sde_indices), dtype=torch.long)
        if sde_indices is not None
        else torch.arange(num_steps, dtype=torch.long)
    )
    sde_logp: Optional[torch.Tensor] = None
    if emit_native_logprob:
        sde_logp = _native_sde_logp(
            results, num_steps=num_steps, sde_indices=sde_indices
        )

    batch_size = int(trajectories_tensor.shape[0])
    return segment_factory(
        latents=trajectories_tensor,
        sigmas=sigmas,
        indices=indices_t,
        sde_logp=sde_logp,
        sde_indices=sde_indices_t,
        sample_indices=torch.arange(batch_size, dtype=torch.long),
    )


def _native_sde_logp(
    results: Sequence[RawResult],
    *,
    num_steps: int,
    sde_indices: Optional[List[int]],
) -> Optional[torch.Tensor]:
    """Best-effort extract of SGLang's native ``trajectory_log_probs`` into ``[B, S]``.

    Returns ``None`` when any result lacks per-step log-probs and lets the
    trainer decide: replay (``algorithm.old_logp_source='replay'``) recomputes;
    native raises trainer-side with an actionable message. The engine stays
    silent — it can't know the intent, and for an intentional replay run a
    missing emission is expected, not warning-worthy. Shape drift, by contrast,
    is a hard error.
    """
    per_result: List[Optional[torch.Tensor]] = [
        result.trajectory_log_probs.detach().cpu()
        if result.trajectory_log_probs is not None
        else None
        for result in results
    ]
    if any(lp is None for lp in per_result):
        return None
    log_prob_tensor = torch.cat([lp for lp in per_result if lp is not None], dim=0)
    # [B, T] (one entry per SDE transition). When sde_indices is a subset but the
    # server emitted the full schedule, slice down to the requested transitions.
    s_dim = int(log_prob_tensor.shape[1])
    expected_s = len(sde_indices) if sde_indices is not None else num_steps
    if s_dim == num_steps and sde_indices is not None and expected_s < num_steps:
        keep_idx = torch.tensor(sorted(int(i) for i in sde_indices), dtype=torch.long)
        log_prob_tensor = log_prob_tensor.index_select(1, keep_idx)
        s_dim = int(log_prob_tensor.shape[1])
    require(
        s_dim == expected_s,
        f"SGLang trajectory_log_probs shape {tuple(log_prob_tensor.shape)} second "
        f"dim={s_dim} does not match expected SDE-step count {expected_s}. "
        f"sigma_schedule / num_inference_steps / sde_indices drift — fix the "
        f"source rather than fall back to replay silently.",
    )
    return log_prob_tensor


# ---------------------------------------------------------------------------
# Decoded media
# ---------------------------------------------------------------------------


def stack_decoded_images(results: Sequence[RawResult]) -> Optional[Images]:
    """Stack per-result decoded ``samples`` into ``Images.pixels [B, C, H, W]``.

    4-D (video) samples are dropped with a warning — there is no video reward
    consumer yet, so packing them as ``Videos`` is deferred (matches the old
    engine's behavior for every family).
    """
    per_sample_tensors: List[torch.Tensor] = []
    skipped_video = False
    for result in results:
        canonical = decode_sample(result.samples)
        if canonical is None:
            continue
        if canonical.dim() == 3:
            per_sample_tensors.append(canonical.to(torch.float32))
        elif canonical.dim() == 4:
            skipped_video = True
        else:
            raise RuntimeError(
                f"stack_decoded_images: unexpected canonical media rank "
                f"{canonical.dim()}; want 3 (image) or 4 (video)."
            )
    if skipped_video:
        logger.warning(
            "SGLang result contained 4D (video) samples — Videos primitive packing "
            "is not yet implemented in the response translator; dropping. "
            "Add a Videos branch when a video reward consumer lands."
        )
    if not per_sample_tensors:
        return None
    return Images(pixels=torch.stack(per_sample_tensors, dim=0))


# ---------------------------------------------------------------------------
# Conditions packing
# ---------------------------------------------------------------------------


def fuse_text_conditions(
    results: Sequence[RawResult],
) -> Tuple[Optional[TextEmbedCondition], Optional[TextEmbedCondition]]:
    """Fuse per-result encoder outputs into ``text`` + optional ``negative_text``.

    Returns ``(text_cond, neg_text_cond)``; either may be ``None`` when the
    corresponding source field was missing across all results (e.g. no CFG → no
    negative branch). Concat is dim-0 across results.
    """
    prompt_embeds_list: List[torch.Tensor] = []
    pooled_list: List[torch.Tensor] = []
    mask_list: List[torch.Tensor] = []
    neg_embeds_list: List[torch.Tensor] = []
    neg_pooled_list: List[torch.Tensor] = []

    for result in results:
        embeds = fuse_encoder_outputs(result.prompt_embeds)
        require(
            embeds is not None,
            "SGLang result missing prompt_embeds — request must pin return_prompt_embeds=True",
        )
        prompt_embeds_list.append(embeds.detach().cpu())

        pooled = fuse_encoder_outputs(result.pooled_prompt_embeds)
        if pooled is not None:
            pooled_list.append(pooled.detach().cpu())

        attn_mask = fuse_encoder_outputs(result.encoder_attention_mask)
        if attn_mask is not None:
            mask_list.append(attn_mask.detach().cpu())

        neg_embeds = fuse_encoder_outputs(result.negative_prompt_embeds)
        if neg_embeds is not None:
            neg_embeds_list.append(neg_embeds.detach().cpu())

        neg_pooled = fuse_encoder_outputs(result.neg_pooled_prompt_embeds)
        if neg_pooled is not None:
            neg_pooled_list.append(neg_pooled.detach().cpu())

    embeds_cat = torch.cat(prompt_embeds_list, dim=0) if prompt_embeds_list else None
    text_cond = (
        TextEmbedCondition(
            embeds=embeds_cat,
            pooled=torch.cat(pooled_list, dim=0) if pooled_list else None,
            attn_mask=torch.cat(mask_list, dim=0) if mask_list else None,
        )
        if embeds_cat is not None
        else None
    )

    neg_embeds_cat = torch.cat(neg_embeds_list, dim=0) if neg_embeds_list else None
    neg_text_cond = (
        TextEmbedCondition(
            embeds=neg_embeds_cat,
            pooled=torch.cat(neg_pooled_list, dim=0) if neg_pooled_list else None,
            attn_mask=None,
        )
        if neg_embeds_cat is not None
        else None
    )

    return text_cond, neg_text_cond


__all__ = [
    "derive_timestep_alignment",
    "build_latent_segment",
    "stack_decoded_images",
    "fuse_text_conditions",
]
