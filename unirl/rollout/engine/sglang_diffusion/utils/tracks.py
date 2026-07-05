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
from unirl.rollout.engine.sglang_diffusion.utils.tensors import (
    decode_sample,
    fuse_encoder_outputs,
)
from unirl.rollout.engine.sigma_verify import verify_engine_used_sigmas
from unirl.types.conditions.text import TextEmbedCondition
from unirl.types.primitives import Images, Video, Videos
from unirl.types.rollout_req import RolloutReq
from unirl.types.sampling import compute_trajectory_positions
from unirl.types.segments.latent import LatentSegment, make_image_segment

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


def validate_packed_trajectory(
    traj: torch.Tensor,
    req: RolloutReq,
    *,
    family: str,
    downsample: int,
    require_divisible: bool = False,
) -> Tuple[int, int, int, int, int, int]:
    """Validate a packed ``[B, T, S, C]`` denoising trajectory; return its dims.

    Packed-token families (FLUX.2-Klein, Qwen-Image) receive the SGLang trajectory
    as 4-D ``[B, T, S, C_packed]`` (``S`` patch tokens). Checks ``S`` against the patch
    grid ``(height // downsample, width // downsample)`` and returns
    ``(B, T, S, C_packed, h_pat, w_pat)`` so the caller can flatten and apply its own
    model-specific unpack + reshape. The caller handles the 5-D image-form passthrough;
    this raises if ``traj`` is not the expected 4-D packed rank. ``family`` only feeds
    error messages; ``require_divisible`` rejects H/W not divisible by ``downsample``
    (Klein) vs. silently flooring (Qwen).
    """
    require(
        traj.ndim == 4,
        f"{family}: packed trajectory must be 4-D [B, T, S, C]; got rank {traj.ndim}, shape {tuple(traj.shape)}.",
    )
    diffusion = req.sampling_params.get("diffusion")
    height = int(diffusion.height) if diffusion.height is not None else None
    width = int(diffusion.width) if diffusion.width is not None else None
    require(
        height is not None and width is not None,
        f"{family}: need height/width from req.sampling_params to unpack the packed "
        f"[B, T, S, C] trajectory; both must be set.",
    )
    if require_divisible:
        require(
            height % downsample == 0 and width % downsample == 0,
            f"{family}: height ({height}) and width ({width}) must be divisible by the "
            f"VAE×patchify downsample ({downsample}).",
        )
    h_pat = height // downsample
    w_pat = width // downsample
    B, T, S, C_packed = traj.shape
    require(
        S == h_pat * w_pat,
        f"{family}: packed token count S={S} != h_pat*w_pat={h_pat * w_pat} "
        f"(from height={height}, width={width}). Schedule/recipe drift — fix the source "
        f"rather than silently reshape to a wrong spatial layout.",
    )
    return B, T, S, C_packed, h_pat, w_pat


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
        sde_logp = _native_sde_logp(results, num_steps=num_steps, sde_indices=sde_indices)

    return segment_factory(
        latents=trajectories_tensor,
        sigmas=sigmas,
        indices=indices_t,
        sde_logp=sde_logp,
        sde_indices=sde_indices_t,
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
        result.trajectory_log_probs.detach().cpu() if result.trajectory_log_probs is not None else None
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


def stack_decoded_images(
    results: Sequence[RawResult],
    *,
    squeeze_single_frame_4d: bool = True,
) -> Optional[Images]:
    """Stack per-result decoded ``samples`` into ``Images.pixels [B, C, H, W]``.

    Image-output adapters may opt into squeezing a singleton temporal axis
    ``[C, T=1, H, W]`` back to ``[C, H, W]``. Video-family adapters that still
    run through the legacy image path should disable this so a true single-frame
    video is dropped like any other 4-D video sample. Multi-frame 4-D samples are
    dropped with a warning either way (no Videos packing on the image track).
    """
    per_sample_tensors: List[torch.Tensor] = []
    skipped_video = False
    for result in results:
        canonical = decode_sample(result.samples)
        if canonical is None:
            continue
        if canonical.dim() == 3:
            per_sample_tensors.append(canonical.to(torch.float32))
        elif squeeze_single_frame_4d and canonical.dim() == 4 and int(canonical.shape[1]) == 1:
            per_sample_tensors.append(canonical.squeeze(1).to(torch.float32))
        elif canonical.dim() == 4:
            skipped_video = True
        else:
            raise RuntimeError(
                f"stack_decoded_images: unexpected canonical media rank {canonical.dim()}; want 3 (image) or 4 (video)."
            )
    if skipped_video:
        logger.warning(
            "SGLang result contained multi-frame 4D video samples while decoding "
            "an image track; dropping samples that cannot be represented as Images."
        )
    if not per_sample_tensors:
        return None
    return Images(pixels=torch.stack(per_sample_tensors, dim=0))


def stack_decoded_videos(results: Sequence[RawResult]) -> Optional[Videos]:
    """Pack per-result decoded video ``samples`` into a ragged ``Videos`` batch.

    The video counterpart of :func:`stack_decoded_images`. ``decode_sample``
    returns canonical channels-first video ``[C, T, H, W]`` (see
    :func:`unirl.rollout.engine.sglang_diffusion.utils.tensors.normalize_media`);
    the :class:`~unirl.types.primitives.Video` primitive — and the video reward
    consumer (``video_pickscore``, which permutes ``frames[T,C,H,W] → [C,T,H,W]``)
    — want frame-major ``[T, C, H, W]``, so we permute before packing.
    ``Videos.from_list`` concatenates along T and lets the Batch framework
    compute the per-sample ``cu_frames`` offsets. Each result carries exactly
    one decoded sample (mirrors :func:`stack_decoded_images`'s one-per-result
    contract). Returns ``None`` when no recognizable video was produced.
    """
    videos: List[Video] = []
    for result in results:
        canonical = decode_sample(result.samples)
        if canonical is None:
            continue
        if canonical.dim() != 4:
            raise RuntimeError(
                f"stack_decoded_videos: expected 4-D canonical video [C, T, H, W]; "
                f"got rank {canonical.dim()}, shape {tuple(canonical.shape)}."
            )
        frames = canonical.permute(1, 0, 2, 3).contiguous().to(torch.float32)  # [T, C, H, W]
        videos.append(Video(frames=frames))
    if not videos:
        return None
    return Videos.from_list(videos)


# ---------------------------------------------------------------------------
# Conditions packing
# ---------------------------------------------------------------------------


def _cat_padded_rows(tensors: List[torch.Tensor]) -> torch.Tensor:
    """dim-0 concat tolerating per-result seq-len (dim-1) differences.

    Variable-length text encoders (Qwen-VL) pad each server request to its own
    batch max, so chunked generates ship different seq lens per result.
    Right-pad dim 1 with zeros to the cross-result max — exactly the server's
    own zero-pad convention, and the parallel attention mask (padded the same
    way) keeps real-vs-pad explicit. Fixed-length encoders (SD3's CLIP/T5)
    always agree on dim 1 and concat unchanged.
    """
    if len(tensors) == 1:
        return tensors[0]
    lens = {int(t.shape[1]) for t in tensors}
    if len(lens) <= 1:
        return torch.cat(tensors, dim=0)
    max_len = max(lens)
    padded: List[torch.Tensor] = []
    for t in tensors:
        if int(t.shape[1]) < max_len:
            pad_shape = list(t.shape)
            pad_shape[1] = max_len - int(t.shape[1])
            t = torch.cat([t, t.new_zeros(pad_shape)], dim=1)
        padded.append(t)
    return torch.cat(padded, dim=0)


def _aligned_mask(
    mask_list: List[torch.Tensor],
    embeds_cat: Optional[torch.Tensor],
    *,
    allow_pad: bool = False,
) -> Optional[torch.Tensor]:
    """Fuse + mount an attention mask only when it aligns with the fused embeds.

    The engine emits the model's embeds-aligned ``prompt_embeds_mask`` (the mask the
    server's DiT itself attended under). Mount it only when its token axis (dim 1)
    matches the fused embeds: Qwen-Image's single-encoder mask matches and flows to
    replay; SD3's per-encoder mask fuses longer than the merged embeds, so it is
    dropped (SD3's ``predict_noise`` ignores the mask — dropping a mismatched mask
    is the safe, correct result and avoids padding the embeds up to a spurious
    length, the historic ~68x LoRA-gradient dilution).

    For Qwen-Image-Edit-Plus the text encoder emits prompt_embeds that include
    image-placeholder tokens (longer than the text-only attention mask). The
    extra positions are all valid (image tokens the DiT attends to), so pad the
    mask with ones up to the embeds seq-len instead of dropping it.
    """
    if not mask_list or embeds_cat is None:
        return None
    mask_cat = _cat_padded_rows(mask_list)
    mask_seq = int(mask_cat.shape[1])
    embeds_seq = int(embeds_cat.shape[1])
    if mask_seq == embeds_seq:
        return mask_cat
    if mask_seq > embeds_seq:
        logger.debug(
            "Dropping attention mask: fused seq-len %d != embeds seq-len %d (mask not embeds-aligned for this family).",
            mask_seq,
            embeds_seq,
        )
        return None
    # mask_seq < embeds_seq: pad with ones only when the adapter opts in
    # (Edit-Plus prompt_embeds carry image-token slots beyond the text mask).
    if not allow_pad:
        logger.debug(
            "Dropping attention mask: fused seq-len %d != embeds seq-len %d (mask not embeds-aligned for this family).",
            mask_seq,
            embeds_seq,
        )
        return None
    batch = mask_cat.shape[0]
    pad = torch.ones((batch, embeds_seq - mask_seq), dtype=mask_cat.dtype, device=mask_cat.device)
    return torch.cat([mask_cat, pad], dim=1)


def fuse_text_conditions(
    results: Sequence[RawResult],
    *,
    allow_mask_pad: bool = False,
) -> Tuple[Optional[TextEmbedCondition], Optional[TextEmbedCondition]]:
    """Fuse per-result encoder outputs into ``text`` + optional ``negative_text``.

    Returns ``(text_cond, neg_text_cond)``; either may be ``None`` when the
    corresponding source field was missing across all results (e.g. no CFG → no
    negative branch). Concat is dim-0 across results; token-axis (dim-1)
    differences between results are zero-padded to the max
    (:func:`_cat_padded_rows`) with the attention masks padded in lockstep.
    """
    prompt_embeds_list: List[torch.Tensor] = []
    pooled_list: List[torch.Tensor] = []
    mask_list: List[torch.Tensor] = []
    neg_embeds_list: List[torch.Tensor] = []
    neg_pooled_list: List[torch.Tensor] = []
    neg_mask_list: List[torch.Tensor] = []

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

        # Negative mask: required alongside negative embeds by mask-consuming
        # replay paths (Qwen-VL conditioning) — fused symmetrically with the
        # positive mask rather than dropped.
        neg_mask = fuse_encoder_outputs(result.negative_attention_mask)
        if neg_mask is not None:
            neg_mask_list.append(neg_mask.detach().cpu())

    embeds_cat = _cat_padded_rows(prompt_embeds_list) if prompt_embeds_list else None

    text_cond = (
        TextEmbedCondition(
            embeds=embeds_cat,
            pooled=torch.cat(pooled_list, dim=0) if pooled_list else None,
            attn_mask=_aligned_mask(mask_list, embeds_cat, allow_pad=allow_mask_pad),
        )
        if embeds_cat is not None
        else None
    )

    neg_embeds_cat = _cat_padded_rows(neg_embeds_list) if neg_embeds_list else None
    neg_text_cond = (
        TextEmbedCondition(
            embeds=neg_embeds_cat,
            pooled=torch.cat(neg_pooled_list, dim=0) if neg_pooled_list else None,
            attn_mask=_aligned_mask(neg_mask_list, neg_embeds_cat, allow_pad=allow_mask_pad),
        )
        if neg_embeds_cat is not None
        else None
    )

    return text_cond, neg_text_cond


__all__ = [
    "derive_timestep_alignment",
    "build_latent_segment",
    "stack_decoded_images",
    "stack_decoded_videos",
    "fuse_text_conditions",
]
