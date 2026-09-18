"""Assemble ``Sample`` Part pieces (segment / decoded / conditions) from raw results."""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional, Sequence, Tuple

import torch

from unirl.config.require import require
from unirl.rollout.engine.sglang_diffusion.backends import RawResult
from unirl.rollout.engine.sglang_diffusion.utils.tensors import (
    decode_sample,
    fuse_encoder_outputs,
)
from unirl.rollout.engine.sigma_verify import verify_engine_used_sigmas
from unirl.types.conditions.text import TextEmbedCondition
from unirl.types.primitives import Image, Images, Video, Videos
from unirl.types.sampling import compute_trajectory_positions
from unirl.types.segments.latent import LatentSegment, make_image_segment

logger = logging.getLogger(__name__)


def collect_trajectory_latents(results: Sequence[RawResult]) -> torch.Tensor:
    """Concat per-result trajectory latents on the batch dim (detached, CPU)."""
    latents = []
    for r in results:
        require(r.trajectory_latents is not None, "SGLang result missing trajectory_latents")
        latents.append(r.trajectory_latents.detach().cpu())
    return torch.cat(latents, dim=0)


def collect_aux_trajectory_latents(results: Sequence[RawResult]) -> Optional[torch.Tensor]:
    """Concat per-result AUXILIARY trajectory latents (LTX-2 audio) on the batch dim."""
    auxes = [getattr(r, "aux_trajectory_latents", None) for r in results]
    present = [a is not None for a in auxes]
    if not any(present):
        return None
    require(all(present), "SGLang results inconsistent: some carry aux_trajectory_latents, some do not")
    return torch.cat([a.detach().cpu() for a in auxes], dim=0)


def validate_packed_trajectory(
    traj: torch.Tensor,
    diffusion: Any,
    *,
    family: str,
    downsample: int,
    require_divisible: bool = False,
) -> Tuple[int, int, int, int, int, int]:
    """Validate a packed ``[B, T, S, C]`` denoising trajectory; return its dims."""
    require(
        traj.ndim == 4,
        f"{family}: packed trajectory must be 4-D [B, T, S, C]; got rank {traj.ndim}, shape {tuple(traj.shape)}.",
    )
    height = int(diffusion.height) if diffusion.height is not None else None
    width = int(diffusion.width) if diffusion.width is not None else None
    require(
        height is not None and width is not None,
        f"{family}: need height/width from the diffusion sampling params to unpack the packed "
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
    """Validate the T+1 trajectory shape and verify SGLang used the σ we sent."""
    traj_len = int(trajectories_tensor.shape[1])
    expected_len = int(expected_sigmas.shape[0])
    require(
        traj_len == expected_len,
        f"SGLang trajectory length {traj_len} != expected_sigmas length {expected_len}. "
        f"Modern SGLang prepends initial latents at "
        f"sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py so "
        f"trajectory carries T+1 latents; expected_sigmas (from sampling_params.sigmas) is T+1 "
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
    aux_trajectory: Optional[torch.Tensor] = None,
) -> LatentSegment:
    """Pack an (already-unpacked) trajectory tensor into one batched ``LatentSegment``."""
    sigmas, step_indices = derive_timestep_alignment(
        trajectories_tensor=trajectories_tensor,
        expected_sigmas=expected_sigmas,
        results=results,
    )

    traj_len = int(trajectories_tensor.shape[1])
    if aux_trajectory is not None:
        require(
            aux_trajectory.ndim >= 2,
            "Auxiliary trajectory must be at least 2-D [B, T+1, ...]; "
            f"got rank {aux_trajectory.ndim}, shape {tuple(aux_trajectory.shape)}.",
        )
        require(
            int(aux_trajectory.shape[0]) == int(trajectories_tensor.shape[0]),
            "Auxiliary/video trajectory batch mismatch: "
            f"{int(aux_trajectory.shape[0])} != {int(trajectories_tensor.shape[0])}.",
        )
        require(
            int(aux_trajectory.shape[1]) == traj_len,
            "Auxiliary/video trajectory length mismatch: "
            f"{int(aux_trajectory.shape[1])} != {traj_len}. Both must carry T+1 states.",
        )
    indices_t: torch.Tensor = step_indices
    if sde_indices is not None and len(sde_indices) < num_steps:
        needed = set(compute_trajectory_positions(set(sde_indices), num_steps))
        needed.add(int(num_steps))
        keep_cols = sorted(p for p in needed if 0 <= p < traj_len)
        if keep_cols and len(keep_cols) < traj_len:
            trajectories_tensor = trajectories_tensor[:, keep_cols]
            indices_t = torch.tensor(keep_cols, dtype=torch.long)
            if aux_trajectory is not None:
                aux_trajectory = aux_trajectory[:, keep_cols]

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
        aux_latents=aux_trajectory,
    )


def _native_sde_logp(
    results: Sequence[RawResult],
    *,
    num_steps: int,
    sde_indices: Optional[List[int]],
) -> Optional[torch.Tensor]:
    """Best-effort extract of SGLang's native ``trajectory_log_probs`` into ``[B, S]``."""
    per_result: List[Optional[torch.Tensor]] = [
        result.trajectory_log_probs.detach().cpu() if result.trajectory_log_probs is not None else None
        for result in results
    ]
    if any(lp is None for lp in per_result):
        return None
    log_prob_tensor = torch.cat([lp for lp in per_result if lp is not None], dim=0)
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


def pack_decoded_images(
    results: Sequence[RawResult],
    *,
    squeeze_single_frame_4d: bool = True,
) -> Optional[Images]:
    """Pack per-result decoded ``samples`` into an ``Images`` batch."""
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
                f"pack_decoded_images: unexpected canonical media rank {canonical.dim()}; want 3 (image) or 4 (video)."
            )
    if skipped_video:
        logger.warning(
            "SGLang result contained multi-frame 4D video samples while decoding "
            "an image track; dropping samples that cannot be represented as Images."
        )
    if not per_sample_tensors:
        return None
    return Images.from_list([Image(pixels=pixels) for pixels in per_sample_tensors])


def stack_decoded_videos(results: Sequence[RawResult]) -> Optional[Videos]:
    """Pack per-result decoded video ``samples`` into a ragged ``Videos`` batch."""
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
        frames = canonical.permute(1, 0, 2, 3).contiguous().to(torch.float32)
        videos.append(Video(frames=frames))
    if not videos:
        return None
    return Videos.from_list(videos)


def _cat_padded_rows(tensors: List[torch.Tensor]) -> torch.Tensor:
    """dim-0 concat tolerating per-result seq-len (dim-1) differences."""
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
    """Fuse + mount an attention mask only when it aligns with the fused embeds."""
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
    """Fuse per-result encoder outputs into ``text`` + optional ``negative_text``."""
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
    "pack_decoded_images",
    "stack_decoded_videos",
    "fuse_text_conditions",
]
