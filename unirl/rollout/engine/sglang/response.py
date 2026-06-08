"""SGLang ``GenerationResult`` list → ``RolloutResp`` translator.

Single free function ``_to_rollout_resp(req, results, *, cfg, num_steps, shift,
sde_indices, emit_native_logprob)`` produces:

- ``resp.tracks['image'].segment`` = :class:`LatentSegment` with ``latents``,
  ``sigmas``, ``indices`` always populated; ``sde_indices``
  populated when the algorithm requested SDE log-probs. ``sde_logp`` is a
  *best-effort* native emission: populated when ``emit_native_logprob`` and the
  SGLang build returns ``trajectory_log_probs``, else left ``None`` (the trainer
  decides whether to use it or replay — see ``algorithm.old_logp_source``).
- ``resp.tracks['image'].decoded`` = :class:`Images` (``float32 [B, C, H, W]``
  in ``[0, 1]``) built from SGLang's per-result ``samples`` output, or ``None``
  if SGLang returned no decoded samples. Video samples surface as
  ``[C, T, H, W]`` and are deferred (TODO — first video consumer).
- ``resp.tracks['image'].conditions['text']`` + (when CFG active)
  ``resp.tracks['image'].conditions['negative_text']`` populated from SGLang's ``prompt_embeds`` /
  ``pooled_prompt_embeds`` / ``encoder_attention_mask`` / ``negative_*``
  outputs when ``cfg.populate_conditions=True``. The slot key
  ``"negative_text"`` matches the SD3 / Mochi typed-container convention so
  trainer-side ``*DiffusionConditions.from_dict(resp.tracks['image'].conditions)`` consumers
  pick up the negative branch automatically.

Trajectory validation (T+1 invariant, sigma-schedule cross-check) and
selective-trim heuristics (``compute_trajectory_positions``) port verbatim
from legacy ``samplers/sglang/response.py``. Trainer-side replay
reconstructs typed conditions from ``resp.tracks[slot].conditions``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple

import torch

from unirl.config.require import require
from unirl.rollout.engine.sglang._sample_decode import decode_sample
from unirl.rollout.engine.sglang._text_fusion import fuse_text_encoder_outputs
from unirl.rollout.engine.sglang.config import SGLangEngineConfig
from unirl.rollout.engine.sigma_verify import verify_engine_used_sigmas
from unirl.types.conditions.text import TextEmbedCondition
from unirl.types.primitives import Images
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp, RolloutTrack
from unirl.types.sampling import get_diffusion_params
from unirl.types.segments.latent import LatentSegment, make_image_segment
from unirl.types.trajectory_store import compute_trajectory_positions

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.entrypoints.utils import GenerationResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Upstream rollout-trajectory accessors (flat fork fields -> nested upstream)
# ---------------------------------------------------------------------------
#
# Stock upstream packs the rollout trajectory + native log-probs into a nested
# ``GenerationResult.rollout_trajectory_data`` (RolloutTrajectoryData) instead of
# the fork's flat ``trajectory_latents`` / ``trajectory_timesteps`` /
# ``trajectory_log_probs``. These accessors navigate that path, tolerating any
# missing level (return None) to match the prior getattr defaults. GRPO uses the
# ``dit_trajectory`` latents (gated by ``rollout_return_dit_trajectory`` in the
# request) so the trajectory stays aligned with ``rollout_log_probs``.


def _traj_latents(result: Any) -> Optional[torch.Tensor]:
    rtd = getattr(result, "rollout_trajectory_data", None)
    return getattr(getattr(rtd, "dit_trajectory", None), "latents", None)


def _traj_timesteps(result: Any) -> Optional[torch.Tensor]:
    rtd = getattr(result, "rollout_trajectory_data", None)
    return getattr(getattr(rtd, "dit_trajectory", None), "timesteps", None)


def _traj_log_probs(result: Any) -> Optional[torch.Tensor]:
    rtd = getattr(result, "rollout_trajectory_data", None)
    return getattr(rtd, "rollout_log_probs", None)


# ---------------------------------------------------------------------------
# Trajectory alignment + sigma cross-check (ported from legacy)
# ---------------------------------------------------------------------------


def _derive_timestep_alignment(
    *,
    trajectories_tensor: torch.Tensor,
    expected_sigmas: torch.Tensor,
    results: Sequence[Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Validate T+1 trajectory shape and verify SGLang used the σ we sent.

    ``expected_sigmas`` is the σ schedule the engine pinned on
    ``RolloutReq.sigmas`` and forwarded via SGLang's
    ``DiffusionSamplingParams.sigmas`` → ``set_timesteps(sigmas=...)``. SGLang
    echoes the same values back via ``trajectory_timesteps`` per
    result; :func:`verify_engine_used_sigmas` asserts elementwise
    equality (with dynamic scale-normalization for sglang builds that
    emit raw ``sigma * num_train_timesteps`` instead of [0, 1] —
    handled inside the helper). Together these guarantee SGLang
    rollout and training-side replay (which reads ``segment.sigmas``)
    used numerically identical σ schedules.

    Design note: main-unified-base commit ``707cc609`` switched to
    "extract σ from sglang's trajectory_timesteps as SOT" (with a
    warning-only drift check). That approach silently keeps training
    going when sglang deviates from main-repo intent — the GRPO
    invariant holds but the model trains under un-intended σ. We keep
    the **main-repo-as-SOT + fatal drift assert** direction here
    because (a) celve fork ``2c5a2ecec`` ensures sglang honors the
    sigmas we send, (b) any deviation surfaces a real supply-chain
    bug we want loud, not silent.
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
            _traj_timesteps(result),
            expected=expected_cpu,
            engine_name=f"sglang (result {i})",
        )
    return expected_cpu, step_indices


# ---------------------------------------------------------------------------
# Segment build
# ---------------------------------------------------------------------------


def _maybe_unpack_packed_trajectory(
    trajectories: torch.Tensor,
    *,
    model_family: str,
    height: Optional[int],
    width: Optional[int],
) -> torch.Tensor:
    """Convert SGLang's packed sequence-style trajectory to the trainer's image-style.

    FLUX.2-klein's SGLang pipeline emits trajectories in packed form
    ``[B, T, H_pat * W_pat, C_packed]`` because Klein's transformer is a
    pure sequence model that takes ``[B, S, C_packed]`` tokens (each token =
    one 2x2 patch with channels concatenated). The trainer-side
    ``Flux2KleinDiffusionStage.replay`` expects the same shape the trainside
    pipeline emits — ``[B, T, C_packed, H_pat, W_pat]`` (5-D, with patchified
    spatial dims preserved) — and fails fast on the 4-D packed shape with
    ``expected latents [B, K, C, H_pat, W_pat]``.

    Other model families (SD3 etc.) keep latents in image form throughout
    the SGLang stack so their trajectories arrive 5-D and pass through
    untouched. The 4-D → 5-D unpack only fires when the rollout-side fork
    actually emits packed tokens AND the model family is one we know wants
    image-form latents on the trainer side.
    """
    if trajectories.ndim == 5:
        return trajectories
    if trajectories.ndim != 4:
        raise ValueError(
            f"_maybe_unpack_packed_trajectory: SGLang trajectory has rank "
            f"{trajectories.ndim}, want 4 (packed) or 5 (image-form); shape="
            f"{tuple(trajectories.shape)}."
        )
    if model_family != "flux2_klein":
        raise ValueError(
            f"_maybe_unpack_packed_trajectory: 4-D trajectory only supported for "
            f"model_family='flux2_klein' (Klein emits packed [B, T, H*W, C] from "
            f"SGLang); got model_family={model_family!r}, shape="
            f"{tuple(trajectories.shape)}."
        )
    if height is None or width is None:
        raise ValueError(
            "_maybe_unpack_packed_trajectory: need height/width from "
            "req.sampling_params to unpack Klein's packed [B, T, H*W, C] "
            "trajectory; both must be set."
        )
    # FLUX.2 patchified spatial size: pixel / (vae_scale_factor=8 * patchify_factor=2).
    # Mirrors Flux2KleinDiffusionStage._patchified_shape.
    _DOWNSAMPLE = 16
    if height % _DOWNSAMPLE or width % _DOWNSAMPLE:
        raise ValueError(
            f"_maybe_unpack_packed_trajectory: Klein height ({height}) and width "
            f"({width}) must be divisible by VAE x patchify downsample ({_DOWNSAMPLE})."
        )
    h_pat = height // _DOWNSAMPLE
    w_pat = width // _DOWNSAMPLE
    B, T, S, C_packed = trajectories.shape
    if S != h_pat * w_pat:
        raise ValueError(
            f"_maybe_unpack_packed_trajectory: packed token count S={S} != "
            f"h_pat * w_pat = {h_pat * w_pat} (derived from height={height}, "
            f"width={width}). Schedule/recipe drift — fix the source rather than "
            f"silently reshape to a wrong spatial layout."
        )
    from unirl.models.flux2_klein.flux2_klein_utils import unpack_latents

    flat = trajectories.reshape(B * T, S, C_packed)
    return unpack_latents(flat, h_pat, w_pat).reshape(B, T, C_packed, h_pat, w_pat).contiguous()


def _build_image_segment(
    results: Sequence["GenerationResult"],
    *,
    expected_sigmas: torch.Tensor,
    num_steps: int,
    sde_indices: Optional[List[int]],
    emit_native_logprob: bool,
    model_family: str,
    height: Optional[int],
    width: Optional[int],
) -> LatentSegment:
    """Pack per-result trajectory tensors into one batched ``LatentSegment``."""
    trajectory_items: List[torch.Tensor] = []
    for result in results:
        traj = _traj_latents(result)
        require(
            traj is not None,
            "SGLang result missing rollout_trajectory_data.dit_trajectory.latents "
            "(rollout requests must set rollout=True + rollout_return_dit_trajectory=True)",
        )
        trajectory_items.append(traj.detach().cpu())
    trajectories_tensor = torch.cat(trajectory_items, dim=0)
    trajectories_tensor = _maybe_unpack_packed_trajectory(
        trajectories_tensor,
        model_family=model_family,
        height=height,
        width=width,
    )

    sigmas, step_indices = _derive_timestep_alignment(
        trajectories_tensor=trajectories_tensor,
        expected_sigmas=expected_sigmas,
        results=results,
    )

    # NOTE: the warning-only ``_verify_sglang_timesteps`` block from
    # main-unified-base commit ``707cc609`` is deliberately *not*
    # ported. ``_derive_timestep_alignment`` above already runs the
    # fatal-on-mismatch :func:`verify_engine_used_sigmas` per result;
    # adding a second (warning-level) verify pass would be redundant
    # and weaken the "fail loudly on supply-chain drift" property.

    # Selective trim: when only a subset of trajectory positions is referenced
    # by the SDE step set, drop unused columns to save Ray IPC bandwidth.
    # ``compute_trajectory_positions`` returns only the (i, i+1) pairs for
    # SDE-gated steps — for ``sde_indices={5}`` at T=10 that's just
    # {5, 6}, *not* the terminal clean latent at position T=10. Downstream
    # legacy bridges read ``samples.latents = seg.latents[:, -1]`` and
    # feed that to VAE decode, so we always preserve T as well to keep
    # the clean image latent available.
    traj_len = int(trajectories_tensor.shape[1])
    indices_t: torch.Tensor = step_indices
    if sde_indices is not None and len(sde_indices) < num_steps:
        needed = set(compute_trajectory_positions(set(sde_indices), num_steps))
        needed.add(int(num_steps))  # always preserve terminal clean latent
        keep_cols = sorted(p for p in needed if 0 <= p < traj_len)
        if keep_cols and len(keep_cols) < traj_len:
            trajectories_tensor = trajectories_tensor[:, keep_cols]
            indices_t = torch.tensor(keep_cols, dtype=torch.long)

    # sde_indices: always populated (trainer needs to know which steps to replay).
    # sde_logp: only populated in rollout mode; replay mode computes it on trainer side.
    sde_indices_t: Optional[torch.Tensor] = (
        torch.tensor(list(sde_indices), dtype=torch.long)
        if sde_indices is not None
        else torch.arange(num_steps, dtype=torch.long)
    )
    sde_logp: Optional[torch.Tensor] = None
    if emit_native_logprob:
        per_result_log_probs: List[Optional[torch.Tensor]] = []
        for result in results:
            lp = _traj_log_probs(result)
            per_result_log_probs.append(lp.detach().cpu() if lp is not None else None)
        # Best-effort emit: if this build returned no per-step log-probs for any
        # result, leave ``sde_logp = None`` and let the trainer decide — replay
        # recomputes; rollout (``algorithm.old_logp_source='rollout'``) raises in
        # ``prepare_segment`` with an actionable message. The engine stays silent:
        # it can't know the intent, and for an intentional replay run a missing
        # emission is expected, not a warning-worthy condition.
        if all(lp is not None for lp in per_result_log_probs):
            log_prob_tensor = torch.cat([lp for lp in per_result_log_probs if lp is not None], dim=0)
            # trajectory_log_probs shape: [B, T] (one entry per SDE transition).
            # When sde_indices is None (rollout SDE mode used full schedule), the
            # second dim equals num_steps. When sde_indices is a subset, SGLang is
            # supposed to emit log-probs at the requested transitions only — but
            # some SGLang builds always emit the full schedule. Tolerate that case
            # by selecting the requested columns; only fail on shapes that can't
            # be reconciled either way.
            s_dim = int(log_prob_tensor.shape[1])
            expected_s = len(sde_indices) if sde_indices is not None else num_steps
            if s_dim == num_steps and sde_indices is not None and expected_s < num_steps:
                # Server emitted full schedule; slice down to the requested SDE indices.
                keep_idx = torch.tensor(sorted(int(i) for i in sde_indices), dtype=torch.long)
                log_prob_tensor = log_prob_tensor.index_select(1, keep_idx)
                s_dim = int(log_prob_tensor.shape[1])
            require(
                s_dim == expected_s,
                f"SGLang trajectory_log_probs shape {tuple(log_prob_tensor.shape)} second "
                f"dim={s_dim} does not match expected SDE-step count {expected_s}. "
                f"sigma_schedule / num_inference_steps / sde_indices drift — fix the "
                f"source rather than emit a misaligned anchor.",
            )
            sde_logp = log_prob_tensor

    return make_image_segment(
        latents=trajectories_tensor,
        sigmas=sigmas,
        indices=indices_t,
        sde_logp=sde_logp,
        sde_indices=sde_indices_t,
    )


# ---------------------------------------------------------------------------
# Decoded media
# ---------------------------------------------------------------------------


def _build_decoded_images(
    results: Sequence["GenerationResult"],
) -> Optional[Images]:
    """Stack per-result decoded ``samples`` into ``Images.pixels [B, C, H, W]``."""
    per_sample_tensors: List[torch.Tensor] = []
    skipped_video = False
    for result in results:
        canonical = decode_sample(getattr(result, "samples", None))
        if canonical is None:
            continue
        if canonical.dim() == 3:
            per_sample_tensors.append(canonical.to(torch.float32))
        elif canonical.dim() == 4:
            # [C, T, H, W] — video. TODO: surface as Videos primitive once a
            # video reward consumer exists. For now drop with a warning so
            # reward scoring fails fast rather than silently using the wrong
            # tensor.
            skipped_video = True
        else:
            raise RuntimeError(
                f"_build_decoded_images: unexpected canonical media rank "
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


def _build_text_conditions(
    results: Sequence["GenerationResult"],
) -> Tuple[Optional[TextEmbedCondition], Optional[TextEmbedCondition]]:
    """Fuse per-result encoder outputs into ``text`` + optional ``negative_text``.

    Returns ``(text_cond, neg_text_cond)``. Either may be ``None`` when the
    corresponding source field was missing across all results (e.g. no CFG →
    no negative branch). Concat is dim-0 across results.
    """
    prompt_embeds_list: List[torch.Tensor] = []
    pooled_list: List[torch.Tensor] = []
    mask_list: List[torch.Tensor] = []
    neg_embeds_list: List[torch.Tensor] = []
    neg_pooled_list: List[torch.Tensor] = []

    for result in results:
        embeds = fuse_text_encoder_outputs(getattr(result, "prompt_embeds", None))

        require(
            embeds is not None,
            "SGLang result missing prompt_embeds — request must pin return_prompt_embeds=True",
        )
        prompt_embeds_list.append(embeds.detach().cpu())

        pooled = fuse_text_encoder_outputs(getattr(result, "pooled_prompt_embeds", None))
        if pooled is not None:
            pooled_list.append(pooled.detach().cpu())

        attn_mask = fuse_text_encoder_outputs(getattr(result, "encoder_attention_mask", None))
        if attn_mask is not None:
            mask_list.append(attn_mask.detach().cpu())

        neg_embeds = fuse_text_encoder_outputs(getattr(result, "negative_prompt_embeds", None))
        if neg_embeds is not None:
            neg_embeds_list.append(neg_embeds.detach().cpu())

        neg_pooled = fuse_text_encoder_outputs(getattr(result, "neg_pooled_prompt_embeds", None))
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


# ---------------------------------------------------------------------------
# Top-level translator
# ---------------------------------------------------------------------------


def _to_rollout_resp(
    req: RolloutReq,
    results: Sequence["GenerationResult"],
    *,
    cfg: SGLangEngineConfig,
    num_steps: int,
    sde_indices: Optional[List[int]],
    emit_native_logprob: bool,
) -> RolloutResp:
    """Translate one SGLang batch result into the typed ``RolloutResp`` container."""
    require(bool(results), "_to_rollout_resp: SGLang returned no results")
    require(
        req.sigmas is not None,
        "_to_rollout_resp: req.sigmas must be set (SGLangRolloutEngine populates "
        "it before dispatch). Without it we can't verify SGLang used the same "
        "schedule the trainer will replay against.",
    )

    # Pull spatial dims from per-request sampling params so families that emit
    # packed sequence-style trajectories (Klein) can be unpacked back to the
    # image-form ``[B, T, C, H_pat, W_pat]`` the trainer-side replay expects.
    diffusion_params = get_diffusion_params(req.sampling_params)
    req_height = int(diffusion_params.height) if diffusion_params.height is not None else None
    req_width = int(diffusion_params.width) if diffusion_params.width is not None else None

    segment = _build_image_segment(
        results,
        expected_sigmas=req.sigmas,
        num_steps=num_steps,
        sde_indices=sde_indices,
        emit_native_logprob=emit_native_logprob,
        model_family=str(cfg.model_family),
        height=req_height,
        width=req_width,
    )

    decoded_images = _build_decoded_images(results)

    conditions: dict = {}
    if cfg.populate_conditions:
        text_cond, neg_text_cond = _build_text_conditions(results)
        if text_cond is not None:
            conditions["text"] = text_cond
        if neg_text_cond is not None:
            conditions["negative_text"] = neg_text_cond

    return RolloutResp(
        tracks={
            "image": RolloutTrack(
                sample_ids=list(req.sample_ids),
                parent_ids=list(req.group_ids),
                conditions=conditions,
                segment=segment,
                decoded=decoded_images,
            ),
        }
    )


__all__ = ["_to_rollout_resp"]
