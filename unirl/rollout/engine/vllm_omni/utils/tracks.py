"""Response-side segment / decoded / track-assembly mechanics.

Pure helpers the adapters' ``build_response`` steps call — they operate on
already-fetched wire data (the seam's :class:`OmniRawResult` protocol;
``SimpleNamespace`` fakes satisfy it structurally in tests) and the
trainer-facing types. No runtime import, no engine state.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from unirl.rollout.engine.sigma_verify import verify_engine_used_sigmas
from unirl.types.conditions import Condition
from unirl.types.primitives import Image, Images, Text, Texts, Video, Videos
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp, RolloutTrack
from unirl.types.segments import Segment
from unirl.types.segments.latent import make_image_segment


def seed_from_sample_id(sample_id: str) -> int:
    """Deterministic 31-bit diffusion seed for one image, keyed by sample_id.

    The M images of a recaption MUST draw distinct noise (else the diffusion
    GRPO advantage is identically 0 — the whole group collapses to the same
    reward). Seeds cannot vary through ``OmniDiffusionSamplingParams`` because
    vllm-omni requires exactly one sampling-params object PER STAGE (not per
    prompt) and shares it across every prompt of a ``generate()`` call — so
    the ``dit_recaption`` adapter issues one call per prompt with its own
    seed derived HERE from the unique sample_id (globally distinct AND
    reproducible). ``< 2**31`` matches vllm-omni's own random-seed fallback.
    """
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def pils_to_images(pil_images: Sequence[Any]) -> Images:
    """``[PIL.Image, …] → Images`` (float32 ``[B, C, H, W]`` in [0, 1])."""
    if not pil_images:
        raise ValueError("pils_to_images: empty image list")
    from torchvision.transforms.functional import pil_to_tensor

    items: List[Image] = []
    for pil in pil_images:
        # uint8 [C, H, W] / 255 → float32 [0, 1] per Image(pixels=...) contract.
        t = pil_to_tensor(pil).to(torch.float32) / 255.0
        items.append(Image(pixels=t))
    return Images.from_list(items)


def grouped_pils_to_videos(pil_frames_per_prompt: Sequence[Sequence[Any]]) -> Videos:
    """Group per-prompt PIL frame lists into ``Videos``.

    Upstream HV1.5 ``post_process_func`` returns ``List[List[PIL.Image]]``
    (one frame list per prompt). Reassemble each as a
    ``Video(frames=[T, C, H, W])`` so the reward layer's ``RewardRequest.videos``
    (which permutes to ``[C, T, H, W]``) gets the right shape.
    """
    if not pil_frames_per_prompt:
        raise ValueError("grouped_pils_to_videos: empty per-prompt frame lists")
    from torchvision.transforms.functional import pil_to_tensor

    items: List[Video] = []
    for frames in pil_frames_per_prompt:
        if not frames:
            raise ValueError("grouped_pils_to_videos: prompt has zero frames")
        frame_tensors = [pil_to_tensor(f).to(torch.float32) / 255.0 for f in frames]
        items.append(Video(frames=torch.stack(frame_tensors, dim=0)))
    return Videos.from_list(items)


def pick_stage_output(
    outputs: Sequence[Any],
    *,
    final_output_type: str,
    stage_id: Optional[int] = None,
) -> Optional[Any]:
    """Find the result with the requested ``final_output_type``.

    Falls back to ``stage_id`` match if provided. Returns ``None`` if neither
    matches — callers decide whether that's an error.
    """
    for out in outputs:
        if getattr(out, "final_output_type", None) == final_output_type:
            return out
    if stage_id is not None:
        for out in outputs:
            if getattr(out, "stage_id", None) == stage_id:
                return out
    return None


_VIDEO_PROCESSOR = None


def _video_frames_from_custom_output(diff_out: Any) -> List[Any]:
    """Recover a video sample's PIL frames from the decoded-video tensor the RL
    pipeline stamped onto ``custom_output["rl_decoded_video"]``.

    The engine decodes video to PIL frames, but those don't survive the engine
    worker->client wire (only tensors carried on custom_output / trajectory_*
    cross). The hv15 RL pipeline stamps the decoded tensor ``[B, C, F, H, W]``
    (``B == 1`` per request) so we can rebuild the frames here for the reward.
    """
    co = getattr(diff_out, "custom_output", None) or {}
    vid = co.get("rl_decoded_video")
    if vid is None or not torch.is_tensor(vid):
        return []
    global _VIDEO_PROCESSOR
    if _VIDEO_PROCESSOR is None:
        from diffusers.video_processor import VideoProcessor

        _VIDEO_PROCESSOR = VideoProcessor(vae_scale_factor=16)
    frames = _VIDEO_PROCESSOR.postprocess_video(vid, output_type="pil")
    # postprocess_video returns List[List[PIL]] (batch x frames); B == 1.
    if frames and isinstance(frames[0], list):
        return frames[0]
    return list(frames)


def collect_dit_outputs(
    per_request: Sequence[Sequence[Any]],
    *,
    final_output_type: str,
    stage_id: int,
    modality: str,
) -> Tuple[List[Any], List[List[Any]], List[Any]]:
    """Pick each request's DiT output + its PIL payload(s).

    Returns ``(diff_outputs, pil_frames_per_prompt, pil_images_flat)`` —
    video shapes consume the per-prompt frame groupings, image shapes the
    flat list. Raises when a request has no DiT output or no PILs surfaced
    (DiT stage failure / pipeline forward didn't populate the output).
    """
    diff_outputs: List[Any] = []
    pil_frames_per_prompt: List[List[Any]] = []
    pil_images: List[Any] = []
    for outputs in per_request:
        diff_out = pick_stage_output(outputs, final_output_type=final_output_type, stage_id=stage_id)
        if diff_out is None:
            raise RuntimeError(
                f"collect_dit_outputs: no {final_output_type} output for request "
                f"(modality={modality}); did the DiT stage fail?"
            )
        diff_outputs.append(diff_out)
        imgs = getattr(diff_out, "images", None) or []
        if not imgs and final_output_type == "video":
            # Video PIL frames don't survive the engine worker->client wire; the
            # RL pipeline stamped the decoded video tensor onto custom_output —
            # recover this sample's frames from there. (LIN-382)
            imgs = _video_frames_from_custom_output(diff_out)
        pil_frames_per_prompt.append(list(imgs))
        pil_images.extend(imgs)
    if not pil_images:
        raise RuntimeError(
            "collect_dit_outputs: DiT outputs carry no PIL images; "
            "check pipeline forward populated DiffusionOutput.output."
        )
    return diff_outputs, pil_frames_per_prompt, pil_images


def build_image_segment(
    diff_outputs: Sequence[Any],
    *,
    expected_sigmas: Optional[torch.Tensor] = None,
) -> Any:
    """Build ``LatentSegment`` from the DiT stage's per-request outputs.

    Each per-prompt result carries its own ``trajectory_latents`` /
    ``trajectory_log_probs`` (``[1, T+1, ...]`` / ``[1, K]`` — with
    ``runtime.max_inflight=1`` they are NOT shared refs to a full-batch
    tensor); concatenate across outputs to recover ``[B, T+1, ...]`` /
    ``[B, K]``. ``sigmas`` / ``indices`` / ``sde_indices`` are sample-shared,
    read off the first output:

    - ``sigmas`` from ``trajectory_timesteps`` — the field name reads
      "timesteps" but the ``RL*Pipeline.forward`` overwrites its contents with
      the true [0, 1] σ schedule (``[T+1]``); do not "fix" the misnomer.
      Verified against ``expected_sigmas`` (the engine-pinned ``req.sigmas``)
      via :func:`verify_engine_used_sigmas` so a broken wire surfaces here.
    - ``sde_logp`` from ``trajectory_log_probs`` ``[B, K]`` (K = SDE-gated
      step count; can be < T for sparse SDE, 0 for NFT/forward-process).
    - ``indices`` — dense ``arange(T+1)`` storage slots; ``sde_indices`` — the
      sparse step ids echoed via ``custom_output["sde_step_indices"]``.
    """
    per_latents: List[torch.Tensor] = []
    per_log_probs: List[torch.Tensor] = []
    for diff_out in diff_outputs:
        traj_l = getattr(diff_out, "trajectory_latents", None)
        if traj_l is not None:
            per_latents.append(traj_l)
        traj_lp = getattr(diff_out, "trajectory_log_probs", None)
        if traj_lp is not None:
            per_log_probs.append(traj_lp)

    traj_latents: Optional[torch.Tensor] = torch.cat(per_latents, dim=0) if per_latents else None
    traj_log_probs: Optional[torch.Tensor] = torch.cat(per_log_probs, dim=0) if per_log_probs else None
    head = diff_outputs[0]
    seg_sigmas = getattr(head, "trajectory_timesteps", None)
    # Engine→worker→response σ contract: the engine pinned ``req.sigmas``
    # before dispatch, the worker consumed it via ``set_timesteps(sigmas=...)``
    # and echoed the same values back. Assert equality so a broken wire
    # surfaces immediately rather than silently de-syncing replay.
    verify_engine_used_sigmas(
        seg_sigmas,
        expected=expected_sigmas,
        engine_name="vllm-omni",
    )
    head_custom = getattr(head, "custom_output", None) or {}
    sde_step_indices_raw = head_custom.get("sde_step_indices")

    indices: Optional[torch.Tensor] = None
    sde_indices: Optional[torch.Tensor] = None
    # K == 0 happens when the algorithm requested zero SDE steps (NFT /
    # forward-process). Treat identically to "no log_probs at all":
    # clean-latents segment with no sde_logp / sde_indices.
    K = int(traj_log_probs.shape[1]) if traj_log_probs is not None else 0
    if K > 0:
        T_plus_1 = int(traj_latents.shape[1]) if traj_latents is not None else K + 1
        indices = torch.arange(T_plus_1, dtype=torch.long)
        if sde_step_indices_raw is not None:
            sde_indices = torch.as_tensor([int(i) for i in sde_step_indices_raw], dtype=torch.long)
            if int(sde_indices.numel()) != K:
                raise RuntimeError(
                    f"build_image_segment: scheduler reported "
                    f"sde_step_indices of length {int(sde_indices.numel())} "
                    f"but trajectory_log_probs has {K} entries — pipeline "
                    f"subclass produced inconsistent outputs."
                )
        else:
            # Legacy fallback when the pipeline subclass didn't echo the real
            # step IDs. Only safe when K == T (dense). For sparse K < T this
            # misaligns replay; raise rather than silently mis-label.
            T = int(traj_latents.shape[1]) - 1 if traj_latents is not None else K
            if K != T:
                raise RuntimeError(
                    "build_image_segment: trajectory log_probs has K="
                    f"{K} but latents has T={T} steps and worker did not "
                    "expose ``custom_output['sde_step_indices']``. Update "
                    "the pipeline subclass to echo last_sde_step_indices."
                )
            sde_indices = torch.arange(K, dtype=torch.long)
    elif traj_latents is not None:
        # Forward-process case (NFT): still emit ``indices`` so the trainer's
        # clean-latents branch can look up the final latent, but drop the
        # ``[B, 0]`` log-probs placeholder (it confuses downstream replay).
        traj_log_probs = None
        T_plus_1 = int(traj_latents.shape[1])
        indices = torch.arange(T_plus_1, dtype=torch.long)
        sde_indices = None

    return make_image_segment(
        latents=traj_latents,
        sigmas=seg_sigmas,
        indices=indices,
        sde_logp=traj_log_probs,
        sde_indices=sde_indices,
    )


def decoded_text_from_ar(per_request: Sequence[Sequence[Any]]) -> Texts:
    """Extract the per-request AR text from Stage 0 outputs."""
    texts: List[Text] = []
    for outputs in per_request:
        ar = pick_stage_output(outputs, final_output_type="text", stage_id=0)
        text_str = ""
        if ar is not None:
            ro = getattr(ar, "request_output", None)
            if ro is not None:
                completions = getattr(ro, "outputs", None) or []
                if completions:
                    text_str = getattr(completions[0], "text", "") or ""
        texts.append(Text(text=text_str))
    return Texts.from_list(texts)


# --------------------------------------------------------------------------- #
# AR segment capture (Stage 0 tokens + per-token log-probs)
# --------------------------------------------------------------------------- #


def _flatten_logprobs(logprobs: Any, fallback_len: int) -> Optional[torch.Tensor]:
    """Best-effort vLLM-logprob → ``[T]`` float tensor.

    vLLM ``CompletionOutput.logprobs`` is ``list[dict[token_id, Logprob]]`` of
    length T; pick the sampled-token entry per step. Returns ``None`` when
    missing/empty (matches the AR config's ``detokenize=False`` path).
    """
    if logprobs is None:
        return None
    if not isinstance(logprobs, Sequence) or len(logprobs) == 0:
        return None
    values: List[float] = []
    for step in logprobs:
        if step is None:
            values.append(0.0)
            continue
        # vLLM Logprob objects expose ``.logprob``; dicts usually have one
        # entry whose value is the Logprob object. Try both shapes.
        if hasattr(step, "logprob"):
            values.append(float(step.logprob))
            continue
        if isinstance(step, dict) and step:
            entry = next(iter(step.values()))
            values.append(float(getattr(entry, "logprob", entry)))
            continue
        values.append(0.0)
    if not values:
        return None
    if len(values) != fallback_len and fallback_len > 0:
        # Truncate or pad-with-zeros so the downstream stack stays well-shaped
        # (pad is rare — early-stop / retokenize mismatches).
        if len(values) > fallback_len:
            values = values[:fallback_len]
        else:
            values.extend([0.0] * (fallback_len - len(values)))
    return torch.tensor(values, dtype=torch.float32)


def _extract_completion(out: Any) -> Tuple[List[int], Optional[torch.Tensor]]:
    """Pull ``(token_ids, per_token_logp)`` out of a Stage-0 result."""
    request_output = getattr(out, "request_output", None)
    if request_output is None:
        return [], None
    completions = getattr(request_output, "outputs", None) or []
    if not completions:
        return [], None
    completion = completions[0]
    tokens = list(getattr(completion, "token_ids", []) or [])
    logp = _flatten_logprobs(getattr(completion, "logprobs", None), fallback_len=len(tokens))
    return tokens, logp


def build_ar_segment(per_request: Sequence[Sequence[Any]]) -> Optional[Any]:
    """Build a ``TextSegment`` from the AR Stage-0 outputs of one batch.

    Picks each request's Stage-0 entry, gathers tokens + per-token logprobs,
    and hands per-sample lists to ``TextSegment.pack`` (which derives the
    framework-managed ``cu_seqlens``). ``log_probs`` is all-or-nothing across
    rows: if any token-bearing row is missing logp, the whole field is
    dropped rather than emitting a ragged shape. Returns ``None`` when no
    Stage-0 output is found in any row.
    """
    from unirl.types.segments.text import TextSegment

    rows_tokens: List[List[int]] = []
    rows_logps: List[Optional[torch.Tensor]] = []
    found_any_stage0 = False

    for outputs in per_request:
        stage0 = None
        for out in outputs:
            if getattr(out, "stage_id", None) == 0:
                stage0 = out
                break
        if stage0 is None:
            rows_tokens.append([])
            rows_logps.append(None)
            continue
        toks, logp = _extract_completion(stage0)
        if toks:
            found_any_stage0 = True
        rows_tokens.append(toks)
        rows_logps.append(logp)

    if not found_any_stage0:
        return None

    tokens_list: List[torch.Tensor] = [torch.tensor(toks, dtype=torch.long) for toks in rows_tokens]
    have_logp = all(lp is not None for toks, lp in zip(rows_tokens, rows_logps) if toks)
    log_probs_list: Optional[List[torch.Tensor]] = None
    if have_logp:
        log_probs_list = [lp if lp is not None else torch.zeros(0, dtype=torch.float32) for lp in rows_logps]

    return TextSegment.pack(
        tokens=tokens_list,
        log_probs=log_probs_list,
    )


# --------------------------------------------------------------------------- #
# Track assembly — the shared tail of every shape's ``build_response``
# --------------------------------------------------------------------------- #


def assemble_tracks(
    req: RolloutReq,
    *,
    segments_for_track: Dict[str, Segment],
    decoded_for_track: Dict[str, Optional[Any]],
    conditions: Dict[str, Condition],
) -> RolloutResp:
    """Pack per-track segments/decoded/conditions into a ``RolloutResp``.

    Tracks are one per ``segments_for_track`` key, each carrying its own
    decoded value (or ``None``). ``conditions`` were resp-wide in the legacy
    shape; keep that behavior by replicating onto every track (the legacy
    single-image-track replay is the only consumer today).

    HI3 think_recaption lineage: when both an "image" and an "ar" segment are
    present the image is generated from the AR output 1-to-1, so
    ``image.parent_track = "ar"`` with parent_ids aligned to ``ar.sample_ids``;
    every other track is a root (``parent_ids = req.group_ids``).
    """
    sample_ids = list(req.sample_ids)
    parent_ids = list(req.group_ids)
    has_ar = "ar" in segments_for_track
    tracks: Dict[str, RolloutTrack] = {}
    for track_name, segment in segments_for_track.items():
        if track_name == "image" and has_ar:
            parent: Optional[str] = "ar"
            track_parent_ids = list(sample_ids)
        else:
            parent = None
            track_parent_ids = list(parent_ids)
        tracks[track_name] = RolloutTrack(
            sample_ids=list(sample_ids),
            parent_ids=track_parent_ids,
            parent_track=parent,
            conditions=dict(conditions),
            segment=segment,
            decoded=decoded_for_track.get(track_name),
        )
    return RolloutResp(tracks=tracks)


__all__ = [
    "assemble_tracks",
    "build_ar_segment",
    "build_image_segment",
    "collect_dit_outputs",
    "decoded_text_from_ar",
    "grouped_pils_to_videos",
    "pick_stage_output",
    "pils_to_images",
    "seed_from_sample_id",
]
