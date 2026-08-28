"""Response-side segment and decoded-output mechanics."""

from __future__ import annotations

import hashlib
from typing import Any, List, Optional, Sequence, Tuple

import torch

from unirl.rollout.engine.sigma_verify import verify_engine_used_sigmas
from unirl.types.primitives import Image, Images, Text, Texts, Video, Videos
from unirl.types.segments.latent import make_image_segment


def seed_from_sample_id(sample_id: str) -> int:
    """Deterministic 31-bit diffusion seed for one image, keyed by sample_id."""
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def pils_to_images(pil_images: Sequence[Any]) -> Images:
    """``[PIL.Image, ...]`` → packed ``Images`` with float32 CHW samples in ``[0, 1]``."""
    if not pil_images:
        raise ValueError("pils_to_images: empty image list")
    from torchvision.transforms.functional import pil_to_tensor

    items: List[Image] = []
    for pil in pil_images:
        t = pil_to_tensor(pil).to(torch.float32) / 255.0
        items.append(Image(pixels=t))
    return Images.from_list(items)


def grouped_pils_to_videos(pil_frames_per_prompt: Sequence[Sequence[Any]]) -> Videos:
    """Group per-prompt PIL frame lists into ``Videos``."""
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
    """Find the result with the requested ``final_output_type``."""
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
    """Recover a video sample's PIL frames from the decoded-video tensor the RL"""
    co = getattr(diff_out, "custom_output", None) or {}
    vid = co.get("rl_decoded_video")
    if vid is None or not torch.is_tensor(vid):
        return []
    global _VIDEO_PROCESSOR
    if _VIDEO_PROCESSOR is None:
        from diffusers.video_processor import VideoProcessor

        _VIDEO_PROCESSOR = VideoProcessor(vae_scale_factor=16)
    frames = _VIDEO_PROCESSOR.postprocess_video(vid, output_type="pil")
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
    """Pick each request's DiT output + its PIL payload(s)."""
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
    """Build ``LatentSegment`` from the DiT stage's per-request outputs."""
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
    verify_engine_used_sigmas(
        seg_sigmas,
        expected=expected_sigmas,
        engine_name="vllm-omni",
    )
    head_custom = getattr(head, "custom_output", None) or {}
    sde_step_indices_raw = head_custom.get("sde_step_indices")
    trajectory_indices_raw = head_custom.get("trajectory_indices")

    indices: Optional[torch.Tensor] = None
    sde_indices: Optional[torch.Tensor] = None
    if traj_latents is not None:
        stored_steps = int(traj_latents.shape[1])
        if trajectory_indices_raw is None:
            indices = torch.arange(stored_steps, dtype=torch.long)
        else:
            indices = torch.as_tensor([int(i) for i in trajectory_indices_raw], dtype=torch.long)
            if int(indices.numel()) != stored_steps:
                raise RuntimeError(
                    "build_image_segment: trajectory_indices has "
                    f"{int(indices.numel())} entries but trajectory_latents stores {stored_steps} steps."
                )
            if indices.numel() and (
                not bool(torch.all(indices[1:] > indices[:-1]))
                or int(indices[0]) < 0
                or (torch.is_tensor(seg_sigmas) and int(indices[-1]) >= int(seg_sigmas.numel()))
            ):
                raise RuntimeError(
                    "build_image_segment: trajectory_indices must be strictly increasing and within the sigma schedule; "
                    f"got {indices.tolist()}."
                )
            for output in diff_outputs[1:]:
                other = (getattr(output, "custom_output", None) or {}).get("trajectory_indices")
                if other is None or [int(i) for i in other] != indices.tolist():
                    raise RuntimeError("build_image_segment: trajectory_indices differ across engine requests.")

    K = int(traj_log_probs.shape[1]) if traj_log_probs is not None else 0
    if K > 0:
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
        traj_log_probs = None
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


def _flatten_logprobs(logprobs: Any, fallback_len: int) -> Optional[torch.Tensor]:
    """Best-effort vLLM-logprob → ``[T]`` float tensor."""
    if logprobs is None:
        return None
    if not isinstance(logprobs, Sequence) or len(logprobs) == 0:
        return None
    values: List[float] = []
    for step in logprobs:
        if step is None:
            values.append(0.0)
            continue
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
    """Build a ``TextSegment`` from the AR Stage-0 outputs of one batch."""
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
        rollout_log_probs=log_probs_list,
    )


__all__ = [
    "build_ar_segment",
    "build_image_segment",
    "collect_dit_outputs",
    "decoded_text_from_ar",
    "grouped_pils_to_videos",
    "pick_stage_output",
    "pils_to_images",
    "seed_from_sample_id",
]
