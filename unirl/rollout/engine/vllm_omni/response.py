"""``Omni.generate`` outputs → ``RolloutResp`` translator.

Single modality-branched function ``_to_rollout_resp(req, per_request_outputs,
*, modality)``. Caller groups ``Omni.generate``'s flat output list into
per-request lists; this function picks the per-stage outputs (Stage 0 AR,
Stage 1 DiT for image modalities) and packs into ``RolloutResp``.

Produces:

- ``resp.tracks["image"].decoded`` (image modalities) — :class:`Images` with
  pixels ``[B, C, H, W]`` in [0, 1] from PIL outputs of the DiT stage.
- ``resp.tracks["ar"].decoded`` (modalities that run AR) — :class:`Texts` from
  ``request_output.outputs[0].text``.
- ``resp.tracks["image"].segment`` (image modalities) — ``LatentSegment`` from
  the DiT stage's trajectory tensors.
- ``resp.tracks["ar"].segment`` (all modalities) — ``TextSegment`` packed by
  ``hi3.ar_capture.extract_ar_segment``.
- ``resp.tracks["image"].conditions["fused"]`` (image modalities) —
  ``HunyuanImage3FusedMultimodalCondition`` built from per-request
  ``OmniRequestOutput.custom_output["fused_mm_capture"]`` (written by
  :class:`RLHunyuanImage3Pipeline` on the first per-request
  ``prepare_inputs_for_generation`` call inside the worker; vllm-omni
  routes ``DiffusionOutput.custom_output`` into ``OmniRequestOutput.custom_output``
  on the IPC boundary, see upstream ``diffusion/data.py:841`` and
  ``stage_diffusion_proc.py:182``). t2i scope for the first cut —
  surfaces ``input_ids`` / ``attention_mask`` / ``position_ids`` /
  ``rope_cache`` / ``gen_image_mask`` / ``gen_timestep_scatter_index``;
  the it2i ``cond_*`` fields stay unpopulated. Missing capture is a
  fatal misconfiguration (pipeline subclass not installed, hook
  regression, ...) — ``_to_rollout_resp`` raises at the rollout boundary
  rather than silently emitting empty conditions that would crash the
  trainer-side replay much later.
- ``resp.tracks["ar"].conditions = {}`` (AR-only modalities) — no diffusion replay
  in scope.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import torch

from unirl.models.hunyuan_image3.conditions import (
    HunyuanImage3FusedMultimodalCondition,
)
from unirl.rollout.engine.sigma_verify import verify_engine_used_sigmas
from unirl.rollout.engine.vllm_omni.hi3.ar_capture import extract_ar_segment
from unirl.types.conditions import Condition
from unirl.types.conditions.text import TextEmbedCondition
from unirl.types.primitives import Image, Images, Text, Texts, Video, Videos
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp, RolloutTrack
from unirl.types.segments import Segment
from unirl.types.segments.latent import make_image_segment

logger = logging.getLogger(__name__)


def group_by_request(
    flat_outputs: Sequence[Any],
    n: int,
) -> List[List[Any]]:
    """Group ``Omni.generate``'s flat output list into per-request lists.

    ``Omni._run_generation`` builds ``request_ids = [f"{i}_{uuid4()}"
    for i in range(B)]`` (one per prompt). After our YAML flips Stage 0
    to ``final_output: true`` for it2i, each request contributes one
    output per final stage (2 for t2i/it2i, 1 for i2t/t2t).

    The mapping from output back to request index is by the ``i_`` prefix
    on ``request_id``. When the orchestrator's ordering invariant
    changes upstream, the count won't match the expected total and we
    raise — better than silently misaligning.
    """
    grouped: List[List[Any]] = [[] for _ in range(n)]
    for out in flat_outputs:
        rid = getattr(out, "request_id", "") or ""
        if "_" in rid:
            idx_part = rid.split("_", 1)[0]
            try:
                idx = int(idx_part)
            except ValueError:
                continue
            if 0 <= idx < n:
                grouped[idx].append(out)
    return grouped


def _pil_list_to_images(pil_images: Sequence[Any]) -> Images:
    """``[PIL.Image, …] → Images`` (float32 ``[B, C, H, W]`` in [0, 1])."""
    if not pil_images:
        raise ValueError("_pil_list_to_images: empty image list")
    from torchvision.transforms.functional import pil_to_tensor

    items: List[Image] = []
    for pil in pil_images:
        # uint8 [C, H, W] / 255 → float32 [0, 1] per Image(pixels=...) contract.
        t = pil_to_tensor(pil).to(torch.float32) / 255.0
        items.append(Image(pixels=t))
    return Images.from_list(items)


def _grouped_pils_to_videos(
    pil_frames_per_prompt: Sequence[Sequence[Any]],
) -> Videos:
    """Group per-prompt PIL frame lists into ``Videos``.

    Upstream HV1.5 ``post_process_func`` returns ``List[List[PIL.Image]]``
    (one frame list per prompt). We reassemble each as a
    ``Video(frames=[T, C, H, W])`` so the reward layer's
    ``RewardRequest.videos`` (which calls ``v.frames.permute(1, 0, 2, 3)``)
    gets the right shape.
    """
    if not pil_frames_per_prompt:
        raise ValueError("_grouped_pils_to_videos: empty per-prompt frame lists")
    from torchvision.transforms.functional import pil_to_tensor

    items: List[Video] = []
    for frames in pil_frames_per_prompt:
        if not frames:
            raise ValueError("_grouped_pils_to_videos: prompt has zero frames")
        # Per-frame uint8 [C, H, W] / 255 → float32 [0, 1]; stack T along dim=0.
        frame_tensors = [pil_to_tensor(f).to(torch.float32) / 255.0 for f in frames]
        items.append(Video(frames=torch.stack(frame_tensors, dim=0)))
    return Videos.from_list(items)


def _pick_stage_output(
    outputs: Sequence[Any],
    *,
    final_output_type: str,
    stage_id: Optional[int] = None,
) -> Optional[Any]:
    """Find the ``OmniRequestOutput`` with the requested ``final_output_type``.

    Falls back to ``stage_id`` match if provided. Returns ``None`` if
    neither matches — callers decide whether that's an error.
    """
    for out in outputs:
        if getattr(out, "final_output_type", None) == final_output_type:
            return out
    if stage_id is not None:
        for out in outputs:
            if getattr(out, "stage_id", None) == stage_id:
                return out
    return None


def _build_image_segment(
    diff_outputs: Sequence[Any],
    *,
    expected_sigmas: Optional[torch.Tensor] = None,
) -> Any:
    """Build ``LatentSegment`` for the DiT stage's outputs.

    Each per-prompt ``OmniRequestOutput`` carries its own
    ``trajectory_latents`` / ``trajectory_log_probs`` for its own request.
    With ``runtime.max_inflight=1`` the diffusion engine processes one
    request at a time, so the per-prompt tensors are NOT shared refs to
    a full-batch tensor — each is shape ``[1, T+1, ...]`` / ``[1, K]``
    where ``K`` is the number of SDE-gated steps (``K`` ranges from
    ``0`` for forward-process / DiffusionNFT runs up to ``T`` for fully-SDE runs).
    We concatenate across all outputs to recover ``[B, T+1, ...]`` /
    ``[B, K]`` where ``B = len(diff_outputs)``.

    ``sigmas`` / ``indices`` / ``sde_indices`` are sample-shared (the SDE
    schedule and stored-slot indexing are identical across all samples in
    the chunk), so we read them off the first output without concat.

    Output shapes:
    - ``latents`` from ``trajectory_latents`` — ``[B, T+1, ...]`` after
      concat across diff_outputs. ALWAYS dense (every step recorded so
      replay has ``x_t`` at every slot regardless of which steps ran SDE).
    - ``sigmas`` from ``trajectory_timesteps`` — the field name reads
      "timesteps" but our ``RL*Pipeline.forward`` overwrites its contents
      with the true [0, 1] sigma schedule (1D ``[T+1]``) drained from
      ``FlowMatchSDEDiscreteScheduler``. Sample-shared.
    - ``sde_logp`` from ``trajectory_log_probs`` — ``[B, K]`` after concat
      (K = number of SDE-gated steps; can be < T when the algorithm picks
      a sparse subset via ``stage_params["diffusion"]["sde_indices"]``).
    - ``indices`` — dense ``arange(T+1)``: latent-storage slots.
    - ``sde_indices`` — sparse step IDs ``[K]`` read off the worker's
      ``custom_output["sde_step_indices"]`` (echoed there by the pipeline
      subclass; falls back to ``arange(K)`` only if the capture is
      missing, e.g. older pipeline build).
    """
    if not diff_outputs:
        raise ValueError("_build_image_segment: empty diff_outputs")

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
    # Sigmas / step-index axes are sample-shared — the SDE schedule and
    # stored-slot count don't vary per sample in a single chunk.
    head = diff_outputs[0]
    seg_sigmas = getattr(head, "trajectory_timesteps", None)
    # Engine→worker→response σ contract: the engine pinned ``req.sigmas``
    # before dispatch, the worker should have consumed it via
    # ``set_timesteps(sigmas=...)`` and echoed the same values back via
    # ``trajectory_timesteps``. Assert equality here so a broken wire
    # surfaces immediately rather than silently de-syncing training-side
    # replay from rollout. Caller passes ``expected_sigmas=None`` to skip
    # the check (legacy entry points that don't run ensure_req_sigmas).
    verify_engine_used_sigmas(
        seg_sigmas,
        expected=expected_sigmas,
        engine_name="vllm-omni",
    )
    head_custom = getattr(head, "custom_output", None) or {}
    sde_step_indices_raw = head_custom.get("sde_step_indices")

    indices: Optional[torch.Tensor] = None
    sde_indices: Optional[torch.Tensor] = None
    # K == 0 happens when the algorithm requested zero SDE steps (DiffusionNFT /
    # forward-process). Treat the empty case identically to "no log_probs
    # at all": clean-latents segment with no sde_logp / sde_indices.
    # Trainer-side `to_training_batch` already branches on
    # ``segment.sde_indices is None`` to take the clean-latents path.
    K = int(traj_log_probs.shape[1]) if traj_log_probs is not None else 0
    if K > 0:
        # ``trajectory_log_probs`` is ``[B, K]`` (one entry per SDE-gated
        # transition; ``K`` equals ``T`` only when every step was SDE).
        # ``trajectory_latents`` is ``[B, T+1, ...]`` (position-0 + T post-step,
        # ALWAYS dense — scheduler captures latent regardless of SDE/ODE).
        # ``indices`` maps step_idx -> storage slot for
        # ``LatentSegment.latents_at``, so it must enumerate every stored
        # slot (0..T). ``sde_indices`` enumerates the SDE-gated step ids
        # only (length K).
        T_plus_1 = int(traj_latents.shape[1]) if traj_latents is not None else K + 1
        indices = torch.arange(T_plus_1, dtype=torch.long)
        if sde_step_indices_raw is not None:
            sde_indices = torch.as_tensor([int(i) for i in sde_step_indices_raw], dtype=torch.long)
            if int(sde_indices.numel()) != K:
                raise RuntimeError(
                    f"_build_image_segment: scheduler reported "
                    f"sde_step_indices of length {int(sde_indices.numel())} "
                    f"but trajectory_log_probs has {K} entries — pipeline "
                    f"subclass produced inconsistent outputs."
                )
        else:
            # Legacy fallback when the pipeline subclass didn't echo the
            # real step IDs (e.g. older HI3 path being upgraded). Only safe
            # when K == T (dense case). For sparse K < T this misaligns
            # replay; raise rather than silently mis-label.
            T = int(traj_latents.shape[1]) - 1 if traj_latents is not None else K
            if K != T:
                raise RuntimeError(
                    "_build_image_segment: trajectory log_probs has K="
                    f"{K} but latents has T={T} steps and worker did not "
                    "expose ``custom_output['sde_step_indices']``. Update "
                    "the pipeline subclass to echo last_sde_step_indices."
                )
            sde_indices = torch.arange(K, dtype=torch.long)
    elif traj_latents is not None:
        # Forward-process case (DiffusionNFT): still emit ``indices`` so the
        # clean-latents branch on the trainer side can look up the final
        # latent, but leave ``sde_indices`` / ``sde_logp`` as None (drop
        # the ``[B, 0]`` placeholder — it confuses downstream stage
        # ``replay`` paths that read ``sde_logp.shape[1]``).
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


def _decoded_text_from_ar(per_request_outputs: Sequence[Sequence[Any]]) -> Texts:
    """Extract the per-request AR text from Stage 0 outputs."""
    texts: List[Text] = []
    for outputs in per_request_outputs:
        ar = _pick_stage_output(outputs, final_output_type="text", stage_id=0)
        text_str = ""
        if ar is not None:
            ro = getattr(ar, "request_output", None)
            if ro is not None:
                completions = getattr(ro, "outputs", None) or []
                if completions:
                    text_str = getattr(completions[0], "text", "") or ""
        texts.append(Text(text=text_str))
    return Texts.from_list(texts)


def _build_fused_mm_condition(
    diff_outputs: Sequence[Any],
) -> Optional[HunyuanImage3FusedMultimodalCondition]:
    """Concat per-request ``fused_mm_capture`` dicts into one fused condition.

    Reads the capture off ``OmniRequestOutput.custom_output["fused_mm_capture"]``
    — the dataclass-routed dict ``RLHunyuanImage3Pipeline`` writes after
    intercepting ``prepare_inputs_for_generation``. Plain runtime attrs on
    ``DiffusionOutput`` don't survive vllm-omni's IPC boundary.

    Returns ``None`` when any diff output is missing the capture (e.g. the
    worker side hasn't installed :class:`RLHunyuanImage3Pipeline`'s hook,
    or upstream's ``prepare_inputs_for_generation`` was bypassed). Callers
    treat ``None`` as "no conditions surfaced" and emit empty per-track ``conditions``,
    preserving the pre-patch contract.

    For think_recaption mode, different prompts produce different AR output
    lengths → different ``L`` per capture. This function right-pads shorter
    sequences to ``max_L`` (pad_token_id=0 for input_ids, False for masks,
    0.0 for rope_cache) so ``torch.cat`` on dim 0 works.
    """
    if not diff_outputs:
        return None
    captures = [(getattr(d, "custom_output", None) or {}).get("fused_mm_capture") for d in diff_outputs]
    if any(c is None for c in captures):
        return None

    sequence_lengths = [int(c["input_ids"].shape[-1]) for c in captures]
    max_L = max(sequence_lengths)

    def _pad_to(t: Any, target_L: int, dim: int = -1, value: Any = 0) -> Any:
        if t is None or not isinstance(t, torch.Tensor):
            return t
        cur_L = t.shape[dim]
        if cur_L >= target_L:
            return t
        pad_size = target_L - cur_L
        ndim = t.ndim
        pad_spec = [0] * (2 * ndim)
        actual_dim = dim if dim >= 0 else ndim + dim
        pad_idx = (ndim - 1 - actual_dim) * 2
        pad_spec[pad_idx + 1] = pad_size
        return torch.nn.functional.pad(t, pad_spec, value=value)

    def _pad_attn_mask(mask: Any, target_L: int) -> Any:
        """Pad attention_mask [N, 1, L, L] → [N, 1, target_L, target_L]."""
        if mask is None or not isinstance(mask, torch.Tensor):
            return mask
        if mask.shape[-1] >= target_L:
            return mask
        N, H, L, _ = mask.shape
        padded = torch.zeros(N, H, target_L, target_L, dtype=mask.dtype, device=mask.device)
        padded[:, :, :L, :L] = mask
        return padded

    padded_captures = []
    for c, L_i in zip(captures, sequence_lengths):
        if L_i == max_L:
            padded_captures.append(c)
        else:
            padded_captures.append(
                {
                    "input_ids": _pad_to(c["input_ids"], max_L, dim=-1, value=0),
                    "attention_mask": _pad_attn_mask(c.get("attention_mask"), max_L),
                    "position_ids": _pad_to(c.get("position_ids"), max_L, dim=-1, value=0),
                    "gen_image_mask": _pad_to(c.get("gen_image_mask"), max_L, dim=-1, value=False),
                    "gen_timestep_scatter_index": c.get("gen_timestep_scatter_index"),
                    "rope_cache": (
                        (
                            _pad_to(c["rope_cache"][0], max_L, dim=-2, value=0.0),
                            _pad_to(c["rope_cache"][1], max_L, dim=-2, value=0.0),
                        )
                        if c.get("rope_cache") is not None and isinstance(c["rope_cache"], tuple)
                        else c.get("rope_cache")
                    ),
                }
            )

    fused_dict: Dict[str, Any] = {
        "input_ids": torch.cat([c["input_ids"] for c in padded_captures], dim=0),
        "attention_mask": torch.cat([c["attention_mask"] for c in padded_captures], dim=0),
        "position_ids": torch.cat([c["position_ids"] for c in padded_captures], dim=0),
        "gen_image_mask": torch.cat([c["gen_image_mask"] for c in padded_captures], dim=0),
        "gen_timestep_scatter_index": torch.cat([c["gen_timestep_scatter_index"] for c in padded_captures], dim=0),
    }
    cos_parts = [c["rope_cache"][0] for c in padded_captures]
    sin_parts = [c["rope_cache"][1] for c in padded_captures]
    fused_dict["rope_cache"] = (
        torch.cat(cos_parts, dim=0),
        torch.cat(sin_parts, dim=0),
    )

    # ``from_dict`` skips optional fields when absent; cond_* fields stay
    # ``None`` for t2i (out of scope for the it2i extension).
    return HunyuanImage3FusedMultimodalCondition.from_dict(fused_dict)


def _build_sd3_text_condition(
    diff_outputs: Sequence[Any],
) -> Optional[TextEmbedCondition]:
    """Concat per-request ``text_capture`` dicts into one TextEmbedCondition.

    Reads ``OmniRequestOutput.custom_output["text_capture"]`` — the
    dataclass-routed dict :class:`RLStableDiffusion3Pipeline` writes
    after intercepting ``encode_prompt``. Plain runtime attrs on
    ``DiffusionOutput`` don't survive vllm-omni's IPC boundary.

    Returns ``None`` when any diff output is missing the capture (e.g.
    the worker side hasn't installed
    :class:`RLStableDiffusion3Pipeline`'s hook). The training side
    requires this condition for :meth:`SD3DiffusionStage.replay`; an
    empty conditions dict will surface as ``SD3Conditions.from_dict``
    failure downstream (clear error rather than silent skip).

    For SD3, all per-request encodes share the same ``L`` (T5 padding
    to ``max_sequence_length=256`` is fixed), so a plain concat on dim 0
    suffices.
    """
    if not diff_outputs:
        return None
    captures = [(getattr(d, "custom_output", None) or {}).get("text_capture") for d in diff_outputs]
    if any(c is None for c in captures):
        return None

    return TextEmbedCondition(
        embeds=torch.cat([c["prompt_embeds"] for c in captures], dim=0),
        pooled=torch.cat([c["pooled_prompt_embeds"] for c in captures], dim=0),
        attn_mask=None,  # SD3 uses fixed-length T5 padding; no attn mask needed
    )


def _build_hv15_conditions(
    diff_outputs: Sequence[Any],
) -> Optional[Dict[str, Condition]]:
    """Unpack per-request HunyuanVideo-1.5 conditions from worker-side capture.

    Reads ``OmniRequestOutput.custom_output["text_capture"]`` — the flat dict
    :class:`RLHunyuanVideo15Pipeline` writes after intercepting
    ``encode_prompt``. The pipeline captures 8 tensors from the dual text
    encoder (Qwen2.5-VL MLLM + ByT5 glyph):

    - ``prompt_embeds`` / ``prompt_embeds_mask`` → text_mllm
    - ``prompt_embeds_2`` / ``prompt_embeds_mask_2`` → text_glyph
    - ``negative_prompt_embeds`` / ``negative_prompt_embeds_mask`` → negative_text_mllm
    - ``negative_prompt_embeds_2`` / ``negative_prompt_embeds_mask_2`` → negative_text_glyph

    Returns the conditions *dict* (keys aligned with
    ``HunyuanVideo15Conditions.from_dict``), NOT the typed wrapper —
    ``RolloutTrack.conditions`` is ``Dict[str, Condition]`` and the trainer
    runs ``HunyuanVideo15Conditions.from_dict(track.conditions)`` itself.

    Returns ``None`` when any diff output is missing the capture (signals the
    worker-side hook didn't fire — surfaces as an explicit error at the call
    site rather than a far-away ``from_dict({})`` failure).
    """
    if not diff_outputs:
        return None

    captures = [(getattr(d, "custom_output", None) or {}).get("text_capture") for d in diff_outputs]
    if any(c is None for c in captures):
        return None

    def _cat_field(field_name: str) -> Optional[torch.Tensor]:
        tensors = [c[field_name] for c in captures if c.get(field_name) is not None]
        if not tensors:
            return None
        return torch.cat(tensors, dim=0)

    prompt_embeds = _cat_field("prompt_embeds")
    prompt_embeds_mask = _cat_field("prompt_embeds_mask")
    prompt_embeds_2 = _cat_field("prompt_embeds_2")
    prompt_embeds_mask_2 = _cat_field("prompt_embeds_mask_2")
    negative_prompt_embeds = _cat_field("negative_prompt_embeds")
    negative_prompt_embeds_mask = _cat_field("negative_prompt_embeds_mask")
    negative_prompt_embeds_2 = _cat_field("negative_prompt_embeds_2")
    negative_prompt_embeds_mask_2 = _cat_field("negative_prompt_embeds_mask_2")

    cond_dict: Dict[str, Condition] = {}
    if prompt_embeds is not None:
        cond_dict["text_mllm"] = TextEmbedCondition(embeds=prompt_embeds, pooled=None, attn_mask=prompt_embeds_mask)
    if prompt_embeds_2 is not None:
        cond_dict["text_glyph"] = TextEmbedCondition(
            embeds=prompt_embeds_2, pooled=None, attn_mask=prompt_embeds_mask_2
        )
    if negative_prompt_embeds is not None:
        cond_dict["negative_text_mllm"] = TextEmbedCondition(
            embeds=negative_prompt_embeds, pooled=None, attn_mask=negative_prompt_embeds_mask
        )
    if negative_prompt_embeds_2 is not None:
        cond_dict["negative_text_glyph"] = TextEmbedCondition(
            embeds=negative_prompt_embeds_2, pooled=None, attn_mask=negative_prompt_embeds_mask_2
        )

    if "text_mllm" not in cond_dict or "text_glyph" not in cond_dict:
        return None
    return cond_dict


def _build_ar_fused_condition(per_request_outputs: Sequence[Sequence[Any]]) -> Optional[Any]:
    """AR fused condition for GRPO replay: per-sample prompt token ids.

    Each AR request's Stage-0 ``OmniRequestOutput`` carries ``prompt_token_ids``
    (vLLM runs prompts per-request with no batch padding, so this is the
    sample's TRUE, un-padded prompt). Right-pad to ``[B, max_len]`` and carry
    each sample's true length in the dedicated 1D ``prompt_lengths`` [B] field
    (NOT ``attention_mask`` — that's typed 4D and its concat does a 4D unpack).
    The teacher-forced replay loops per-sample and slices
    ``input_ids[b, :prompt_lengths[b]]``, so the right-pad never leaks into
    attention / position / the prediction slice.

    Returns ``None`` if no Stage-0 output carries prompt tokens.
    """
    from unirl.models.hunyuan_image3.conditions import HunyuanImage3FusedMultimodalCondition

    rows: List[List[int]] = []
    for outputs in per_request_outputs:
        ids = None
        for out in outputs:
            if getattr(out, "stage_id", None) == 0:
                ids = getattr(out, "prompt_token_ids", None)
                break
        rows.append([int(t) for t in ids] if ids else [])

    if not any(rows):
        return None

    bsz = len(rows)
    max_len = max(len(r) for r in rows)
    input_ids = torch.zeros((bsz, max_len), dtype=torch.long)
    prompt_lengths = torch.zeros((bsz,), dtype=torch.long)
    for b, r in enumerate(rows):
        if r:
            input_ids[b, : len(r)] = torch.tensor(r, dtype=torch.long)
            prompt_lengths[b] = len(r)
    # Carry the per-sample TRUE prompt length in the dedicated ``prompt_lengths``
    # field (1D [B], plain-cat CONCAT) — NOT ``attention_mask`` (typed 4D
    # [B,1,L,L], whose concat does a 4D unpack and would break on a 2D mask).
    # ARStage.replay slices ``input_ids[b, :prompt_lengths[b]]`` per sample.
    return HunyuanImage3FusedMultimodalCondition(input_ids=input_ids, prompt_lengths=prompt_lengths)


def _to_rollout_resp(
    req: RolloutReq,
    per_request_outputs: Sequence[Sequence[Any]],
    *,
    modality: str,
) -> RolloutResp:
    """``Omni.generate`` per-request outputs → ``RolloutResp``."""
    if not per_request_outputs or not any(per_request_outputs):
        raise ValueError("_to_rollout_resp: empty per_request_outputs (Omni.generate returned nothing surfaceable).")

    # Per-track decoded slots (at most one value per track, by modality).
    decoded_image: Optional[Images] = None
    decoded_video: Optional[Videos] = None
    decoded_text: Optional[Texts] = None
    segments_for_track: Dict[str, Segment] = {}
    conditions: Dict[str, Condition] = {}

    if modality in ("t2i", "it2i", "sd35_t2i", "t2v", "dit_recaption"):
        # Per-request DiT (image/video) output. For HI3 (t2i/it2i) it's Stage 1;
        # for SD3.5 (sd35_t2i), HV1.5 (t2v) and the standalone HI3 DiT
        # (dit_recaption) the diffusion stage is the only stage so stage_id=0.
        # Either way, ``_pick_stage_output`` matches by ``final_output_type``
        # first and falls back to stage_id.
        dit_stage_id = 0 if modality in ("sd35_t2i", "t2v", "dit_recaption") else 1
        final_output_type = "video" if modality == "t2v" else "image"
        # Track key matches training.tracks.<name>: video models use "video".
        diffusion_track_key = "video" if modality == "t2v" else "image"
        diff_outputs: List[Any] = []
        # t2v needs per-prompt frame groupings to pack into Videos; image
        # modalities use a flat PIL list.
        pil_frames_per_prompt: List[List[Any]] = []
        pil_images: List[Any] = []
        for outputs in per_request_outputs:
            diff_out = _pick_stage_output(
                outputs,
                final_output_type=final_output_type,
                stage_id=dit_stage_id,
            )
            if diff_out is None:
                raise RuntimeError(
                    f"_to_rollout_resp: no {final_output_type} output for request "
                    f"(modality={modality}); did the DiT stage fail?"
                )
            diff_outputs.append(diff_out)
            imgs = getattr(diff_out, "images", None) or []
            pil_frames_per_prompt.append(list(imgs))
            pil_images.extend(imgs)

        if not pil_images:
            raise RuntimeError(
                "_to_rollout_resp: DiT outputs carry no PIL images; "
                "check pipeline forward populated DiffusionOutput.output."
            )
        if modality == "t2v":
            decoded_video = _grouped_pils_to_videos(pil_frames_per_prompt)
        else:
            decoded_image = _pil_list_to_images(pil_images)
        segments_for_track[diffusion_track_key] = _build_image_segment(
            diff_outputs,
            expected_sigmas=req.sigmas,
        )

        # Surface conditions captured worker-side. Modality-specific:
        # HI3 (t2i/it2i) captures fused MM tensors via
        # ``prepare_inputs_for_generation``; SD3 (sd35_t2i) captures text
        # embeds via ``encode_prompt``. Both are *required* for the training
        # side's ``replay`` step — silently returning empty conditions makes
        # the trainer crash with ``KeyError``/``from_dict({})`` errors far
        # from the root cause. Fail fast at the rollout→trainer boundary
        # instead so the pipeline hook regression is visible immediately.
        if modality == "sd35_t2i":
            text_cond = _build_sd3_text_condition(diff_outputs)
            if text_cond is None:
                raise RuntimeError(
                    "_to_rollout_resp: SD3 rollout returned no 'text_capture' on "
                    "DiffusionOutput.custom_output. Check that "
                    "RLStableDiffusion3Pipeline._install_encode_prompt_hook ran "
                    "in every DiT worker — the subclass swap may not have taken "
                    "effect (verify custom_pipeline_args.pipeline_class in the "
                    "stage YAML)."
                )
            conditions["text"] = text_cond
        elif modality == "t2v":
            hv_conds = _build_hv15_conditions(diff_outputs)
            if hv_conds is None:
                raise RuntimeError(
                    "_to_rollout_resp: HV1.5 t2v rollout returned no 'text_capture' "
                    "on DiffusionOutput.custom_output (or it lacked the dual-stream "
                    "text_mllm/text_glyph embeds). Check that "
                    "RLHunyuanVideo15Pipeline's encode_prompt hook ran in every DiT "
                    "worker — verify custom_pipeline_args.pipeline_class in the stage "
                    "YAML."
                )
            conditions.update(hv_conds)
        else:
            fused_cond = _build_fused_mm_condition(diff_outputs)
            if fused_cond is None:
                raise RuntimeError(
                    f"_to_rollout_resp: HI3 rollout (modality={modality!r}) "
                    "returned no 'fused_mm_capture' on DiffusionOutput.custom_output. "
                    "Check that RLHunyuanImage3Pipeline.prepare_inputs_for_generation "
                    "hook ran in every DiT worker — the subclass swap may not have "
                    "taken effect (verify custom_pipeline_args.pipeline_class in "
                    "the stage YAML)."
                )
            conditions["fused"] = fused_cond
    elif modality in ("i2t", "t2t", "ar_recaption"):
        # AR-only stages. ``ar_recaption`` (two-engine trainer) runs
        # is_comprehension:false so the decoded text is the think/recaption
        # the DiT engine later consumes as cot_text; ar_capture surfaces the
        # token+logp TextSegment below for GRPO replay.
        decoded_text = _decoded_text_from_ar(per_request_outputs)
        if modality == "ar_recaption":
            # GRPO.replay teacher-forces over prompt+response; it needs the
            # prompt token ids (conditions['fused'].input_ids). vLLM processes
            # each request's prompt independently (no batch padding), so the
            # output's prompt_token_ids is the sample's true, un-padded prompt.
            ar_fused = _build_ar_fused_condition(per_request_outputs)
            if ar_fused is not None:
                conditions["fused"] = ar_fused
    else:
        raise ValueError(f"_to_rollout_resp: unknown modality {modality!r}")

    # Surface AR-generated text for all modalities that run AR (Stage 0).
    # For image modalities the AR text is a byproduct (the product is the
    # image, already decoded above) and does not feed training, so a failed
    # extraction must not break an otherwise-successful image rollout. The AR
    # text that DOES feed training (cot_text for the two-engine path) flows
    # through the i2t/t2t/ar_recaption branch above, which is — and stays —
    # un-guarded.
    if modality in ("t2i", "it2i") and decoded_text is None:
        try:
            decoded_text = _decoded_text_from_ar(per_request_outputs)
        except Exception:
            logger.debug(
                "AR text extraction failed for modality=%s; leaving decoded_text=None.",
                modality,
                exc_info=True,
            )

    # AR segment is shared by all modalities (Stage 0 always runs).
    ar_segment = extract_ar_segment(per_request_outputs)
    if ar_segment is not None:
        segments_for_track["ar"] = ar_segment

    # Tracks are one per ``segments_for_track`` key, each carrying its
    # own decoded value (or ``None``): the "image" track holds the DiT
    # pixels, the "ar" track holds the AR-decoded text.
    #
    # ``conditions`` were resp-wide in the legacy shape (one dict shared
    # across all modalities); keep that behavior by replicating onto every
    # track. The trainer reads conditions per-track during replay, and today
    # the legacy single-image-track replay is the only consumer; any AR
    # consumer that needs different conditions later can override after
    # construction.
    sample_ids = list(req.sample_ids)
    parent_ids = list(req.group_ids)
    decoded_for_track: Dict[str, Optional[Any]] = {
        "image": decoded_image,
        "video": decoded_video,
        "ar": decoded_text,
    }
    # HI3 think_recaption: image is generated from AR output 1-to-1,
    # so image.parent_track = "ar" and parent_ids align with ar.sample_ids.
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


__all__ = ["_to_rollout_resp", "group_by_request"]
