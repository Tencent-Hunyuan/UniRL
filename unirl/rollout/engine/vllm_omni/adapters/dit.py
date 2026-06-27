"""Shared DiT sub-adapter bases — the universal request/response skeletons.

Two bases, one per conversion side. They hold the frozen single-stage DiT
skeletons; a model family derives a small subclass **in its own file**
overriding hooks only (``Hv15InputAdapter.extras``,
``Sd3OutputAdapter.conditions``, …). A hook or parameter is added here only
when a second family needs the same one — family quirks otherwise stay in
the family's subclass.

Naming rule: universal classes live here with no family prefix;
family-specific sub-adapters carry the family prefix and live in the family
file (``hi3.py`` / ``sd3.py`` / ``hv15.py``).
"""

from __future__ import annotations

from typing import Any, Dict, List

from unirl.rollout.engine.vllm_omni.backends import (
    STAGE_KIND_DIFFUSION,
    GenerateCall,
    OmniRawResult,
    StageSampling,
)
from unirl.rollout.engine.vllm_omni.utils import (
    assemble_tracks,
    build_ar_segment,
    build_image_segment,
    collect_dit_outputs,
    pils_to_images,
    texts_from_req,
)
from unirl.rollout.engine.vllm_omni.utils.diff_kwargs import core_diff_kwargs, sde_extra_args
from unirl.rollout.engine.vllm_omni.utils.noise import pack_initial_noise_extra_args
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp


class DitInputAdapter:
    """``RolloutReq`` → one single-diffusion-stage :class:`GenerateCall`.

    ``build`` is pure assembly over two parallel hooks that mirror the wire
    type's payload fields 1:1 — :meth:`build_prompts` / :meth:`build_sampling`
    — both with the raw-``req`` currency (each hook derives what it needs
    from the typed sampling params; the accessors are cheap). Children
    derive either via ``super()``.
    """

    def __init__(self, modality: str) -> None:
        self.modality = modality

    # ------------------------------------------------------------------ #
    # Family hooks — one per GenerateCall payload field
    # ------------------------------------------------------------------ #

    def build_prompts(self, req: RolloutReq) -> List[Any]:
        """The per-prompt dicts: the ``{"prompt", "negative_prompt"}`` shape."""
        if req.primitives.get("image") is not None:
            raise ValueError(f"modality={self.modality!r} does not accept req.primitives['image']")
        texts = texts_from_req(req)
        diff_params = req.sampling_params.get("diffusion")
        negative_prompt = str(getattr(diff_params, "negative_prompt", "") or "")
        return [{"prompt": text, "negative_prompt": negative_prompt} for text in texts.texts]

    def build_sampling(self, req: RolloutReq) -> List[StageSampling]:
        """The single diffusion-stage intent: the typed diffusion kwargs,
        optional ``max_sequence_length`` / ``seed``, sparse SDE indices, and
        the driver-authoritative x_T recipe."""
        texts = texts_from_req(req)
        diff_params = req.sampling_params.get("diffusion")

        diff_kwargs = core_diff_kwargs(req, diff_params)
        max_seq_len = getattr(diff_params, "max_sequence_length", None)
        if max_seq_len is not None:
            diff_kwargs["max_sequence_length"] = int(max_seq_len)
        seed = getattr(diff_params, "seed", None)
        if seed is not None:
            diff_kwargs["seed"] = int(seed)

        extra_args = sde_extra_args(diff_params)
        pack_initial_noise_extra_args(extra_args, req, diff_params, n_samples=len(texts.texts), caller=self.modality)
        if extra_args:
            diff_kwargs["extra_args"] = extra_args

        return [StageSampling(kind=STAGE_KIND_DIFFUSION, kwargs=diff_kwargs)]

    # ------------------------------------------------------------------ #
    # Skeleton
    # ------------------------------------------------------------------ #

    def build(self, req: RolloutReq) -> List[GenerateCall]:
        return [GenerateCall(prompts=self.build_prompts(req), sampling=self.build_sampling(req))]


class DitOutputAdapter:
    """Per-request DiT results → a DiT-track :class:`RolloutResp`.

    ``build`` is guard + ``assemble_tracks`` over three parallel hooks that
    mirror its parameter list 1:1 — :meth:`build_segments` /
    :meth:`build_decoded` / :meth:`build_conditions` — all with the uniform
    ``(req, per_request)`` currency: raw wire groups in, each hook collects
    what it needs (cheap — ``collect_dit_outputs`` only gathers references).
    Children derive any of the three via ``super()``.
    """

    #: Track key + the wire ``final_output_type`` to collect. Video families
    #: override both together.
    track_name = "image"
    final_output_type = "image"

    def __init__(self, modality: str, *, stage_id: int = 0) -> None:
        self.modality = modality
        self.stage_id = stage_id

    # ------------------------------------------------------------------ #
    # Family hooks — one per assemble_tracks parameter
    # ------------------------------------------------------------------ #

    def build_segments(self, req: RolloutReq, per_request: List[List[OmniRawResult]]) -> Dict[str, Any]:
        """The per-track segments: the DiT trajectory (asserting the σ echo)
        plus the v1-parity Stage-0 AR sweep."""
        diff_outputs, _, _ = collect_dit_outputs(
            per_request, final_output_type=self.final_output_type, stage_id=self.stage_id, modality=self.modality
        )
        segments = {self.track_name: build_image_segment(diff_outputs, expected_sigmas=req.sigmas)}
        # Parity with v1's unconditional Stage-0 sweep: a single-DiT stage
        # carries no completions, so this is None unless something upstream
        # surfaces one (the HI3 two-stage shape always does).
        ar_segment = build_ar_segment(per_request)
        if ar_segment is not None:
            segments["ar"] = ar_segment
        return segments

    def build_decoded(self, req: RolloutReq, per_request: List[List[OmniRawResult]]) -> Dict[str, Any]:
        """The per-track ``decoded`` payloads. Must keep the ``track_name``
        entry (a missing key silently yields ``decoded=None`` on that track).

        Default: the flat PILs as ``Images``; hv15 swaps the payload for
        packed frame groups; the HI3 two-track shape adds the AR text via
        ``super()``.
        """
        del req
        _, _, pil_images = collect_dit_outputs(
            per_request, final_output_type=self.final_output_type, stage_id=self.stage_id, modality=self.modality
        )
        return {self.track_name: pils_to_images(pil_images)}

    def build_conditions(self, req: RolloutReq, per_request: List[List[OmniRawResult]]) -> Dict[str, Any]:
        """The family's replay conditions."""
        raise NotImplementedError(f"{type(self).__name__} must implement build_conditions()")

    # ------------------------------------------------------------------ #
    # Skeleton
    # ------------------------------------------------------------------ #

    def build(self, req: RolloutReq, per_request: List[List[OmniRawResult]]) -> RolloutResp:
        if not per_request or not any(per_request):
            raise ValueError("build_response: empty per-request outputs (Omni.generate returned nothing surfaceable).")
        return assemble_tracks(
            req,
            segments_for_track=self.build_segments(req, per_request),
            decoded_for_track=self.build_decoded(req, per_request),
            conditions=self.build_conditions(req, per_request),
        )


__all__ = ["DitInputAdapter", "DitOutputAdapter"]
