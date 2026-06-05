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

from typing import Any, Dict, List, Tuple

from unirl.rollout.engine.vllm_omni_v2.backends import (
    STAGE_KIND_DIFFUSION,
    GenerateCall,
    OmniRawResult,
    StageSampling,
)
from unirl.rollout.engine.vllm_omni_v2.utils import (
    assemble_tracks,
    build_ar_segment,
    build_image_segment,
    collect_dit_outputs,
    pils_to_images,
    texts_from_req,
)
from unirl.rollout.engine.vllm_omni_v2.utils.diff_kwargs import core_diff_kwargs, sde_extra_args
from unirl.rollout.engine.vllm_omni_v2.utils.noise import pack_initial_noise_extra_args
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp
from unirl.types.sampling import get_diffusion_params


class DitInputAdapter:
    """``RolloutReq`` → one single-diffusion-stage :class:`GenerateCall`.

    The shared request skeleton of the pure-DiT modalities: text prompts +
    ``negative_prompt``, the typed diffusion kwargs, optional
    ``max_sequence_length`` / ``seed``, sparse SDE indices, and the
    driver-authoritative x_T recipe. Families contribute extra fields via
    :meth:`extras`.
    """

    def __init__(self, modality: str) -> None:
        self.modality = modality

    def extras(self, diff_params: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """``(per-prompt extras, diff-kwargs extras)``. Default: none."""
        del diff_params
        return {}, {}

    def build(self, req: RolloutReq) -> List[GenerateCall]:
        if req.primitives.get("image") is not None:
            raise ValueError(f"modality={self.modality!r} does not accept req.primitives['image']")

        texts = texts_from_req(req)
        diff_params = get_diffusion_params(req.sampling_params)
        negative_prompt = str(getattr(diff_params, "negative_prompt", "") or "")
        prompt_extras, kwargs_extras = self.extras(diff_params)

        prompts: List[Any] = [
            {"prompt": text, "negative_prompt": negative_prompt, **prompt_extras} for text in texts.texts
        ]

        diff_kwargs = core_diff_kwargs(req, diff_params)
        diff_kwargs.update(kwargs_extras)
        max_seq_len = getattr(diff_params, "max_sequence_length", None)
        if max_seq_len is not None:
            diff_kwargs["max_sequence_length"] = int(max_seq_len)
        seed = getattr(diff_params, "seed", None)
        if seed is not None:
            diff_kwargs["seed"] = int(seed)

        extra_args = sde_extra_args(diff_params)
        pack_initial_noise_extra_args(extra_args, req, diff_params, n_prompts=len(texts.texts), caller=self.modality)
        if extra_args:
            diff_kwargs["extra_args"] = extra_args

        return [
            GenerateCall(
                prompts=prompts,
                sampling=[StageSampling(kind=STAGE_KIND_DIFFUSION, kwargs=diff_kwargs)],
            )
        ]


class DitOutputAdapter:
    """Per-request DiT results → a DiT-track :class:`RolloutResp`.

    The shared response skeleton of every DiT-bearing modality: collect the
    DiT outputs, pack the trajectory segment (asserting the σ echo), decode
    the final media, attach the family's replay conditions, sweep a Stage-0
    AR segment for v1 parity, and assemble the tracks.
    """

    #: Track key + the wire ``final_output_type`` to collect. Video families
    #: override both together.
    track_name = "image"
    final_output_type = "image"

    def __init__(self, modality: str, *, stage_id: int = 0) -> None:
        self.modality = modality
        self.stage_id = stage_id

    # ------------------------------------------------------------------ #
    # Family hooks
    # ------------------------------------------------------------------ #

    def conditions(self, diff_outputs: List[OmniRawResult]) -> Dict[str, Any]:
        """The family's replay conditions, extracted from the DiT outputs."""
        raise NotImplementedError(f"{type(self).__name__} must implement conditions()")

    def build_decoded(self, per_request: List[List[OmniRawResult]]) -> Dict[str, Any]:
        """The per-track ``decoded`` payloads, from the raw per-request groups.

        Takes the raw wire groups (not the collected DiT slices) because
        decoded may span tracks beyond the DiT one — the HI3 two-track shape
        adds the AR text via ``super()``. Must keep the ``track_name`` entry
        (a missing key silently yields ``decoded=None`` on that track).

        Default: the flat PILs as ``Images``; hv15 swaps the payload for
        packed frame groups. Re-collecting here is deliberate and cheap —
        ``collect_dit_outputs`` only gathers references.
        """
        _, _, pil_images = collect_dit_outputs(
            per_request, final_output_type=self.final_output_type, stage_id=self.stage_id, modality=self.modality
        )
        return {self.track_name: pils_to_images(pil_images)}

    # ------------------------------------------------------------------ #
    # Skeleton
    # ------------------------------------------------------------------ #

    def build(self, req: RolloutReq, per_request: List[List[OmniRawResult]]) -> RolloutResp:
        if not per_request or not any(per_request):
            raise ValueError("build_response: empty per-request outputs (Omni.generate returned nothing surfaceable).")

        diff_outputs, _frames, _pils = collect_dit_outputs(
            per_request, final_output_type=self.final_output_type, stage_id=self.stage_id, modality=self.modality
        )
        decoded = self.build_decoded(per_request)
        segments = {self.track_name: build_image_segment(diff_outputs, expected_sigmas=req.sigmas)}
        conditions = self.conditions(diff_outputs)

        # Parity with v1's unconditional Stage-0 sweep: a single-DiT stage
        # carries no completions, so this is None unless something upstream
        # surfaces one (the HI3 two-stage shape always does).
        ar_segment = build_ar_segment(per_request)
        if ar_segment is not None:
            segments["ar"] = ar_segment

        return assemble_tracks(
            req,
            segments_for_track=segments,
            decoded_for_track=decoded,
            conditions=conditions,
        )


__all__ = ["DitInputAdapter", "DitOutputAdapter"]
