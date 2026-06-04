"""Single-"video"-track shape: HunyuanVideo-1.5 text → video.

One member today, so the concrete ``t2v`` adapter IS the shape (no separate
per-shape base until a second video modality shares it). Mirrors the
``sd35_t2i`` request side (single-stage pure-DiT, no AR prelude) plus the
video-only ``num_frames`` knob; the response side packs per-prompt PIL frame
groupings into ``Videos`` and the dual-stream HV1.5 text conditions.
"""

from __future__ import annotations

from typing import Any, Dict, List

from unirl.rollout.engine.vllm_omni_v2.adapters.base import ModelAdapter, register_adapter
from unirl.rollout.engine.vllm_omni_v2.backends import (
    STAGE_KIND_DIFFUSION,
    GenerateCall,
    OmniRawResult,
    StageSampling,
)
from unirl.rollout.engine.vllm_omni_v2.utils import (
    assemble_tracks,
    build_ar_segment,
    build_hv15_conditions,
    build_image_segment,
    collect_dit_outputs,
    grouped_pils_to_videos,
    texts_from_req,
)
from unirl.rollout.engine.vllm_omni_v2.utils.noise import pack_initial_noise_extra_args
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp
from unirl.types.sampling import get_diffusion_params


@register_adapter("t2v")
class DiTVideoAdapter(ModelAdapter):
    """HunyuanVideo-1.5 text → video (single diffusion stage, TP=1)."""

    stage_yaml = "hunyuan_video15_t2v_rl.yaml"
    # HV1.5's tokenizers live in tokenizer/ + tokenizer_2/ subfolders; the
    # worker loads them internally and the driver-side translator needs none.
    needs_driver_tokenizer = False

    def validate_request(self, req: RolloutReq) -> None:
        if req.primitives.get("image") is not None:
            raise ValueError(
                "modality='t2v' rejects image-bearing requests; "
                "use an image-conditioned modality instead."
            )

    # ------------------------------------------------------------------ #
    # Request side
    # ------------------------------------------------------------------ #

    def build_inputs(self, req: RolloutReq) -> List[GenerateCall]:
        if req.primitives.get("image") is not None:
            raise ValueError("modality='t2v' does not accept req.primitives['image']")

        texts = texts_from_req(req)
        diff_params = get_diffusion_params(req.sampling_params)
        negative_prompt = str(getattr(diff_params, "negative_prompt", "") or "")
        num_frames = int(getattr(diff_params, "num_frames", 5))

        prompts: List[Any] = [
            {"prompt": text, "negative_prompt": negative_prompt, "num_frames": num_frames}
            for text in texts.texts
        ]

        diff_kwargs = self.core_diff_kwargs(req, diff_params)
        diff_kwargs["num_frames"] = num_frames
        max_seq_len = getattr(diff_params, "max_sequence_length", None)
        if max_seq_len is not None:
            diff_kwargs["max_sequence_length"] = int(max_seq_len)
        seed = getattr(diff_params, "seed", None)
        if seed is not None:
            diff_kwargs["seed"] = int(seed)

        extra_args = self.sde_extra_args(diff_params)
        pack_initial_noise_extra_args(
            extra_args, req, diff_params, n_prompts=len(texts.texts), caller="t2v"
        )
        if extra_args:
            diff_kwargs["extra_args"] = extra_args

        return [
            GenerateCall(
                prompts=prompts,
                sampling=[StageSampling(kind=STAGE_KIND_DIFFUSION, kwargs=diff_kwargs)],
            )
        ]

    # ------------------------------------------------------------------ #
    # Response side
    # ------------------------------------------------------------------ #

    def build_response(self, req: RolloutReq, per_request: List[List[OmniRawResult]]) -> RolloutResp:
        if not per_request or not any(per_request):
            raise ValueError("build_response: empty per-request outputs (Omni.generate returned nothing surfaceable).")

        diff_outputs, pil_frames_per_prompt, _flat = collect_dit_outputs(
            per_request, final_output_type="video", stage_id=0, modality=self.modality
        )
        decoded_video = grouped_pils_to_videos(pil_frames_per_prompt)
        segments = {"video": build_image_segment(diff_outputs, expected_sigmas=req.sigmas)}
        conditions = self.build_dit_condition(diff_outputs)

        # Parity with v1's unconditional Stage-0 sweep (None for single-DiT).
        ar_segment = build_ar_segment(per_request)
        if ar_segment is not None:
            segments["ar"] = ar_segment

        return assemble_tracks(
            req,
            segments_for_track=segments,
            decoded_for_track={"video": decoded_video},
            conditions=conditions,
        )

    def build_dit_condition(self, diff_outputs: List[OmniRawResult]) -> Dict[str, Any]:
        hv_conds = build_hv15_conditions(diff_outputs)
        if hv_conds is None:
            raise RuntimeError(
                "build_response: HV1.5 t2v rollout returned no 'text_capture' "
                "on DiffusionOutput.custom_output (or it lacked the dual-stream "
                "text_mllm/text_glyph embeds). Check that "
                "RLHunyuanVideo15Pipeline's encode_prompt hook ran in every DiT "
                "worker — verify custom_pipeline_args.pipeline_class in the stage "
                "YAML."
            )
        return dict(hv_conds)


__all__ = ["DiTVideoAdapter"]
