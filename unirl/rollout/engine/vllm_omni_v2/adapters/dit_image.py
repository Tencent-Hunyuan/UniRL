"""Single-"image"-track shape: pure-DiT image modalities (no AR prelude).

One member here: ``sd35_t2i`` — SD3.5-medium single diffusion stage; prompt
dicts are the ``{"prompt", "negative_prompt"}`` shape
``StableDiffusion3Pipeline.forward`` accepts; conditions come from the
``encode_prompt`` text capture. The other single-"image" modality,
``dit_recaption`` (HI3), lives in :mod:`.hi3_dit_recaption` and subclasses
:class:`DiTImageAdapter`.
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
    build_image_segment,
    build_sd3_text_condition,
    collect_dit_outputs,
    pils_to_images,
    texts_from_req,
)
from unirl.rollout.engine.vllm_omni_v2.utils.noise import pack_initial_noise_extra_args
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp
from unirl.types.sampling import get_diffusion_params


class DiTImageAdapter(ModelAdapter):
    """Per-shape base for the single-stage DiT → image modalities."""

    omni_mode = "text-to-image"

    def build_response(self, req: RolloutReq, per_request: List[List[OmniRawResult]]) -> RolloutResp:
        if not per_request or not any(per_request):
            raise ValueError("build_response: empty per-request outputs (Omni.generate returned nothing surfaceable).")

        # Single-stage: the diffusion stage IS stage 0.
        diff_outputs, _frames, pil_images = collect_dit_outputs(
            per_request, final_output_type="image", stage_id=0, modality=self.modality
        )
        decoded_image = pils_to_images(pil_images)
        segments = {"image": build_image_segment(diff_outputs, expected_sigmas=req.sigmas)}
        conditions = self.build_dit_condition(diff_outputs)

        # Parity with v1's unconditional Stage-0 sweep: a single-DiT stage
        # carries no completions, so this is None unless something upstream
        # surfaces one.
        ar_segment = build_ar_segment(per_request)
        if ar_segment is not None:
            segments["ar"] = ar_segment

        return assemble_tracks(
            req,
            segments_for_track=segments,
            decoded_for_track={"image": decoded_image},
            conditions=conditions,
        )

    def build_dit_condition(self, diff_outputs: List[OmniRawResult]) -> Dict[str, Any]:
        raise NotImplementedError(f"{type(self).__name__} must implement build_dit_condition")


@register_adapter("sd35_t2i")
class Sd35T2iAdapter(DiTImageAdapter):
    """SD3.5-medium text → image (single diffusion stage, TP=1)."""

    stage_yaml = "sd35_t2i_rl.yaml"
    # SD3.5 has no top-level tokenizer (only subfolder CLIP/T5 ones) and the
    # single-stage path never calls build_prompt_tokens.
    needs_driver_tokenizer = False

    def validate_request(self, req: RolloutReq) -> None:
        if req.primitives.get("image") is not None:
            raise ValueError(
                "modality='sd35_t2i' rejects image-bearing requests; "
                "use an image-conditioned modality instead."
            )

    def build_inputs(self, req: RolloutReq) -> List[GenerateCall]:
        if req.primitives.get("image") is not None:
            raise ValueError("modality='sd35_t2i' does not accept req.primitives['image']")

        texts = texts_from_req(req)
        diff_params = get_diffusion_params(req.sampling_params)
        negative_prompt = str(getattr(diff_params, "negative_prompt", "") or "")

        prompts: List[Any] = [{"prompt": text, "negative_prompt": negative_prompt} for text in texts.texts]

        diff_kwargs = self.core_diff_kwargs(req, diff_params)
        max_seq_len = getattr(diff_params, "max_sequence_length", None)
        if max_seq_len is not None:
            diff_kwargs["max_sequence_length"] = int(max_seq_len)
        seed = getattr(diff_params, "seed", None)
        if seed is not None:
            diff_kwargs["seed"] = int(seed)

        extra_args = self.sde_extra_args(diff_params)
        pack_initial_noise_extra_args(
            extra_args, req, diff_params, n_prompts=len(texts.texts), caller="sd35_t2i"
        )
        if extra_args:
            diff_kwargs["extra_args"] = extra_args

        return [
            GenerateCall(
                prompts=prompts,
                sampling=[StageSampling(kind=STAGE_KIND_DIFFUSION, kwargs=diff_kwargs)],
            )
        ]

    def build_dit_condition(self, diff_outputs: List[OmniRawResult]) -> Dict[str, Any]:
        text_cond = build_sd3_text_condition(diff_outputs)
        if text_cond is None:
            raise RuntimeError(
                "build_response: SD3 rollout returned no 'text_capture' on "
                "DiffusionOutput.custom_output. Check that "
                "RLStableDiffusion3Pipeline._install_encode_prompt_hook ran "
                "in every DiT worker — the subclass swap may not have taken "
                "effect (verify custom_pipeline_args.pipeline_class in the "
                "stage YAML)."
            )
        return {"text": text_cond}


__all__ = ["DiTImageAdapter", "Sd35T2iAdapter"]
