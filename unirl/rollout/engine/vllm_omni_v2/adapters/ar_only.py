"""Single-"ar"-track shape: HI3 AR-only modalities (no DiT stage).

``i2t`` / ``t2t`` are the upstream comprehension modalities (upstream stage
YAMLs unchanged); ``ar_recaption`` is the two-engine trainer's think/recaption
producer (``is_comprehension: false`` — the decoded text is the recaption the
DiT engine later consumes as ``cot_text``, and ``conditions["fused"]`` carries
the prompt token ids ARGRPO's teacher-forced replay needs).
"""

from __future__ import annotations

from typing import Any, Dict, List

from unirl.rollout.engine.vllm_omni_v2.adapters.ar_dit import ArDiTAdapter
from unirl.rollout.engine.vllm_omni_v2.adapters.base import ModelAdapter, register_adapter
from unirl.rollout.engine.vllm_omni_v2.backends import GenerateCall, OmniRawResult
from unirl.rollout.engine.vllm_omni_v2.utils import (
    assemble_tracks,
    build_ar_fused_condition,
    build_ar_segment,
    build_prompt_entries,
    decoded_text_from_ar,
    images_to_pil,
    resolve_task,
    texts_from_req,
)
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp
from unirl.types.sampling import get_ar_params, get_diffusion_params


class ArOnlyAdapter(ModelAdapter):
    """Per-shape base for the HI3 AR-only modalities."""

    #: AR-only requests carry ``ARSamplingParams`` with no diffusion sub-block
    #: — ``ensure_req_sigmas`` would raise on them.
    needs_sigmas = False
    ar_lora_passthrough = True
    clear_cuda_visible = True
    #: i2t overrides: the request carries ``primitives['image']``.
    image_input = False

    # The AR sampling kwargs are shared with the two-stage shape — borrow the
    # step rather than duplicate it (same config defaults, same logprobs=1).
    build_ar_sampling = ArDiTAdapter.build_ar_sampling

    # ------------------------------------------------------------------ #
    # Request side
    # ------------------------------------------------------------------ #

    def build_inputs(self, req: RolloutReq) -> List[GenerateCall]:
        stage_config = req.stage_config or {}
        task, sys_type, modalities_field = resolve_task(self.modality, stage_config)

        texts = texts_from_req(req)
        n = len(texts.texts)

        pil_images = images_to_pil(req, n) if self.image_input else []
        if self.image_input and not pil_images:
            raise ValueError(f"modality={self.modality!r} requires req.primitives['image']")
        if not self.image_input and req.primitives.get("image") is not None:
            raise ValueError(f"modality={self.modality!r} does not accept req.primitives['image']")

        diff_params = get_diffusion_params(req.sampling_params)
        ar_params = get_ar_params(req.sampling_params) or {}

        height = int(getattr(diff_params, "height", self.cfg.default_height))
        width = int(getattr(diff_params, "width", self.cfg.default_width))

        prompts = build_prompt_entries(
            texts,
            task=task,
            sys_type=sys_type,
            modalities_field=modalities_field,
            tokenize_fn=self.tokenize_fn,
            decorate=lambda entry, i: self.decorate_prompt_entry(
                entry, i, pil_images=pil_images, height=height, width=width
            ),
        )

        return [GenerateCall(prompts=prompts, sampling=[self.build_ar_sampling(ar_params)])]

    def decorate_prompt_entry(
        self, entry: Dict[str, Any], i: int, *, pil_images: List[Any], height: int, width: int
    ) -> None:
        """Default: no extras (the t2t shape). i2t attaches the image;
        ar_recaption attaches the generation height/width."""
        del entry, i, pil_images, height, width

    # ------------------------------------------------------------------ #
    # Response side
    # ------------------------------------------------------------------ #

    def build_response(self, req: RolloutReq, per_request: List[List[OmniRawResult]]) -> RolloutResp:
        if not per_request or not any(per_request):
            raise ValueError("build_response: empty per-request outputs (Omni.generate returned nothing surfaceable).")

        decoded_text = decoded_text_from_ar(per_request)
        conditions = self.build_ar_condition(per_request)

        segments = {}
        ar_segment = build_ar_segment(per_request)
        if ar_segment is not None:
            segments["ar"] = ar_segment

        return assemble_tracks(
            req,
            segments_for_track=segments,
            decoded_for_track={"ar": decoded_text},
            conditions=conditions,
        )

    def build_ar_condition(self, per_request: List[List[OmniRawResult]]) -> Dict[str, Any]:
        """AR-track conditions. Default: none (no diffusion replay in scope)."""
        del per_request
        return {}


@register_adapter("i2t")
class I2tAdapter(ArOnlyAdapter):
    """HI3 image+text → AR text (upstream comprehension YAML)."""

    stage_yaml = "hunyuan_image3_i2t.yaml"
    stage_yaml_source = "upstream"
    image_input = True

    def validate_request(self, req: RolloutReq) -> None:
        if req.primitives.get("image") is None:
            raise ValueError("modality='i2t' requires req.primitives['image'].")

    def decorate_prompt_entry(
        self, entry: Dict[str, Any], i: int, *, pil_images: List[Any], height: int, width: int
    ) -> None:
        # Carry h/w for completeness even though i2t doesn't run the DiT;
        # harmless and matches end2end.py.
        del height, width
        pil = pil_images[i]
        entry["multi_modal_data"] = {"image": pil}
        entry["height"] = pil.height
        entry["width"] = pil.width


@register_adapter("t2t")
class T2tAdapter(ArOnlyAdapter):
    """HI3 text → AR text (upstream comprehension YAML)."""

    stage_yaml = "hunyuan_image3_t2t.yaml"
    stage_yaml_source = "upstream"

    def validate_request(self, req: RolloutReq) -> None:
        if req.primitives.get("image") is not None:
            raise ValueError("modality='t2t' rejects image-bearing requests; use modality='i2t' instead.")


@register_adapter("ar_recaption")
class ArRecaptionAdapter(ArOnlyAdapter):
    """Two-engine trainer's AR think/recaption producer."""

    stage_yaml = "hunyuan_image3_ar_recaption_rl.yaml"
    #: HI3 two-engine stages are TP>1 — wake-time LoRA re-push must use the
    #: byte-copy transport (a zero-copy handle crashes ranks 2..N).
    lora_copy_transport = True

    def decorate_prompt_entry(
        self, entry: Dict[str, Any], i: int, *, pil_images: List[Any], height: int, width: int
    ) -> None:
        del i, pil_images
        entry["height"] = height
        entry["width"] = width

    def build_ar_condition(self, per_request: List[List[OmniRawResult]]) -> Dict[str, Any]:
        # ARGRPO.replay teacher-forces over prompt+response; it needs the
        # prompt token ids (conditions['fused'].input_ids). vLLM processes
        # each request's prompt independently (no batch padding), so the
        # output's prompt_token_ids is the sample's true, un-padded prompt.
        ar_fused = build_ar_fused_condition(per_request)
        return {"fused": ar_fused} if ar_fused is not None else {}


__all__ = ["ArOnlyAdapter", "ArRecaptionAdapter", "I2tAdapter", "T2tAdapter"]
