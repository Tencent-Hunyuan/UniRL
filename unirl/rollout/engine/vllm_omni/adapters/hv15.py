"""HunyuanVideo-1.5 family: input/output sub-adapters + the ``t2v`` modality class.

Single diffusion stage, TP=1, no AR prelude. The request side derives from
the shared :class:`~.dit.DitInputAdapter` adding the video-only
``num_frames`` knob; the response side derives from
:class:`~.dit.DitOutputAdapter` packing per-prompt PIL frame groupings into
``Videos`` and the dual-stream HV1.5 text conditions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from unirl.rollout.engine.vllm_omni.adapters.base import ModelAdapter, register_adapter
from unirl.rollout.engine.vllm_omni.adapters.dit import DitInputAdapter, DitOutputAdapter
from unirl.rollout.engine.vllm_omni.backends import GenerateCall, OmniRawResult, StageSampling
from unirl.rollout.engine.vllm_omni.utils import collect_dit_outputs, grouped_pils_to_videos
from unirl.types.conditions.text import TextEmbedCondition
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp
from unirl.types.sampling import get_diffusion_params


def _num_frames(req: RolloutReq) -> int:
    return int(getattr(get_diffusion_params(req.sampling_params), "num_frames", 5))


class Hv15InputAdapter(DitInputAdapter):
    """SD3-style request side + the video-only ``num_frames`` knob.

    ``num_frames`` rides both the per-prompt dict (read by
    ``RLHunyuanVideo15Pipeline.forward``) and the diffusion kwargs — one
    ``super()``-extend override per side.
    """

    def build_prompts(self, req: RolloutReq) -> List[Any]:
        prompts = super().build_prompts(req)
        num_frames = _num_frames(req)
        for prompt in prompts:
            prompt["num_frames"] = num_frames
        return prompts

    def build_sampling(self, req: RolloutReq) -> List[StageSampling]:
        sampling = super().build_sampling(req)
        sampling[0].kwargs["num_frames"] = _num_frames(req)
        return sampling


class Hv15VideoOutputAdapter(DitOutputAdapter):
    """Single-"video"-track response: frame groupings + dual-stream conditions."""

    track_name = "video"
    final_output_type = "video"

    _MISSING_CAPTURE_MSG = (
        "build_response: HV1.5 t2v rollout returned no 'text_capture' "
        "on DiffusionOutput.custom_output (or it lacked the dual-stream "
        "text_mllm/text_glyph embeds). Check that "
        "RLHunyuanVideo15Pipeline's encode_prompt hook ran in every DiT "
        "worker — verify custom_pipeline_args.pipeline_class in the stage "
        "YAML."
    )

    def build_decoded(self, req: RolloutReq, per_request: List[List[OmniRawResult]]) -> Dict[str, Any]:
        del req
        _, frame_groups, _ = collect_dit_outputs(
            per_request, final_output_type=self.final_output_type, stage_id=self.stage_id, modality=self.modality
        )
        return {self.track_name: grouped_pils_to_videos(frame_groups)}

    def build_conditions(self, req: RolloutReq, per_request: List[List[OmniRawResult]]) -> Dict[str, Any]:
        """Unpack the per-request HV1.5 dual-stream text conditions.

        Written by ``RLHunyuanVideo15Pipeline`` after intercepting
        ``encode_prompt`` — 8 tensors from the dual text encoder (Qwen2.5-VL
        MLLM + ByT5 glyph), mapped to ``text_mllm`` / ``text_glyph``
        (+ negatives). Returns the conditions *dict* (keys aligned with
        ``HunyuanVideo15Conditions.from_dict``), NOT the typed wrapper — the
        trainer runs ``from_dict(track.conditions)`` itself.
        """
        del req
        diff_outputs, _, _ = collect_dit_outputs(
            per_request, final_output_type=self.final_output_type, stage_id=self.stage_id, modality=self.modality
        )

        captures = [(getattr(d, "custom_output", None) or {}).get("text_capture") for d in diff_outputs]
        if any(c is None for c in captures):
            raise RuntimeError(self._MISSING_CAPTURE_MSG)

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

        cond_dict: Dict[str, Any] = {}
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
            raise RuntimeError(self._MISSING_CAPTURE_MSG)
        return cond_dict


@register_adapter("hv15_t2v")
class Hv15T2vAdapter(ModelAdapter):
    """HunyuanVideo-1.5 text → video (single diffusion stage, TP=1)."""

    stage_yaml = "hunyuan_video15_t2v_rl.yaml"
    # HV1.5's tokenizers live in tokenizer/ + tokenizer_2/ subfolders; the
    # worker loads them internally and the driver-side translator needs none.
    needs_driver_tokenizer = False

    def __init__(self, config: Any, model_config: Any, *, strategy: Any = None, tokenize_fn: Any = None) -> None:
        super().__init__(config, model_config, strategy=strategy, tokenize_fn=tokenize_fn)
        self.input_adapter = Hv15InputAdapter(self.modality)
        self.output_adapter = Hv15VideoOutputAdapter(self.modality)

    def validate_request(self, req: RolloutReq) -> None:
        if req.primitives.get("image") is not None:
            raise ValueError(
                f"modality={self.modality!r} rejects image-bearing requests; use an image-conditioned modality instead."
            )

    def build_inputs(self, req: RolloutReq) -> List[GenerateCall]:
        return self.input_adapter.build(req)

    def build_response(self, req: RolloutReq, per_request: List[List[OmniRawResult]]) -> RolloutResp:
        return self.output_adapter.build(req, per_request)


__all__ = ["Hv15InputAdapter", "Hv15T2vAdapter", "Hv15VideoOutputAdapter"]
