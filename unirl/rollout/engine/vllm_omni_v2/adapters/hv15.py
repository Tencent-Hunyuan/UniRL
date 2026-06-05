"""HunyuanVideo-1.5 family: input/output sub-adapters + the ``t2v`` modality class.

Single diffusion stage, TP=1, no AR prelude. The request side derives from
the shared :class:`~.dit.DitInputAdapter` adding the video-only
``num_frames`` knob; the response side derives from
:class:`~.dit.DitOutputAdapter` packing per-prompt PIL frame groupings into
``Videos`` and the dual-stream HV1.5 text conditions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from unirl.rollout.engine.vllm_omni_v2.adapters.base import ModelAdapter, register_adapter
from unirl.rollout.engine.vllm_omni_v2.adapters.dit import DitInputAdapter, DitOutputAdapter
from unirl.rollout.engine.vllm_omni_v2.backends import GenerateCall, OmniRawResult
from unirl.rollout.engine.vllm_omni_v2.utils import (
    build_hv15_conditions,
    collect_dit_outputs,
    grouped_pils_to_videos,
)
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp


class Hv15InputAdapter(DitInputAdapter):
    """SD3-style request side + the video-only ``num_frames`` knob.

    ``num_frames`` rides both the per-prompt dict (read by
    ``RLHunyuanVideo15Pipeline.forward``) and the diffusion kwargs.
    """

    def extras(self, diff_params: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        num_frames = int(getattr(diff_params, "num_frames", 5))
        return {"num_frames": num_frames}, {"num_frames": num_frames}


class Hv15VideoOutputAdapter(DitOutputAdapter):
    """Single-"video"-track response: frame groupings + dual-stream conditions."""

    track_name = "video"
    final_output_type = "video"

    def build_decoded(self, per_request: List[List[OmniRawResult]]) -> Dict[str, Any]:
        _, frame_groups, _ = collect_dit_outputs(
            per_request, final_output_type=self.final_output_type, stage_id=self.stage_id, modality=self.modality
        )
        return {self.track_name: grouped_pils_to_videos(frame_groups)}

    def conditions(self, diff_outputs: List[OmniRawResult]) -> Dict[str, Any]:
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
