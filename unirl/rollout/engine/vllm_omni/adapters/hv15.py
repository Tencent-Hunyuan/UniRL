"""HunyuanVideo-1.5 family: input/output sub-adapters + the ``t2v`` modality class."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from unirl.rollout.engine.vllm_omni.adapters.base import ModelAdapter, register_adapter
from unirl.rollout.engine.vllm_omni.adapters.dit import DitInputAdapter, DitOutputAdapter
from unirl.rollout.engine.vllm_omni.backends import GenerateCall, OmniRawResult, StageSampling
from unirl.rollout.engine.vllm_omni.utils import (
    collect_dit_outputs,
    pick_stage_output,
)
from unirl.types.conditions.text import TextEmbedCondition
from unirl.types.primitives import Video, Videos
from unirl.types.sample import Sample
from unirl.types.sampling import DiffusionSamplingParams


def _num_frames(sample: Sample) -> int:
    return int(getattr(sample.frontier_gen_part(DiffusionSamplingParams).sampling_params, "num_frames", 5))


class Hv15InputAdapter(DitInputAdapter):
    """SD3-style request side + the video-only ``num_frames`` knob."""

    def build_prompts(self, sample: Sample) -> List[Any]:
        prompts = super().build_prompts(sample)
        num_frames = _num_frames(sample)
        for prompt in prompts:
            prompt["num_frames"] = num_frames
        return prompts

    def build_sampling(self, sample: Sample) -> List[StageSampling]:
        sampling = super().build_sampling(sample)
        sampling[0].kwargs["num_frames"] = _num_frames(sample)
        frontier = sample.frontier_gen_part(DiffusionSamplingParams)
        extra_args = sampling[0].kwargs.setdefault("extra_args", {})
        extra_args["denoise_seeds"] = [str(sample_id) for sample_id in frontier.sample_ids]
        return sampling


class Hv15VideoOutputAdapter(DitOutputAdapter):
    """Single diffusion Part response: video frame groups + dual-stream conditions."""

    final_output_type = "video"

    _MISSING_CAPTURE_MSG = (
        "build_response: HV1.5 t2v rollout returned no 'text_capture' "
        "on DiffusionOutput.custom_output (or it lacked the dual-stream "
        "text_mllm/text_glyph embeds). Check that "
        "RLHunyuanVideo15Pipeline's encode_prompt hook ran in every DiT "
        "worker — verify custom_pipeline_args.pipeline_class in the stage "
        "YAML."
    )

    def build_decoded(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Videos:
        """Build float32 ``[T,C,H,W]`` videos directly from raw ``[-1,1]`` VAE output."""
        del sample
        items: List[Video] = []
        for outputs in per_request:
            diff_out = pick_stage_output(outputs, final_output_type=self.final_output_type, stage_id=self.stage_id)
            if diff_out is None:
                raise RuntimeError(f"build_response: no video output for modality={self.modality}")
            decoded = (getattr(diff_out, "custom_output", None) or {}).get("rl_decoded_video")
            if not torch.is_tensor(decoded) or decoded.ndim != 5 or int(decoded.shape[0]) != 1:
                shape = None if not torch.is_tensor(decoded) else tuple(decoded.shape)
                raise RuntimeError(f"build_response: expected raw VAE video [1,C,T,H,W], got {shape}")
            frames = ((decoded[0].to(torch.float32) + 1.0) / 2.0).clamp(0.0, 1.0)
            items.append(Video(frames=frames.permute(1, 0, 2, 3).contiguous()))
        return Videos.from_list(items)

    def build_conditions(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Dict[str, Any]:
        """Unpack the per-request HV1.5 dual-stream text conditions."""
        del sample
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
    needs_driver_tokenizer = False

    def boot_kwargs(self) -> Dict[str, Any]:
        """Pin the vLLM diffusion kernel to the trainer's SDPA path."""
        kwargs = super().boot_kwargs()
        kwargs["diffusion_attention_backend"] = "TORCH_SDPA"
        return kwargs

    def __init__(self, config: Any, model_config: Any, *, strategy: Any = None, tokenize_fn: Any = None) -> None:
        super().__init__(config, model_config, strategy=strategy, tokenize_fn=tokenize_fn)
        self.input_adapter = Hv15InputAdapter(self.modality)
        self.output_adapter = Hv15VideoOutputAdapter(self.modality)

    def validate_request(self, sample: Sample) -> None:
        if sample.has_image_input():
            raise ValueError(
                f"modality={self.modality!r} rejects image-bearing requests; use an image-conditioned modality instead."
            )

    def build_inputs(self, sample: Sample) -> List[GenerateCall]:
        return self.input_adapter.build(sample)

    def build_response(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Sample:
        return self.output_adapter.build(sample, per_request)


__all__ = ["Hv15InputAdapter", "Hv15T2vAdapter", "Hv15VideoOutputAdapter"]
