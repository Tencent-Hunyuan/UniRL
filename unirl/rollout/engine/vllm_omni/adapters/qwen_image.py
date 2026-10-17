"""Qwen-Image family: output sub-adapter + the ``qwen_image_t2i`` modality class."""

from __future__ import annotations

from typing import Any, Dict, List

from unirl.rollout.engine.vllm_omni.adapters.base import ModelAdapter, register_adapter
from unirl.rollout.engine.vllm_omni.adapters.dit import DitInputAdapter, DitOutputAdapter
from unirl.rollout.engine.vllm_omni.backends import GenerateCall, OmniRawResult
from unirl.rollout.engine.vllm_omni.pipelines._shared.interception import read_captures
from unirl.rollout.engine.vllm_omni.utils import collect_dit_outputs
from unirl.types.conditions.text import TextEmbedCondition
from unirl.types.sample import Sample
from unirl.types.sampling import DiffusionSamplingParams


class QwenImageOutputAdapter(DitOutputAdapter):
    """Single diffusion-Part response with Qwen text-capture conditions."""

    MISSING_CAPTURE_MSG = (
        "build_response: Qwen-Image rollout returned no 'text_capture' on "
        "the output envelope's unirl metadata. Check that RLQwenImagePipeline's "
        "encode_prompt tap ran in every DiT worker — the subclass swap may "
        "not have taken effect (verify custom_pipeline_args.pipeline_class "
        "in the deploy config)."
    )

    def build_conditions(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Dict[str, Any]:
        """Ragged-pad-concat the per-request Qwen ``text_capture`` dicts."""
        diff_outputs, _, _ = collect_dit_outputs(
            per_request, final_output_type=self.final_output_type, stage_id=self.stage_id, modality=self.modality
        )

        captures = [read_captures(d).get("text_capture") for d in diff_outputs]
        if any(c is None for c in captures):
            raise RuntimeError(self.MISSING_CAPTURE_MSG)

        neg_present = [c.get("negative_prompt_embeds") is not None for c in captures]
        if any(neg_present) and not all(neg_present):
            raise RuntimeError(
                "build_response: Qwen-Image negative text captured on some "
                "requests but not others — CFG arming must be uniform "
                "across a generate call."
            )

        specs = [("text", "prompt_embeds", "prompt_embeds_mask")]
        if all(neg_present):
            specs.append(("negative_text", "negative_prompt_embeds", "negative_prompt_embeds_mask"))

        n_samples = len(sample.frontier_gen_part(DiffusionSamplingParams).sample_ids)
        cond_dict: Dict[str, Any] = {}
        for name, emb_key, mask_key in specs:
            condition = TextEmbedCondition.concat(
                [TextEmbedCondition(embeds=c[emb_key], attn_mask=c[mask_key]) for c in captures]
            )
            if int(condition.embeds.shape[0]) != n_samples:
                raise RuntimeError(
                    f"build_response: Qwen-Image {name} condition batch "
                    f"{int(condition.embeds.shape[0])} != diffusion sample count {n_samples}."
                )
            cond_dict[name] = condition
        return cond_dict


@register_adapter("qwen_image_t2i")
class QwenImageT2iAdapter(ModelAdapter):
    """Qwen-Image text → image (single diffusion stage, TP=1)."""

    deploy_config = "qwen_image_t2i_rl.yaml"
    omni_mode = "text-to-image"
    needs_driver_tokenizer = False
    packs_groups = True

    def __init__(self, config: Any, model_config: Any, *, strategy: Any = None, tokenize_fn: Any = None) -> None:
        super().__init__(config, model_config, strategy=strategy, tokenize_fn=tokenize_fn)
        self.input_adapter = DitInputAdapter(self.modality)
        self.output_adapter = QwenImageOutputAdapter(self.modality)

    def validate_request(self, sample: Sample) -> None:
        if sample.has_image_input():
            raise ValueError(
                f"modality={self.modality!r} rejects image-bearing requests; use an image-conditioned modality instead."
            )

    def build_inputs(self, sample: Sample) -> List[GenerateCall]:
        grouped_texts, spp = self.input_adapter.build_grouped_texts(sample)
        diff_params = sample.frontier_gen_part(DiffusionSamplingParams).sampling_params
        if diff_params.guidance_scale > 1.0:
            neg = diff_params.sampler_kwargs.get("negative_prompt")
            negative_prompt = " " if neg is None else neg
            prompts = [{"prompt": text, "negative_prompt": negative_prompt} for text in grouped_texts]
        else:
            prompts = [{"prompt": text} for text in grouped_texts]
        sampling = self.input_adapter.build_sampling(sample)
        kwargs = sampling[0].kwargs
        kwargs["true_cfg_scale"] = diff_params.guidance_scale
        if "max_sequence_length" not in kwargs:
            kwargs["max_sequence_length"] = self.model_config.max_sequence_length
        kwargs["num_outputs_per_prompt"] = spp
        return [GenerateCall(prompts=prompts, sampling=sampling)]

    def build_response(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Sample:
        return self.output_adapter.build(sample, per_request)


__all__ = ["QwenImageOutputAdapter", "QwenImageT2iAdapter"]
