"""Qwen-Image family: input/output sub-adapters + the ``qwen_image_t2i`` modality class."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import torch

from unirl.rollout.engine.vllm_omni.adapters.base import ModelAdapter, register_adapter
from unirl.rollout.engine.vllm_omni.adapters.dit import (
    DitInputAdapter,
    DitOutputAdapter,
    _grouped_texts_from_sample,
    _negative_prompt_from_params,
)
from unirl.rollout.engine.vllm_omni.backends import GenerateCall, OmniRawResult, StageSampling
from unirl.rollout.engine.vllm_omni.pipelines._shared.interception import read_captures
from unirl.rollout.engine.vllm_omni.utils import collect_dit_outputs
from unirl.types.conditions.text import TextEmbedCondition
from unirl.types.sample import Sample
from unirl.types.sampling import DiffusionSamplingParams


def _ragged_pad_cat(pairs: Sequence[Tuple[torch.Tensor, torch.Tensor]]) -> TextEmbedCondition:
    """Per-request ``(embeds, mask)`` pairs into one ``TextEmbedCondition`` right-padded to the batch-max ``L``."""
    max_len = max(int(e.shape[1]) for e, _ in pairs)
    embeds: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    for e, m in pairs:
        pad = max_len - int(e.shape[1])
        if pad:
            e = torch.cat([e, e.new_zeros(e.shape[0], pad, e.shape[2])], dim=1)
            m = torch.cat([m, m.new_zeros(m.shape[0], pad)], dim=1)
        embeds.append(e)
        masks.append(m)
    return TextEmbedCondition(embeds=torch.cat(embeds, dim=0), pooled=None, attn_mask=torch.cat(masks, dim=0))


class QwenImageInputAdapter(DitInputAdapter):
    """SD3-style request side with the Qwen CFG mapping."""

    def __init__(self, modality: str, *, model_config: Any = None) -> None:
        super().__init__(modality)
        self.model_config = model_config

    def build_prompts(self, sample: Sample) -> List[Any]:
        """``{"prompt"}`` dicts; ``negative_prompt`` ONLY when CFG is armed."""
        # text-only consumer: text_conditioning() fails loud if an image turn is present.
        texts = sample.text_conditioning()[0].content
        diff_params = sample.frontier_gen_part(DiffusionSamplingParams).sampling_params
        if float(diff_params.guidance_scale) > 1.0:
            negative_prompt = _negative_prompt_from_params(diff_params, default=" ")
            return [{"prompt": text, "negative_prompt": negative_prompt} for text in texts.texts]
        return [{"prompt": text} for text in texts.texts]

    def build_sampling(self, sample: Sample) -> List[StageSampling]:
        sampling = super().build_sampling(sample)
        diff_params = sample.frontier_gen_part(DiffusionSamplingParams).sampling_params
        kwargs = sampling[0].kwargs
        kwargs["true_cfg_scale"] = float(diff_params.guidance_scale)
        if "max_sequence_length" not in kwargs:
            max_seq_len = getattr(self.model_config, "max_sequence_length", None)
            if max_seq_len is not None:
                kwargs["max_sequence_length"] = int(max_seq_len)
        return sampling


class QwenImageGroupedInputAdapter(QwenImageInputAdapter):
    """Qwen-Image request builder using vLLM-Omni's native multi-output prompt shape."""

    def build_prompts(self, sample: Sample) -> List[Any]:
        grouped_texts, _ = _grouped_texts_from_sample(
            sample,
            caller=f"{self.modality}.build_prompts",
        )
        diff_params = sample.frontier_gen_part(DiffusionSamplingParams).sampling_params
        if float(diff_params.guidance_scale) > 1.0:
            negative_prompt = _negative_prompt_from_params(diff_params, default=" ")
            return [{"prompt": text, "negative_prompt": negative_prompt} for text in grouped_texts]
        return [{"prompt": text} for text in grouped_texts]

    def build_sampling(self, sample: Sample) -> List[StageSampling]:
        _, spp = _grouped_texts_from_sample(
            sample,
            caller=f"{self.modality}.build_sampling",
        )
        sampling = super().build_sampling(sample)
        sampling[0].kwargs["num_outputs_per_prompt"] = spp
        return sampling


class QwenImageOutputAdapter(DitOutputAdapter):
    """Single diffusion-Part response with Qwen text-capture conditions."""

    _MISSING_CAPTURE_MSG = (
        "build_response: Qwen-Image rollout returned no 'text_capture' on "
        "the output envelope's unirl metadata. Check that RLQwenImagePipeline's "
        "encode_prompt tap ran in every DiT worker — the subclass swap may "
        "not have taken effect (verify custom_pipeline_args.pipeline_class "
        "in the stage YAML)."
    )

    def build_conditions(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Dict[str, Any]:
        """Ragged-pad-concat the per-request Qwen ``text_capture`` dicts."""
        diff_outputs, _, _ = collect_dit_outputs(
            per_request, final_output_type=self.final_output_type, stage_id=self.stage_id, modality=self.modality
        )

        captures = [read_captures(d).get("text_capture") for d in diff_outputs]
        if any(c is None for c in captures):
            raise RuntimeError(self._MISSING_CAPTURE_MSG)

        cond_dict: Dict[str, Any] = {
            "text": _ragged_pad_cat([(c["prompt_embeds"], c["prompt_embeds_mask"]) for c in captures])
        }
        neg_present = [c.get("negative_prompt_embeds") is not None for c in captures]
        if any(neg_present):
            if not all(neg_present):
                raise RuntimeError(
                    "build_response: Qwen-Image negative text captured on some "
                    "requests but not others — CFG arming must be uniform "
                    "across a generate call."
                )
            cond_dict["negative_text"] = _ragged_pad_cat(
                [(c["negative_prompt_embeds"], c["negative_prompt_embeds_mask"]) for c in captures]
            )
        n_samples = len(sample.frontier_gen_part(DiffusionSamplingParams).sample_ids)
        for name, condition in cond_dict.items():
            if int(condition.embeds.shape[0]) != n_samples:
                raise RuntimeError(
                    f"build_response: Qwen-Image {name} condition batch "
                    f"{int(condition.embeds.shape[0])} != diffusion sample count {n_samples}."
                )
        return cond_dict


@register_adapter("qwen_image_t2i")
class QwenImageT2iAdapter(ModelAdapter):
    """Qwen-Image text → image (single diffusion stage, TP=1)."""

    stage_yaml = "qwen_image_t2i_rl.yaml"
    omni_mode = "text-to-image"
    needs_driver_tokenizer = False

    def __init__(self, config: Any, model_config: Any, *, strategy: Any = None, tokenize_fn: Any = None) -> None:
        super().__init__(config, model_config, strategy=strategy, tokenize_fn=tokenize_fn)
        self.input_adapter = QwenImageGroupedInputAdapter(self.modality, model_config=model_config)
        self.output_adapter = QwenImageOutputAdapter(self.modality)

    def validate_request(self, sample: Sample) -> None:
        if sample.has_image_input():
            raise ValueError(
                f"modality={self.modality!r} rejects image-bearing requests; use an image-conditioned modality instead."
            )

    def build_inputs(self, sample: Sample) -> List[GenerateCall]:
        return self.input_adapter.build(sample)

    def build_response(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Sample:
        return self.output_adapter.build(sample, per_request)


__all__ = ["QwenImageGroupedInputAdapter", "QwenImageInputAdapter", "QwenImageOutputAdapter", "QwenImageT2iAdapter"]
