"""Shared DiT sub-adapter bases — the universal request/response skeletons."""

from __future__ import annotations

from typing import Any, Dict, List

from unirl.rollout.engine.vllm_omni.backends import (
    STAGE_KIND_DIFFUSION,
    OmniRawResult,
    StageSampling,
)
from unirl.rollout.engine.vllm_omni.utils import (
    build_image_segment,
    collect_dit_outputs,
    pils_to_images,
)
from unirl.rollout.engine.vllm_omni.utils.diff_kwargs import core_diff_kwargs, sde_extra_args
from unirl.rollout.engine.vllm_omni.utils.noise import pack_initial_noise_extra_args
from unirl.types.primitives import Texts, primitive_modality_key
from unirl.types.sample import Sample
from unirl.types.sampling import DiffusionSamplingParams


class DitInputAdapter:
    """Request ``Sample`` → single-diffusion-stage prompt and sampling builders."""

    def __init__(self, modality: str) -> None:
        self.modality = modality

    def build_prompts(self, sample: Sample) -> List[Any]:
        """The per-prompt dicts: the ``{"prompt", "negative_prompt"}`` shape."""
        # text-only consumer: text_conditioning() fails loud if an image turn is present.
        texts = sample.text_conditioning()[0].content
        diff_params = sample.frontier_gen_part(DiffusionSamplingParams).sampling_params
        neg = diff_params.sampler_kwargs.get("negative_prompt")
        negative_prompt = "" if neg is None else neg
        return [{"prompt": text, "negative_prompt": negative_prompt} for text in texts.texts]

    def build_grouped_texts(self, sample: Sample) -> tuple[List[str], int]:
        """Collapse a Sample's complete contiguous diffusion groups to engine prompts."""
        caller = f"{self.modality}.build_inputs"
        gen_part = sample.frontier_gen_part(DiffusionSamplingParams)
        diff_params = gen_part.sampling_params
        spp = diff_params.samples_per_prompt
        if spp < 1:
            raise ValueError(f"{caller}: samples_per_prompt must be >= 1, got {spp}")

        turns = sample.text_conditioning()
        if len(turns) != 1 or not isinstance(turns[0].content, Texts):
            raise ValueError(f"{caller}: expected exactly one frontier-aligned text turn, got {len(turns)}")
        texts = list(turns[0].content.texts)
        n_samples = len(gen_part.sample_ids)
        if len(texts) != n_samples:
            raise RuntimeError(f"{caller}: prompt count {len(texts)} != diffusion sample count {n_samples}")
        if n_samples % spp != 0:
            raise RuntimeError(
                f"{caller}: shard sample count {n_samples} is not divisible by samples_per_prompt={spp}; "
                "DP scatter split a prompt group. Use a layout where each shard contains whole groups."
            )

        group_ids = list(gen_part.group_ids)
        if len(group_ids) != n_samples:
            raise RuntimeError(f"{caller}: group_ids count {len(group_ids)} != diffusion sample count {n_samples}")

        grouped: List[str] = []
        seen_groups: set[str] = set()
        for start in range(0, n_samples, spp):
            end = start + spp
            group_texts = texts[start:end]
            if any(text != group_texts[0] for text in group_texts[1:]):
                raise RuntimeError(
                    f"{caller}: prompt rows [{start}:{end}] are not repeated within their generation group."
                )
            group = group_ids[start:end]
            if any(group_id != group[0] for group_id in group[1:]):
                raise RuntimeError(f"{caller}: group_ids rows [{start}:{end}] are not one contiguous group.")
            if group[0] in seen_groups:
                raise RuntimeError(f"{caller}: group_id {group[0]!r} appears in multiple non-contiguous groups.")
            seen_groups.add(group[0])
            grouped.append(group_texts[0])
        return grouped, spp

    def build_sampling(self, sample: Sample) -> List[StageSampling]:
        """The diffusion-stage intent: typed kwargs, optional length/seed, sparse SDE indices, and the x_T recipe."""
        gen_part = sample.frontier_gen_part(DiffusionSamplingParams)
        diff_params = gen_part.sampling_params

        diff_kwargs = core_diff_kwargs(diff_params)
        max_seq_len = diff_params.max_sequence_length
        if max_seq_len is not None:
            diff_kwargs["max_sequence_length"] = max_seq_len
        seed = diff_params.seed
        if seed is not None:
            diff_kwargs["seed"] = seed

        extra_args = sde_extra_args(diff_params)
        pack_initial_noise_extra_args(extra_args, gen_part, diff_params, caller=self.modality)
        if extra_args:
            diff_kwargs["extra_args"] = extra_args

        return [StageSampling(kind=STAGE_KIND_DIFFUSION, kwargs=diff_kwargs)]


class DitOutputAdapter:
    """Per-request DiT results → the filled diffusion ``Part`` of the ``Sample``."""

    final_output_type = "image"

    def __init__(self, modality: str, *, stage_id: int = 0) -> None:
        self.modality = modality
        self.stage_id = stage_id

    def build_decoded(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Any:
        """The decoded diffusion primitive."""
        del sample
        _, _, pil_images = collect_dit_outputs(
            per_request, final_output_type=self.final_output_type, stage_id=self.stage_id, modality=self.modality
        )
        return pils_to_images(pil_images)

    def build_conditions(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Dict[str, Any]:
        """The family's replay conditions."""
        raise NotImplementedError(f"{type(self).__name__} must implement build_conditions()")

    def build(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Sample:
        if not per_request or not any(per_request):
            raise ValueError("build_response: empty per-request outputs (Omni.generate returned nothing surfaceable).")
        diff_outputs, _, _ = collect_dit_outputs(
            per_request, final_output_type=self.final_output_type, stage_id=self.stage_id, modality=self.modality
        )
        expected_sigmas = sample.frontier_gen_part(DiffusionSamplingParams).sampling_params.sigmas
        segment = build_image_segment(diff_outputs, expected_sigmas=expected_sigmas)
        decoded = self.build_decoded(sample, per_request)
        conditions = self.build_conditions(sample, per_request)
        frontier = sample.frontier_gen_part(DiffusionSamplingParams)
        return sample.replace_frontier(
            frontier.fill(
                segment=segment,
                primitives={primitive_modality_key(decoded): decoded},
                conditions=dict(conditions),
            )
        )


__all__ = ["DitInputAdapter", "DitOutputAdapter"]
