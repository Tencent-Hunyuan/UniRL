"""BAGEL-7B-MoT family: input/output sub-adapters + the ``bagel_t2i`` modality class."""

from __future__ import annotations

from typing import Any, Dict, List

from unirl.models.bagel.conditions import BagelDiffusionConditions
from unirl.rollout.engine.vllm_omni.adapters.base import ModelAdapter, register_adapter
from unirl.rollout.engine.vllm_omni.adapters.dit import (
    DitInputAdapter,
    DitOutputAdapter,
    _grouped_texts_from_sample,
)
from unirl.rollout.engine.vllm_omni.backends import (
    STAGE_KIND_DIFFUSION,
    GenerateCall,
    OmniRawResult,
    StageSampling,
)
from unirl.rollout.engine.vllm_omni.utils import (
    build_image_segment,
    collect_dit_outputs,
    pils_to_images,
)
from unirl.rollout.engine.vllm_omni.utils.noise import pack_initial_noise_extra_args
from unirl.rollout.engine.vllm_omni.utils.sigmas import sigmas_list_from_diffusion
from unirl.sde.runtime import FlowMatchSchedulePolicy
from unirl.types.sample import Sample
from unirl.types.sampling import DiffusionSamplingParams


class BagelInputAdapter(DitInputAdapter):
    """Request side: prompt dicts + the BAGEL diffusion-stage sampling intent."""

    def _spp(self, sample: Sample) -> int:
        """``samples_per_prompt`` — the GRPO group size; 1 disables packing."""
        diff_params = sample.frontier_gen_part(DiffusionSamplingParams).sampling_params
        raw_spp = getattr(diff_params, "samples_per_prompt", 1)
        spp = 1 if raw_spp is None else int(raw_spp)
        if spp < 1:
            raise ValueError(f"{self.modality}: samples_per_prompt must be >= 1, got {spp}")
        return spp

    def _is_packable_t2i(self, sample: Sample) -> bool:
        """Collapse spp samples into one ``num_outputs_per_prompt=spp`` request."""
        if self._spp(sample) <= 1:
            return False
        diff_params = sample.frontier_gen_part(DiffusionSamplingParams).sampling_params
        return float(diff_params.cfg_text_scale) <= 1.0 and float(diff_params.cfg_img_scale) <= 1.0

    def build_prompts(self, sample: Sample) -> List[Any]:
        """Plain ``{"prompt": text}`` dicts (no ``modalities`` → image path)."""
        if sample.has_image_input():
            raise ValueError(f"modality={self.modality!r} does not accept image conditioning")
        spp = self._spp(sample)
        grouped_texts, grouped_spp = _grouped_texts_from_sample(
            sample,
            caller=f"{self.modality}.build_prompts",
        )
        if grouped_spp != spp:
            raise RuntimeError(
                f"{self.modality}.build_prompts: inconsistent samples_per_prompt "
                f"({grouped_spp} from grouping, {spp} from diffusion params)."
            )

        gen_part = sample.frontier_gen_part(DiffusionSamplingParams)
        pack = self._is_packable_t2i(sample)
        if pack:
            prompt_texts = grouped_texts
            num_outputs_per_prompt = spp
        else:
            prompt_texts = list(sample.text_conditioning()[0].content.texts)
            num_outputs_per_prompt = 1

        n_samples = len(gen_part.sample_ids)
        if len(prompt_texts) * num_outputs_per_prompt != n_samples:
            raise RuntimeError(
                f"{self.modality}.build_prompts: prompt count {len(prompt_texts)} * "
                f"num_outputs_per_prompt={num_outputs_per_prompt} != diffusion sample count {n_samples}."
            )
        return [{"prompt": text} for text in prompt_texts]

    def build_sampling(self, sample: Sample) -> List[StageSampling]:
        """One diffusion-stage intent with the BAGEL-specific kwargs."""
        spp = self._spp(sample)
        grouped_texts, grouped_spp = _grouped_texts_from_sample(
            sample,
            caller=f"{self.modality}.build_sampling",
        )
        gen_part = sample.frontier_gen_part(DiffusionSamplingParams)
        diff_params = gen_part.sampling_params
        if grouped_spp != spp:
            raise RuntimeError(
                f"{self.modality}.build_sampling: inconsistent samples_per_prompt "
                f"({grouped_spp} from grouping, {spp} from diffusion params)."
            )
        pack = self._is_packable_t2i(sample)

        n_samples = len(gen_part.sample_ids)
        n_prompts = len(grouped_texts) if pack else n_samples
        num_outputs_per_prompt = spp if pack else 1
        if n_prompts * num_outputs_per_prompt != n_samples:
            raise RuntimeError(
                f"{self.modality}.build_sampling: prompt count {n_prompts} * "
                f"num_outputs_per_prompt={num_outputs_per_prompt} != diffusion sample count {n_samples}."
            )

        T = int(diff_params.num_inference_steps)
        diff_kwargs: Dict[str, Any] = dict(
            height=int(diff_params.height),
            width=int(diff_params.width),
            num_inference_steps=T + 1,
            eta=float(diff_params.eta),
            return_trajectory_latents=True,
            return_trajectory_decoded=False,
            num_outputs_per_prompt=num_outputs_per_prompt,
        )
        seed = getattr(diff_params, "seed", None)
        if seed is not None:
            diff_kwargs["seed"] = int(seed)

        _ = sigmas_list_from_diffusion(diff_params, T)

        extra_args: Dict[str, Any] = {
            "cfg_text_scale": float(getattr(diff_params, "cfg_text_scale", 1.0)),
            "cfg_img_scale": float(getattr(diff_params, "cfg_img_scale", 1.0)),
            "cfg_interval": tuple(getattr(diff_params, "cfg_interval", (0.0, 1.0))),
            "cfg_renorm_min": float(getattr(diff_params, "cfg_renorm_min", 0.0)),
            "cfg_renorm_type": str(getattr(diff_params, "cfg_renorm_type", "global")),
        }
        sde_indices = getattr(diff_params, "sde_indices", None)
        if sde_indices is not None:
            extra_args["sde_indices"] = sorted({int(i) for i in sde_indices})
        if diff_params.sigmas is not None and int(diff_params.sigmas.shape[0]) > 1:
            extra_args["sigma_max"] = float(diff_params.sigmas[1].item())
        traj_prec = getattr(diff_params, "trajectory_precision", None)
        if traj_prec is not None:
            extra_args["trajectory_precision"] = str(traj_prec)

        pack_initial_noise_extra_args(extra_args, gen_part, diff_params, caller=self.modality)
        diff_kwargs["extra_args"] = extra_args

        return [StageSampling(kind=STAGE_KIND_DIFFUSION, kwargs=diff_kwargs)]


class BagelOutputAdapter(DitOutputAdapter):
    """Response side: one image generation Part with prompt-carrying conditions."""

    def build_segment(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Any:
        """The DiT trajectory segment (asserts the σ echo)."""
        diff_outputs, _, _ = collect_dit_outputs(
            per_request, final_output_type=self.final_output_type, stage_id=self.stage_id, modality=self.modality
        )
        diff_params = sample.frontier_gen_part(DiffusionSamplingParams).sampling_params
        return build_image_segment(diff_outputs, expected_sigmas=diff_params.sigmas)

    def build_decoded(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Any:
        del sample
        _, _, pil_images = collect_dit_outputs(
            per_request, final_output_type=self.final_output_type, stage_id=self.stage_id, modality=self.modality
        )
        return pils_to_images(pil_images)

    def build_conditions(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Dict[str, Any]:
        """Ship the PROMPTS (deferred conditions) for trainer-side KV rebuild."""
        del per_request
        turns = sample.text_conditioning()
        if len(turns) != 1:
            raise ValueError(f"{self.modality}.build_conditions: expected exactly one text turn, got {len(turns)}")
        texts = turns[0].content
        gen_part = sample.frontier_gen_part(DiffusionSamplingParams)
        diff_params = gen_part.sampling_params
        image_shape = (int(diff_params.height), int(diff_params.width))
        prompts = list(texts.texts)
        if len(prompts) != len(gen_part.sample_ids):
            raise RuntimeError(
                f"{self.modality}.build_conditions: prompt count {len(prompts)} "
                f"!= diffusion sample count {len(gen_part.sample_ids)}"
            )
        conditions = BagelDiffusionConditions(
            prompts=prompts,
            image_shapes=[image_shape] * len(prompts),
        )
        return conditions.to_dict()


@register_adapter("bagel_t2i")
class BagelT2iAdapter(ModelAdapter):
    """BAGEL-7B-MoT text → image (single diffusion stage, TP=1)."""

    stage_yaml = "bagel_t2i_rl.yaml"
    omni_mode = "text-to-image"
    needs_driver_tokenizer = False

    def __init__(self, config: Any, model_config: Any, *, strategy: Any = None, tokenize_fn: Any = None) -> None:
        super().__init__(config, model_config, strategy=strategy, tokenize_fn=tokenize_fn)
        self.input_adapter = BagelInputAdapter(self.modality)
        self.output_adapter = BagelOutputAdapter(self.modality)

    def schedule_policy(self) -> FlowMatchSchedulePolicy:
        """Static-shift FlowMatch σ policy (BAGEL uses no dynamic shifting)."""
        shift = float(getattr(self.model_config, "shift", 3.0))
        return FlowMatchSchedulePolicy.static_only(shift)

    def validate_request(self, sample: Sample) -> None:
        if sample.has_image_input():
            raise ValueError(
                f"modality={self.modality!r} rejects image-bearing requests; use an image-conditioned modality instead."
            )

    def build_inputs(self, sample: Sample) -> List[GenerateCall]:
        return self.input_adapter.build(sample)

    def build_response(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Sample:
        return self.output_adapter.build(sample, per_request)


__all__ = ["BagelInputAdapter", "BagelOutputAdapter", "BagelT2iAdapter"]
