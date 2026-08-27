"""BAGEL-7B-MoT input/output adapters for the t2i and it2i modalities."""

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
from unirl.types.primitives import Images, Texts
from unirl.types.sample import Sample
from unirl.types.sampling import DiffusionSamplingParams


def _conditioning_rows(
    sample: Sample,
    *,
    image_input: bool,
    caller: str,
) -> tuple[List[str], List[Any]]:
    """Return frontier-aligned prompt rows and optional source PIL images."""
    conditioning = sample.conditioning()
    text_batches = [value for value in conditioning if isinstance(value, Texts)]
    if len(text_batches) != 1:
        raise ValueError(f"{caller}: expected exactly one Texts conditioning batch, got {len(text_batches)}")

    prompt_rows = list(text_batches[0].texts)
    n_samples = len(sample.frontier_gen_part(DiffusionSamplingParams).sample_ids)
    if len(prompt_rows) != n_samples:
        raise RuntimeError(f"{caller}: prompt count {len(prompt_rows)} != diffusion sample count {n_samples}")

    image_batches = [value for value in conditioning if isinstance(value, Images)]
    if image_input:
        if len(image_batches) != 1:
            raise ValueError(f"{caller}: expected exactly one Images conditioning batch, got {len(image_batches)}")
        image_rows = [image.to_pil() for image in image_batches[0].to_list()]
        if len(image_rows) != n_samples:
            raise RuntimeError(f"{caller}: image count {len(image_rows)} != diffusion sample count {n_samples}")
    else:
        if image_batches:
            raise ValueError(f"{caller}: modality does not accept image conditioning")
        image_rows = []
    return prompt_rows, image_rows


class BagelInputAdapter(DitInputAdapter):
    """Build BAGEL prompt dictionaries and diffusion-stage sampling intent."""

    def __init__(self, modality: str, *, image_input: bool = False) -> None:
        super().__init__(modality)
        self.image_input = bool(image_input)

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
        if self.image_input:
            return False
        if self._spp(sample) <= 1:
            return False
        diff_params = sample.frontier_gen_part(DiffusionSamplingParams).sampling_params
        return float(diff_params.cfg_text_scale) <= 1.0 and float(diff_params.cfg_img_scale) <= 1.0

    def build_prompts(self, sample: Sample) -> List[Any]:
        """Plain ``{"prompt": text}`` dicts (no ``modalities`` → image path)."""
        caller = f"{self.modality}.build_prompts"
        prompt_rows, pil_images = _conditioning_rows(
            sample,
            image_input=self.image_input,
            caller=caller,
        )
        gen_part = sample.frontier_gen_part(DiffusionSamplingParams)
        n_samples = len(gen_part.sample_ids)
        if self.image_input:
            return [
                {"prompt": text, "multi_modal_data": {"image": image}}
                for text, image in zip(prompt_rows, pil_images, strict=True)
            ]
        spp = self._spp(sample)
        grouped_texts, grouped_spp = _grouped_texts_from_sample(
            sample,
            caller=caller,
        )
        if grouped_spp != spp:
            raise RuntimeError(
                f"{self.modality}.build_prompts: inconsistent samples_per_prompt "
                f"({grouped_spp} from grouping, {spp} from diffusion params)."
            )

        pack = self._is_packable_t2i(sample)
        if pack:
            prompt_texts = grouped_texts
            num_outputs_per_prompt = spp
        else:
            prompt_texts = prompt_rows
            num_outputs_per_prompt = 1

        if len(prompt_texts) * num_outputs_per_prompt != n_samples:
            raise RuntimeError(
                f"{self.modality}.build_prompts: prompt count {len(prompt_texts)} * "
                f"num_outputs_per_prompt={num_outputs_per_prompt} != diffusion sample count {n_samples}."
            )
        return [{"prompt": text} for text in prompt_texts]

    def build_sampling(self, sample: Sample) -> List[StageSampling]:
        """One diffusion-stage intent with the BAGEL-specific kwargs."""
        spp = self._spp(sample)
        gen_part = sample.frontier_gen_part(DiffusionSamplingParams)
        diff_params = gen_part.sampling_params
        pack = self._is_packable_t2i(sample)

        n_samples = len(gen_part.sample_ids)
        if self.image_input:
            n_prompts = n_samples
        else:
            grouped_texts, grouped_spp = _grouped_texts_from_sample(
                sample,
                caller=f"{self.modality}.build_sampling",
            )
            if grouped_spp != spp:
                raise RuntimeError(
                    f"{self.modality}.build_sampling: inconsistent samples_per_prompt "
                    f"({grouped_spp} from grouping, {spp} from diffusion params)."
                )
            n_prompts = len(grouped_texts) if pack else n_samples
        num_outputs_per_prompt = spp if pack else 1
        if n_prompts * num_outputs_per_prompt != n_samples:
            raise RuntimeError(
                f"{self.modality}.build_sampling: prompt count {n_prompts} * "
                f"num_outputs_per_prompt={num_outputs_per_prompt} != diffusion sample count {n_samples}."
            )

        num_steps = int(diff_params.num_inference_steps)
        diff_kwargs: Dict[str, Any] = dict(
            height=int(diff_params.height),
            width=int(diff_params.width),
            # +1: BAGEL builds linspace(1, 0, num_timesteps) and loops num_timesteps-1.
            num_inference_steps=num_steps + 1,
            eta=float(diff_params.eta),
            return_trajectory_latents=True,
            return_trajectory_decoded=False,
            num_outputs_per_prompt=num_outputs_per_prompt,
        )
        seed = getattr(diff_params, "seed", None)
        if seed is not None:
            diff_kwargs["seed"] = int(seed)

        # σ contract self-check: the engine-pinned Part schedule for num_steps
        # steps must have num_steps+1 points. We don't send sigmas (BAGEL ignores
        # them), but assert the engine resolved the schedule the worker will loop.
        sigmas_list_from_diffusion(diff_params, num_steps)

        extra_args: Dict[str, Any] = {
            "cfg_text_scale": float(getattr(diff_params, "cfg_text_scale", 1.0)),
            "cfg_img_scale": float(getattr(diff_params, "cfg_img_scale", 1.0)),
            "cfg_interval": tuple(getattr(diff_params, "cfg_interval", (0.0, 1.0))),
            "cfg_renorm_min": float(getattr(diff_params, "cfg_renorm_min", 0.0)),
            "cfg_renorm_type": str(getattr(diff_params, "cfg_renorm_type", "global")),
        }
        sde_indices = getattr(diff_params, "sde_indices", None)
        # eta == 0 (deterministic eval) means no step is stochastic. Shipping a
        # non-empty gate anyway makes the worker scheduler raise ("step_index=N is in
        # the SDE gate but eta=0.0"); an absent gate is its documented pure-Euler
        # path, and matches trainside, whose ``diffuse`` gates per-step eta on the same
        # params.eta and simply records no log-probs. FlowSDEStrategy uses 1e-7
        # as the deterministic cutoff; the wire gate must use the same threshold.
        if sde_indices is not None and float(getattr(diff_params, "eta", 0.0)) >= 1e-7:
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
    """Build one image Part with deferred BAGEL replay conditions."""

    def __init__(self, modality: str, *, image_input: bool = False) -> None:
        super().__init__(modality)
        self.image_input = bool(image_input)

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
        """Ship raw prompt, shape, and optional source image for trainer-side KV rebuild."""
        del per_request
        prompts, input_images = _conditioning_rows(
            sample,
            image_input=self.image_input,
            caller=f"{self.modality}.build_conditions",
        )
        gen_part = sample.frontier_gen_part(DiffusionSamplingParams)
        diff_params = gen_part.sampling_params
        image_shape = (int(diff_params.height), int(diff_params.width))
        conditions = BagelDiffusionConditions(
            prompts=prompts,
            input_images=input_images,
            image_shapes=[image_shape] * len(prompts),
        )
        return conditions.to_dict()


class BagelAdapter(ModelAdapter):
    """Bind BAGEL t2i and it2i to one single-stage DiT worker."""

    stage_yaml = "bagel_t2i_rl.yaml"
    omni_mode = "text-to-image"
    needs_driver_tokenizer = False
    image_input: bool = False  # Whether the modality requires an edit-source image.

    def __init__(self, config: Any, model_config: Any, *, strategy: Any = None, tokenize_fn: Any = None) -> None:
        super().__init__(config, model_config, strategy=strategy, tokenize_fn=tokenize_fn)
        self.input_adapter = BagelInputAdapter(self.modality, image_input=self.image_input)
        self.output_adapter = BagelOutputAdapter(self.modality, image_input=self.image_input)

    def schedule_policy(self) -> FlowMatchSchedulePolicy:
        """Static-shift FlowMatch σ policy (BAGEL uses no dynamic shifting)."""
        shift = float(getattr(self.model_config, "shift", 3.0))
        return FlowMatchSchedulePolicy.static_only(shift)

    def validate_request(self, sample: Sample) -> None:
        has_image = sample.has_image_input()
        if self.image_input and not has_image:
            raise ValueError(
                f"modality={self.modality!r} requires image conditioning (the edit source); "
                "use modality='bagel_t2i' for prompt-only generation."
            )
        if not self.image_input and has_image:
            raise ValueError(
                f"modality={self.modality!r} rejects image-bearing requests; use modality='bagel_it2i' instead."
            )

    def build_inputs(self, sample: Sample) -> List[GenerateCall]:
        return self.input_adapter.build(sample)

    def build_response(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Sample:
        return self.output_adapter.build(sample, per_request)


@register_adapter("bagel_t2i")
class BagelT2iAdapter(BagelAdapter):
    """BAGEL-7B-MoT text → image."""


@register_adapter("bagel_it2i")
class BagelIt2iAdapter(BagelAdapter):
    """BAGEL-7B-MoT text + source image → edited image (editing / it2i)."""

    image_input = True


__all__ = ["BagelAdapter", "BagelInputAdapter", "BagelIt2iAdapter", "BagelOutputAdapter", "BagelT2iAdapter"]
