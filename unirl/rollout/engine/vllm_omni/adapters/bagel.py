"""BAGEL-7B-MoT input/output adapters for the t2i and it2i modalities."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any, Dict, List

from unirl.config.require import require
from unirl.models.bagel.conditions import BagelDiffusionConditions
from unirl.models.bagel.diffusion import BagelDiffusionParams
from unirl.models.bagel.pipeline import BagelPipeline
from unirl.rollout.engine.vllm_omni.adapters.base import ModelAdapter, register_adapter
from unirl.rollout.engine.vllm_omni.adapters.dit import DitOutputAdapter
from unirl.rollout.engine.vllm_omni.backends import (
    STAGE_KIND_DIFFUSION,
    GenerateCall,
    OmniRawResult,
    StageSampling,
)
from unirl.rollout.engine.vllm_omni.pipelines._shared.interception import read_captures
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
    n_samples = len(sample.frontier_gen_part(BagelDiffusionParams).sample_ids)
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


class BagelInputAdapter:
    """Build BAGEL prompt dictionaries and diffusion-stage sampling intent."""

    def __init__(self, modality: str, *, image_input: bool = False, model_config: Any) -> None:
        self.modality = modality
        self.image_input = bool(image_input)
        self.model_config = model_config

    def _canvas_params(self, params: BagelDiffusionParams, height: int, width: int) -> BagelDiffusionParams:
        """The request's params re-targeted at a row canvas, x_T shape included; the default canvas returns them."""
        if (height, width) == (params.height, params.width):
            return params
        shape = BagelPipeline.latent_shape(
            model_config=self.model_config, sampling_spec=SimpleNamespace(height=height, width=width)
        )
        require(
            0 < shape[0] <= BagelPipeline.latent_shape(model_config=self.model_config, sampling_spec=params)[0],
            f"{self.modality}: row canvas {height}x{width} must hold at least one token and no more than "
            f"sampling.height/width ({params.height}x{params.width}), which bounds the stored trajectory.",
        )
        noise_shape = None if params.init_noise_latent_shape is None else list(shape)
        return dataclasses.replace(params, height=height, width=width, init_noise_latent_shape=noise_shape)

    def build(self, sample: Sample) -> List[GenerateCall]:
        """One request per sibling run, one call per contiguous requests of equal canvas and size; see the README."""
        gen_part = sample.frontier_gen_part(BagelDiffusionParams)
        params = gen_part.sampling_params
        texts, images = _conditioning_rows(sample, image_input=self.image_input, caller=f"{self.modality}.build")
        canvases = sample.canvases()
        packs = params.guidance_scale <= 1.0 and params.cfg_img_scale <= 1.0
        owners = gen_part.group_ids if packs else gen_part.sample_ids
        starts = [row for row in range(len(owners)) if row == 0 or owners[row] != owners[row - 1]]
        requests = list(zip(starts, starts[1:] + [len(owners)]))
        shapes = [(canvases[start], end - start) for start, end in requests]
        calls: List[GenerateCall] = []
        first = 0
        for last in range(1, len(requests) + 1):
            if last < len(requests) and shapes[last] == shapes[first]:
                continue
            canvas, outputs = shapes[first]
            rows = gen_part.slice(requests[first][0], requests[last - 1][1])
            prompts = [
                {"prompt": texts[start], "multi_modal_data": {"image": images[start]}}
                if self.image_input
                else {"prompt": texts[start]}
                for start, _ in requests[first:last]
            ]
            sampling = self._sampling(rows, self._canvas_params(params, *canvas), outputs)
            calls.append(GenerateCall(prompts=prompts, sampling=sampling))
            first = last
        return calls

    def _sampling(self, gen_part: Any, diff_params: BagelDiffusionParams, outputs: int) -> List[StageSampling]:
        """One diffusion-stage intent with the BAGEL-specific kwargs; ``outputs`` images per request."""
        num_steps = int(diff_params.num_inference_steps)
        diff_kwargs: Dict[str, Any] = dict(
            height=int(diff_params.height),
            width=int(diff_params.width),
            # +1: BAGEL builds linspace(1, 0, num_timesteps) and loops num_timesteps-1.
            num_inference_steps=num_steps + 1,
            eta=float(diff_params.eta),
            return_trajectory_latents=True,
            return_trajectory_decoded=False,
            num_outputs_per_prompt=outputs,
        )
        seed = diff_params.seed
        if seed is not None:
            diff_kwargs["seed"] = int(seed)

        # σ contract self-check: the engine-pinned Part schedule for num_steps
        # steps must have num_steps+1 points. We don't send sigmas (BAGEL ignores
        # them), but assert the engine resolved the schedule the worker will loop.
        sigmas_list_from_diffusion(diff_params, num_steps)

        extra_args: Dict[str, Any] = {
            # Translate the canonical field to BAGEL's worker API.
            "cfg_text_scale": float(diff_params.guidance_scale),
            "cfg_img_scale": float(diff_params.cfg_img_scale),
            "cfg_interval": tuple(diff_params.cfg_interval),
            "cfg_renorm_min": float(diff_params.cfg_renorm_min),
            "cfg_renorm_type": str(diff_params.cfg_renorm_type),
        }
        sde_indices = diff_params.sde_indices
        # eta == 0 (deterministic eval) means no step is stochastic. Shipping a
        # non-empty gate anyway makes the worker scheduler raise ("step_index=N is in
        # the SDE gate but eta=0.0"); an absent gate is its documented pure-Euler
        # path, and matches trainside, whose ``diffuse`` gates per-step eta on the same
        # params.eta and simply records no log-probs. FlowSDEStrategy uses 1e-7
        # as the deterministic cutoff; the wire gate must use the same threshold.
        if sde_indices is not None and float(diff_params.eta) >= 1e-7:
            extra_args["sde_indices"] = sorted({int(i) for i in sde_indices})
        if diff_params.sigmas is not None and int(diff_params.sigmas.shape[0]) > 1:
            extra_args["sigma_max"] = float(diff_params.sigmas[1].item())
        extra_args["trajectory_precision"] = diff_params.trajectory_precision

        pack_initial_noise_extra_args(extra_args, gen_part, diff_params, caller=self.modality)
        diff_kwargs["extra_args"] = extra_args

        return [StageSampling(kind=STAGE_KIND_DIFFUSION, kwargs=diff_kwargs)]


class BagelOutputAdapter(DitOutputAdapter):
    """Build one image Part with deferred BAGEL replay conditions."""

    def __init__(self, modality: str, *, image_input: bool = False, model_config: Any) -> None:
        super().__init__(modality)
        self.image_input = bool(image_input)
        self.model_config = model_config

    def build_segment(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Any:
        """The DiT trajectory segment (asserts the σ echo), smaller canvases padded to the default's token count."""
        diff_outputs, _, _ = collect_dit_outputs(
            per_request, final_output_type=self.final_output_type, stage_id=self.stage_id, modality=self.modality
        )
        diff_params = sample.frontier_gen_part(BagelDiffusionParams).sampling_params
        max_tokens = BagelPipeline.latent_shape(model_config=self.model_config, sampling_spec=diff_params)[0]
        return build_image_segment(diff_outputs, expected_sigmas=diff_params.sigmas, pad_tokens_to=max_tokens)

    def build_decoded(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Any:
        del sample
        _, _, pil_images = collect_dit_outputs(
            per_request, final_output_type=self.final_output_type, stage_id=self.stage_id, modality=self.modality
        )
        return pils_to_images(pil_images)

    def build_conditions(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Dict[str, Any]:
        """Ship raw prompt, shape, optional source image and the worker's token layout for trainer-side KV rebuild."""
        diff_outputs, frame_groups, _ = collect_dit_outputs(
            per_request, final_output_type=self.final_output_type, stage_id=self.stage_id, modality=self.modality
        )
        prompts, input_images = _conditioning_rows(
            sample,
            image_input=self.image_input,
            caller=f"{self.modality}.build_conditions",
        )
        conditions = BagelDiffusionConditions(
            prompts=prompts,
            input_images=input_images,
            image_shapes=sample.canvases(),
            layouts=[read_captures(out)["layout"] for out, frames in zip(diff_outputs, frame_groups) for _ in frames],
        )
        return conditions.to_dict()


class BagelAdapter(ModelAdapter):
    """Bind BAGEL t2i and it2i to one single-stage DiT worker."""

    deploy_config = "bagel_t2i_rl.yaml"
    omni_mode = "text-to-image"
    needs_driver_tokenizer = False
    image_input: bool = False  # Whether the modality requires an edit-source image.
    supports_row_canvas = True

    def __init__(self, config: Any, model_config: Any, *, strategy: Any = None, tokenize_fn: Any = None) -> None:
        super().__init__(config, model_config, strategy=strategy, tokenize_fn=tokenize_fn)
        self.input_adapter = BagelInputAdapter(self.modality, image_input=self.image_input, model_config=model_config)
        self.output_adapter = BagelOutputAdapter(self.modality, image_input=self.image_input, model_config=model_config)

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
