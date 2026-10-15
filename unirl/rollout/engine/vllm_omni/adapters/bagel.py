"""BAGEL-7B-MoT rollout adapters for the t2i and it2i modalities."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List

from unirl.config.require import require
from unirl.models.bagel.conditions import BagelDiffusionConditions
from unirl.models.bagel.diffusion import BagelDiffusionParams
from unirl.models.bagel.pipeline import BagelPipeline
from unirl.rollout.engine.vllm_omni.adapters.base import ModelAdapter, register_adapter
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


class BagelAdapter(ModelAdapter):
    """Bind BAGEL t2i and it2i to one single-stage DiT worker."""

    deploy_config = "bagel_t2i_rl.yaml"
    omni_mode = "text-to-image"
    needs_driver_tokenizer = False
    image_input: bool = False  # Whether the modality requires an edit-source image.
    supports_row_canvas = True

    def schedule_policy(self) -> FlowMatchSchedulePolicy:
        """Static-shift FlowMatch σ policy (BAGEL uses no dynamic shifting)."""
        return FlowMatchSchedulePolicy.static_only(float(self.model_config.shift))

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
        conditioning = sample.conditioning()
        require(
            sum(isinstance(v, Texts) for v in conditioning) == 1,
            f"modality={self.modality!r} requires exactly one Texts conditioning batch.",
        )
        if self.image_input:
            require(
                sum(isinstance(v, Images) for v in conditioning) == 1 and len(conditioning) == 2,
                f"modality={self.modality!r} requires exactly one Texts and one Images conditioning batch.",
            )
        else:
            require(len(conditioning) == 1, f"modality={self.modality!r} accepts only Texts conditioning.")

    def build_inputs(self, sample: Sample) -> List[GenerateCall]:
        """One request per sibling run, one call per contiguous requests of equal canvas and size; see the README."""
        gen_part = sample.frontier_gen_part(BagelDiffusionParams)
        params = gen_part.sampling_params
        conditioning = sample.conditioning()
        texts = list(next(v for v in conditioning if isinstance(v, Texts)).texts)
        images = (
            [image.to_pil() for image in next(v for v in conditioning if isinstance(v, Images)).to_list()]
            if self.image_input
            else []
        )
        canvases = sample.canvases()
        should_pack = params.guidance_scale <= 1.0 and params.cfg_img_scale <= 1.0
        owner_ids = gen_part.group_ids if should_pack else gen_part.sample_ids
        run_starts = [row for row in range(len(owner_ids)) if row == 0 or owner_ids[row] != owner_ids[row - 1]]
        run_spans = list(zip(run_starts, run_starts[1:] + [len(owner_ids)]))
        run_shapes = [(canvases[start], end - start) for start, end in run_spans]
        max_tokens = BagelPipeline.latent_shape(model_config=self.model_config, sampling_spec=params)[0]

        num_steps = params.num_inference_steps
        sigmas_list_from_diffusion(params, num_steps)
        sde_indices = sorted(set(params.sde_indices)) if params.sde_indices is not None and params.eta >= 1e-7 else None
        sigma_max = float(params.sigmas[1].item()) if params.sigmas is not None else None

        calls: List[GenerateCall] = []
        call_start = 0
        for call_end in range(1, len(run_spans) + 1):
            if call_end < len(run_spans) and run_shapes[call_end] == run_shapes[call_start]:
                continue
            (height, width), num_outputs = run_shapes[call_start]
            call_part = gen_part.slice(run_spans[call_start][0], run_spans[call_end - 1][1])
            prompts = [
                {"prompt": texts[start], "multi_modal_data": {"image": images[start]}}
                if self.image_input
                else {"prompt": texts[start]}
                for start, _ in run_spans[call_start:call_end]
            ]
            diff_params = params
            if (height, width) != (params.height, params.width):
                diff_params = dataclasses.replace(params, height=height, width=width)
                latent_shape = BagelPipeline.latent_shape(model_config=self.model_config, sampling_spec=diff_params)
                require(
                    0 < latent_shape[0] <= max_tokens,
                    f"{self.modality}: row canvas {height}x{width} must hold at least one token and no more than "
                    f"sampling.height/width ({params.height}x{params.width}), which bounds the stored trajectory.",
                )
                if params.init_noise_latent_shape is not None:
                    diff_params = dataclasses.replace(diff_params, init_noise_latent_shape=list(latent_shape))

            diff_kwargs: Dict[str, Any] = {
                "height": diff_params.height,
                "width": diff_params.width,
                # +1: BAGEL builds linspace(1, 0, num_timesteps) and loops num_timesteps-1.
                "num_inference_steps": num_steps + 1,
                "eta": diff_params.eta,
                "return_trajectory_latents": True,
                "return_trajectory_decoded": False,
                "num_outputs_per_prompt": num_outputs,
            }
            if diff_params.seed is not None:
                diff_kwargs["seed"] = diff_params.seed

            extra_args: Dict[str, Any] = {
                "cfg_text_scale": diff_params.guidance_scale,
                "cfg_img_scale": diff_params.cfg_img_scale,
                "cfg_interval": tuple(diff_params.cfg_interval),
                "cfg_renorm_min": diff_params.cfg_renorm_min,
                "cfg_renorm_type": diff_params.cfg_renorm_type,
                "trajectory_precision": diff_params.trajectory_precision,
            }
            if sde_indices is not None:
                extra_args["sde_indices"] = sde_indices
            if sigma_max is not None:
                extra_args["sigma_max"] = sigma_max
            pack_initial_noise_extra_args(extra_args, call_part, diff_params, caller=self.modality)
            diff_kwargs["extra_args"] = extra_args

            calls.append(
                GenerateCall(
                    prompts=prompts,
                    sampling=[StageSampling(kind=STAGE_KIND_DIFFUSION, kwargs=diff_kwargs)],
                )
            )
            call_start = call_end
        return calls

    def build_response(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Sample:
        """Fill the frontier diffusion Part with trajectory, decoded images, and BAGEL replay conditions."""
        if not per_request or not any(per_request):
            raise ValueError("build_response: empty per-request outputs (Omni.generate returned nothing surfaceable).")
        diff_outputs, frame_groups, pil_images = collect_dit_outputs(
            per_request, final_output_type="image", stage_id=0, modality=self.modality
        )
        frontier = sample.frontier_gen_part(BagelDiffusionParams)
        diff_params = frontier.sampling_params
        max_tokens = BagelPipeline.latent_shape(model_config=self.model_config, sampling_spec=diff_params)[0]
        segment = build_image_segment(diff_outputs, expected_sigmas=diff_params.sigmas, pad_tokens_to=max_tokens)
        conditioning = sample.conditioning()
        prompts = list(next(v for v in conditioning if isinstance(v, Texts)).texts)
        input_images = (
            [image.to_pil() for image in next(v for v in conditioning if isinstance(v, Images)).to_list()]
            if self.image_input
            else []
        )
        conditions = BagelDiffusionConditions(
            prompts=prompts,
            input_images=input_images,
            image_shapes=sample.canvases(),
            layouts=[read_captures(out)["layout"] for out, frames in zip(diff_outputs, frame_groups) for _ in frames],
        )
        return sample.replace_frontier(
            frontier.fill(
                segment=segment,
                primitives={"image": pils_to_images(pil_images)},
                conditions=conditions.to_dict(),
            )
        )


@register_adapter("bagel_t2i")
class BagelT2iAdapter(BagelAdapter):
    """BAGEL-7B-MoT text → image."""


@register_adapter("bagel_it2i")
class BagelIt2iAdapter(BagelAdapter):
    """BAGEL-7B-MoT text + source image → edited image (editing / it2i)."""

    image_input = True


__all__ = ["BagelAdapter", "BagelIt2iAdapter", "BagelT2iAdapter"]
