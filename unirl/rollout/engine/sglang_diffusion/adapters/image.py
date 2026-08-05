"""``ImageAdapter`` — per-output-shape base adapter holding the conversion logic.

Output shape = a 5-D image-form latent trajectory ``[B, T+1, C, H, W]`` decoded to
``Images``. Holds ``build_inputs`` / ``build_response`` once and exposes the per-model
variation points as overridable stages (request side: ``build_prompts``,
``build_sampling``; response side: ``build_segment``, ``build_decoded``,
``build_condition``) and the ``segment_factory`` class knob.
Concrete adapters override only the stages that differ — no sub-step hooks below
a stage.

Convention: the ``build_*`` stages are the overridable variation points;
``build_inputs`` / ``build_response`` are sealed templates that own validation,
the engine pins, and merge order — override the stages, not the templates.

Ported from the old engine's ``request.py`` / ``response.py`` free functions, with
the model-family branches lifted into overridable methods.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from unirl.config.require import require
from unirl.rollout.engine.sglang_diffusion import utils
from unirl.rollout.engine.sglang_diffusion.adapters.base import ModelAdapter
from unirl.rollout.engine.sglang_diffusion.backends import RawResult
from unirl.types.primitives import Texts, primitive_modality_key
from unirl.types.sample import Sample
from unirl.types.sampling import DiffusionSamplingParams, is_forward_process
from unirl.types.segments.latent import make_image_segment


class ImageAdapter(ModelAdapter):
    """Conversion for image-output families (SD3, FLUX, …). Default path end-to-end."""

    segment_factory = staticmethod(make_image_segment)
    squeeze_single_frame_4d: bool = True
    pad_mask_to_embeds: bool = False

    def build_inputs(self, sample: Sample, *, initial_noise: Any) -> Dict[str, Any]:
        """Sealed template: validate, then merge the stage payloads in layer order.

        Reads the request off the ``Sample``: the input ``Part``'s text
        ``primitives`` entry and the frontier gen ``Part``'s ``sampling_params`` (the
        ``DiffusionSamplingParams``, σ pinned on ``.sigmas`` by the engine).
        Override the ``build_prompts`` / ``build_sampling`` stages, not this
        method — the validation gates, engine pins, noise wiring, and SDE-kernel
        layer are RL-correctness contracts that must survive any per-model
        variation. Stages return plain kwargs dicts whose keys must stay
        disjoint from the pins; later updates override earlier ones.
        """
        input_part = sample.parts[0]
        gen_part = sample.frontier_gen_part(DiffusionSamplingParams)
        # sealed validation (fail-fast gates survive any stage override)
        text_prim = input_part.primitives.get("text")
        if not isinstance(text_prim, Texts):
            raise TypeError(
                f"build_inputs: input Part.primitives['text'] must be Texts; got "
                f"{type(text_prim).__name__ if text_prim is not None else 'None'}"
            )
        require(bool(text_prim.texts), "build_inputs: input Part.primitives['text'] must be non-empty")
        require(bool(gen_part.sample_ids), "build_inputs: gen Part has no samples")

        diffusion = gen_part.sampling_params
        require(
            diffusion is not None,
            "build_inputs: the gen Part must carry diffusion sampling_params",
        )

        require(
            diffusion.sigmas is not None,
            "build_inputs: diffusion.sigmas must be pinned by the engine before "
            "conversion (see SGLangDiffusionRolloutEngine._ensure_sample_sigmas).",
        )
        require(
            int(diffusion.sigmas.shape[0]) == int(diffusion.num_inference_steps) + 1,
            f"build_inputs: sigmas length {int(diffusion.sigmas.shape[0])} != "
            f"num_inference_steps+1 ({int(diffusion.num_inference_steps) + 1}).",
        )

        sampler_kwargs: Dict[str, Any] = dict(diffusion.sampler_kwargs or {})

        neg_prompt = sampler_kwargs.get("negative_prompt")
        return_neg_embeds = bool(sampler_kwargs.get("return_negative_prompt_embeds", False))
        require(
            neg_prompt is None or return_neg_embeds,
            "build_inputs: sampler_kwargs.negative_prompt is set but "
            "return_negative_prompt_embeds is not True — rollout would condition on "
            "the negative prompt while replay uses zero negative embeds (silent GRPO "
            "ratio mismatch). Set return_negative_prompt_embeds=True.",
        )

        sde_indices_raw = diffusion.sde_indices
        sde_indices = sorted(int(v) for v in sde_indices_raw) if sde_indices_raw is not None else None

        kwargs: Dict[str, Any] = dict(sampler_kwargs)
        kwargs.update(self.build_prompts(sample))
        kwargs.update(self.build_sampling(sample, diffusion=diffusion))
        kwargs.update(
            {
                "save_output": False,
                "return_file_paths_only": False,
                "return_trajectory_latents": True,
                "return_trajectory_decoded": False,
            }
        )

        # Request negative prompt embeds only when CFG is active.
        if self.cfg.populate_conditions:
            kwargs["return_prompt_embeds"] = True
            if float(diffusion.guidance_scale) > 1.0 and neg_prompt is not None:
                kwargs["return_negative_prompt_embeds"] = True

        # Key per-step SDE noise by sample ID to preserve within-group exploration.
        if initial_noise is not None:
            kwargs["initial_noise"] = initial_noise
            if gen_part.sample_ids:
                kwargs["denoise_seeds"] = [str(sid) for sid in gen_part.sample_ids]

        # RawResult must use dit_trajectory; trajectory_latents omits x_T and is not output-sliced.
        kwargs["rollout"] = True
        kwargs["rollout_return_dit_trajectory"] = True
        kwargs["return_trajectory_latents"] = True
        if is_forward_process(sde_indices):
            kwargs["rollout_sde_type"] = "ode"
            kwargs["rollout_noise_level"] = float(diffusion.eta)
            kwargs["rollout_log_prob_no_const"] = True
            kwargs["rollout_sde_step_indices"] = []
        else:
            require(
                self._sde_label is not None,
                "build_inputs: SDE mode requires an sde_label (resolved from the strategy)",
            )
            kwargs["rollout_sde_type"] = self._sde_label
            kwargs["rollout_noise_level"] = float(diffusion.eta)
            kwargs["rollout_sde_step_indices"] = sde_indices

        return kwargs

    def build_prompts(self, sample: Sample) -> Dict[str, Any]:
        """Prompt payload: unique prompts + repeat count, recovered by lineage.

        Reconstructs the per-gen-sample prompt from the input ``Part`` (each gen
        sample's parent id → its prompt) and collapses it to unique +
        ``num_outputs_per_prompt`` via the group structure — robust to partial
        forward-batch chunks (a chunk may hold a partial group).
        """
        gen_part = sample.frontier_gen_part(DiffusionSamplingParams)
        prompts = list(sample.text_conditioning()[0].content.texts)
        unique_prompts, k = utils.deexpand_prompts_from_groups(prompts, list(gen_part.group_ids))
        out: Dict[str, Any] = {
            "prompt": unique_prompts if len(unique_prompts) > 1 else unique_prompts[0],
        }
        if k > 1:
            out["num_outputs_per_prompt"] = k
        return out

    def build_sampling(self, sample: Sample, *, diffusion: Any) -> Dict[str, Any]:
        """Sampling scalars + the σ schedule slice.

        ``diffusion.sigmas`` is length T+1 (terminal 0 included); SGLang's
        ``set_timesteps`` wants the interior T. Pass the seed even when
        ``initial_noise`` pins x_T verbatim: SGLang derives per-step SDE noise
        deterministically (its ``_make_step_generators``, keyed on
        ``denoise_seeds``) only when seed is not None — a None seed silently
        falls back to global RNG.
        """
        return {
            "num_inference_steps": int(diffusion.num_inference_steps),
            "guidance_scale": float(diffusion.guidance_scale),
            "height": int(diffusion.height),
            "width": int(diffusion.width),
            "num_frames": int(diffusion.num_frames),
            "sigmas": diffusion.sigmas.detach().cpu().tolist()[:-1],
            "seed": int(diffusion.seed) if diffusion.seed is not None else 0,
        }

    def build_response(self, sample: Sample, raw: List[RawResult]) -> Sample:
        require(bool(raw), "build_response: SGLang returned no results")
        gen_part = sample.frontier_gen_part(DiffusionSamplingParams)
        diffusion = gen_part.sampling_params
        require(
            diffusion is not None and diffusion.sigmas is not None,
            "build_response: diffusion.sigmas must be set",
        )

        num_steps = int(diffusion.num_inference_steps)
        sde_indices_raw = diffusion.sde_indices
        sde_indices = sorted(int(v) for v in sde_indices_raw) if sde_indices_raw is not None else None
        emit_native_logprob = not is_forward_process(sde_indices)

        segment = self.build_segment(
            sample,
            raw,
            num_steps=num_steps,
            sde_indices=sde_indices,
            emit_native_logprob=emit_native_logprob,
        )
        decoded = self.build_decoded(sample, raw)

        conditions: Dict[str, Any] = {}
        if self.cfg.populate_conditions:
            conditions = self.build_condition(raw)

        primitives = {primitive_modality_key(decoded): decoded} if decoded is not None else {}
        return sample.with_filled_frontier(segment=segment, primitives=primitives, conditions=conditions)

    def build_segment(
        self,
        sample: Sample,
        results: List[RawResult],
        *,
        num_steps: int,
        sde_indices: Optional[List[int]],
        emit_native_logprob: bool,
    ):
        """Latent-trajectory stage: collect, gate the 5-D image-form shape, assemble.

        Image-form families keep latents 5-D throughout; packed-token families
        (FLUX.2-Klein, Qwen-Image) override this stage to unpack their packed
        trajectory to image form first.
        """
        traj = utils.collect_trajectory_latents(results)
        if traj.ndim != 5:
            raise ValueError(
                f"{self.model_family}: expected a 5-D image-form trajectory "
                f"[B, T+1, C, H, W]; got rank {traj.ndim}, shape {tuple(traj.shape)}. "
                f"Packed-trajectory families override build_segment."
            )
        return utils.build_latent_segment(
            traj,
            results=results,
            expected_sigmas=sample.frontier_gen_part(DiffusionSamplingParams).sampling_params.sigmas,
            num_steps=num_steps,
            sde_indices=sde_indices,
            emit_native_logprob=emit_native_logprob,
            segment_factory=self.segment_factory,
        )

    def build_decoded(self, sample: Sample, results: List[RawResult]):
        return utils.stack_decoded_images(results, squeeze_single_frame_4d=self.squeeze_single_frame_4d)

    def build_condition(self, results: List[RawResult]) -> Dict[str, Any]:
        text_cond, neg_text_cond = utils.fuse_text_conditions(results, allow_mask_pad=self.pad_mask_to_embeds)
        out: Dict[str, Any] = {}
        if text_cond is not None:
            out["text"] = text_cond
        if neg_text_cond is not None:
            out["negative_text"] = neg_text_cond
        return out


__all__ = ["ImageAdapter"]
