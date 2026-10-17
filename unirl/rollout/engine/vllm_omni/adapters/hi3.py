"""HunyuanImage-3 family: input/output sub-adapters + the six modality classes."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from unirl.models.hunyuan_image3.conditions import HunyuanImage3FusedMultimodalCondition
from unirl.rollout.engine.vllm_omni.adapters.base import ModelAdapter, register_adapter
from unirl.rollout.engine.vllm_omni.adapters.dit import DitOutputAdapter
from unirl.rollout.engine.vllm_omni.backends import (
    STAGE_KIND_AR,
    STAGE_KIND_DIFFUSION,
    GenerateCall,
    OmniRawResult,
    StageSampling,
)
from unirl.rollout.engine.vllm_omni.pipelines._shared.interception import read_captures
from unirl.rollout.engine.vllm_omni.utils import (
    build_ar_segment,
    collect_dit_outputs,
    decoded_text_from_ar,
    seed_from_sample_id,
)
from unirl.rollout.engine.vllm_omni.utils.diff_kwargs import core_diff_kwargs, sde_extra_args
from unirl.types.primitives import Texts
from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams, DiffusionSamplingParams
from unirl.types.segments.latent import LatentSegment


def trailing_gen_parts(sample: Sample, *params_types: type, caller: str) -> Tuple[Part, ...]:
    """Validate and return the current generation-stage suffix."""
    count = len(params_types)
    if len(sample.parts) < count:
        raise ValueError(f"{caller}: expected {count} trailing generation Parts, got {len(sample.parts)} total Parts")
    start = len(sample.parts) - count
    trailing = tuple(sample.parts[start:])
    for offset, (part, params_type) in enumerate(zip(trailing, params_types)):
        index = start + offset
        if not part.is_gen:
            raise ValueError(f"{caller}: trailing Part at index {index} is not generated")
        if not isinstance(part.sampling_params, params_type):
            raise ValueError(
                f"{caller}: trailing Part at index {index} has sampling_params of type "
                f"{type(part.sampling_params).__name__}, expected {params_type.__name__}"
            )
    return trailing


def hi3_fused_conditions(diff_outputs: List[OmniRawResult], *, modality: str) -> Dict[str, Any]:
    """The HI3 DiT replay conditions — concat the ``fused_mm_capture`` dicts."""
    captures = [read_captures(d).get("fused_mm_capture") for d in diff_outputs]
    if any(c is None for c in captures):
        raise RuntimeError(
            f"build_response: HI3 rollout (modality={modality!r}) "
            "returned no 'fused_mm_capture' on the output envelope's unirl metadata. "
            "Check that RLHunyuanImage3Pipeline.prepare_inputs_for_generation "
            "hook ran in every DiT worker — the subclass swap may not have "
            "taken effect (verify custom_pipeline_args.pipeline_class in "
            "the deploy config)."
        )

    return {
        "fused": HunyuanImage3FusedMultimodalCondition.concat(
            [HunyuanImage3FusedMultimodalCondition.from_dict(c) for c in captures]
        )
    }


def hi3_ar_fused_conditions(per_request: List[List[OmniRawResult]]) -> Dict[str, Any]:
    """The AR Part replay conditions for the recaption producer."""
    rows: List[List[int]] = []
    for outputs in per_request:
        ids = None
        for out in outputs:
            if out.stage_id == 0:
                ids = out.prompt_token_ids
                break
        rows.append(list(ids) if ids else [])

    if not any(rows):
        return {}

    bsz = len(rows)
    max_len = max(len(r) for r in rows)
    input_ids = torch.zeros((bsz, max_len), dtype=torch.long)
    prompt_lengths = torch.zeros((bsz,), dtype=torch.long)
    for b, r in enumerate(rows):
        if r:
            input_ids[b, : len(r)] = torch.tensor(r, dtype=torch.long)
            prompt_lengths[b] = len(r)
    return {"fused": HunyuanImage3FusedMultimodalCondition(input_ids=input_ids, prompt_lengths=prompt_lengths)}


class Hi3InputAdapter:
    """Request ``Sample`` → HI3 AR-bearing :class:`GenerateCall` (one, whole batch)."""

    def __init__(
        self,
        modality: str,
        *,
        tokenize_fn: Optional[Callable[..., List[int]]],
        task_key: str,
        output_modalities: Tuple[str, ...],
        stages: Tuple[str, ...],
        image_input: bool = False,
        carries_target_size: bool = False,
        bot_task_base: Optional[str] = None,
        vanilla_task: Optional[Tuple[str, str]] = None,
        sys_type: str = "en_unified",
    ) -> None:
        self.modality = modality
        self.tokenize_fn = tokenize_fn
        self.task_key = task_key
        self.output_modalities = tuple(output_modalities)
        self.stages = tuple(stages)
        self.image_input = image_input
        self.carries_target_size = carries_target_size
        self.bot_task_base = bot_task_base
        self.vanilla_task = vanilla_task
        self.sys_type = sys_type

    def build(self, sample: Sample) -> List[GenerateCall]:
        caller = f"{type(self).__name__}({self.modality})"
        if "dit" in self.stages:
            ar_part, diff_part = trailing_gen_parts(
                sample,
                ARSamplingParams,
                DiffusionSamplingParams,
                caller=caller,
            )
        elif self.carries_target_size:
            diff_part, ar_part = trailing_gen_parts(
                sample,
                DiffusionSamplingParams,
                ARSamplingParams,
                caller=caller,
            )
        else:
            ar_part, diff_part = sample.frontier_gen_part(ARSamplingParams), None

        control = sample.parts[0].control
        sys_type = control.get("sys_type") or self.sys_type
        bot_task = control.get("bot_task")
        task = self.task_key
        if self.bot_task_base and bot_task:
            if bot_task == "vanilla" and self.vanilla_task is not None:
                task, sys_type = self.vanilla_task
            elif bot_task in ("think", "recaption"):
                task = f"{self.bot_task_base}_{bot_task}"

        if self.image_input:
            turns, images = sample.vision_conditioning()
            texts = next(t.content for t in turns if isinstance(t.content, Texts))
            pil_images = images[0].to_pils()
        else:
            texts = sample.text_conditioning()[0].content
            pil_images = []

        diff_params = diff_part.sampling_params if diff_part is not None else None
        if self.tokenize_fn is None:
            raise RuntimeError(
                "Hi3InputAdapter.build: tokenize_fn not provided (AR modalities need the driver tokenizer)"
            )
        prompts: List[Dict[str, Any]] = []
        for i, text in enumerate(texts.texts):
            token_ids = self.tokenize_fn(text, task=task, sys_type=sys_type)
            entry: Dict[str, Any] = {
                "prompt_token_ids": token_ids,
                "prompt": text,
                "use_system_prompt": sys_type,
                "modalities": list(self.output_modalities),
            }
            if self.image_input:
                pil = pil_images[i]
                entry["multi_modal_data"] = {"image": pil}
                if diff_params is None:
                    entry["height"] = pil.height
                    entry["width"] = pil.width
                else:
                    entry["height"] = diff_params.height
                    entry["width"] = diff_params.width
            elif self.carries_target_size:
                entry["height"] = diff_params.height
                entry["width"] = diff_params.width
            prompts.append(entry)

        ar_params = ar_part.sampling_params
        ar_sampling = StageSampling(
            kind=STAGE_KIND_AR,
            kwargs=dict(
                temperature=ar_params.temperature,
                top_p=ar_params.top_p,
                top_k=ar_params.top_k,
                max_tokens=ar_params.max_new_tokens,
                logprobs=1,
            ),
        )
        if "dit" not in self.stages:
            return [GenerateCall(prompts=prompts, sampling=[ar_sampling])]

        seg = diff_part.segment
        if isinstance(seg, LatentSegment) and seg.initial_latents is not None:
            raise NotImplementedError(
                f"{type(self).__name__}: modality={self.modality!r} cannot consume a "
                f"pre-materialized LatentSegment.initial_latents tensor "
                f"(HI3 DiT latent shape is AR-dynamic). Ship the x_T RECIPE via the "
                f"gen Part's lineage instead."
            )

        base_diff_kwargs = core_diff_kwargs(diff_params)
        seed = diff_params.seed
        if seed is not None:
            base_diff_kwargs["seed"] = seed
        base_extra = sde_extra_args(diff_params)

        explicit_keys = list(diff_part.init_noise_group_ids)
        recipe_keys = (
            []
            if diff_params.disable_driver_xt
            else (explicit_keys or (diff_part.group_ids if diff_params.init_same_noise else list(diff_part.sample_ids)))
        )
        if not recipe_keys:
            diff_kwargs = dict(base_diff_kwargs)
            if base_extra:
                diff_kwargs["extra_args"] = dict(base_extra)
            return [
                GenerateCall(
                    prompts=prompts,
                    sampling=[ar_sampling, StageSampling(kind=STAGE_KIND_DIFFUSION, kwargs=diff_kwargs)],
                )
            ]
        if len(recipe_keys) != len(prompts):
            raise ValueError(
                f"{self.modality}: x_T recipe key count {len(recipe_keys)} != prompt count {len(prompts)}."
            )

        noise_seed = seed if seed is not None else 0
        calls: List[GenerateCall] = []
        for index, prompt in enumerate(prompts):
            diff_kwargs = dict(base_diff_kwargs)
            diff_kwargs["extra_args"] = {
                **base_extra,
                "init_noise_group_ids": [recipe_keys[index]],
                "init_noise_seed": noise_seed,
            }
            calls.append(
                GenerateCall(
                    prompts=[prompt],
                    sampling=[ar_sampling, StageSampling(kind=STAGE_KIND_DIFFUSION, kwargs=diff_kwargs)],
                )
            )
        return calls


class Hi3TextOutputAdapter:
    """Per-request AR results → the filled AR generation Part."""

    def __init__(self, modality: str) -> None:
        self.modality = modality

    def build(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Sample:
        if not per_request or not any(per_request):
            raise ValueError("build_response: empty per-request outputs (Omni.generate returned nothing surfaceable).")
        segment = build_ar_segment(per_request)
        if segment is None:
            return sample
        decoded = decoded_text_from_ar(per_request)
        frontier = sample.frontier_gen_part(ARSamplingParams)
        return sample.replace_frontier(
            frontier.fill(
                segment=segment,
                primitives={"text": decoded},
                conditions={},
            )
        )


class Hi3ImageOutputAdapter(DitOutputAdapter):
    """Two-stage HI3 response: fill AR and diffusion Parts independently."""

    def __init__(self, modality: str) -> None:
        super().__init__(modality, stage_id=1)

    def build_conditions(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Dict[str, Any]:
        del sample
        diff_outputs, _, _ = collect_dit_outputs(
            per_request, final_output_type=self.final_output_type, stage_id=self.stage_id, modality=self.modality
        )
        return hi3_fused_conditions(diff_outputs, modality=self.modality)

    def build(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Sample:
        filled = super().build(sample, per_request)

        ar_segment = build_ar_segment(per_request)
        if ar_segment is None:
            return filled

        ar_decoded = decoded_text_from_ar(per_request)
        ar_conditions = hi3_ar_fused_conditions(per_request)
        ar_part, _ = trailing_gen_parts(
            filled,
            ARSamplingParams,
            DiffusionSamplingParams,
            caller=f"{type(self).__name__}({self.modality}).build",
        )
        idx = len(filled.parts) - 2
        parts = list(filled.parts)
        parts[idx] = ar_part.fill(
            segment=ar_segment,
            primitives={"text": ar_decoded},
            conditions=dict(ar_conditions),
        )
        return filled.with_parts(parts)


class Hi3DitRecaptionOutputAdapter(DitOutputAdapter):
    """Single diffusion-Part response of the standalone HI3 DiT (Stage 0)."""

    def build_conditions(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Dict[str, Any]:
        del sample
        diff_outputs, _, _ = collect_dit_outputs(
            per_request, final_output_type=self.final_output_type, stage_id=self.stage_id, modality=self.modality
        )
        return hi3_fused_conditions(diff_outputs, modality=self.modality)


@register_adapter("hi3_t2i")
class Hi3T2iAdapter(ModelAdapter):
    """HI3 text → AR think → DiT image."""

    deploy_config = "hunyuan_image3_t2i_rl.yaml"
    omni_mode = "text-to-image"
    ar_lora_passthrough = True
    clear_cuda_visible = True

    def __init__(self, config: Any, model_config: Any, *, strategy: Any = None, tokenize_fn: Any = None) -> None:
        super().__init__(config, model_config, strategy=strategy, tokenize_fn=tokenize_fn)
        self.input_adapter = Hi3InputAdapter(
            self.modality,
            tokenize_fn=self.tokenize_fn,
            task_key="t2i_think",
            output_modalities=("image",),
            stages=("ar", "dit"),
            carries_target_size=True,
            bot_task_base="t2i",
            vanilla_task=("t2i_vanilla", "en_vanilla"),
        )
        self.output_adapter = Hi3ImageOutputAdapter(self.modality)

    def validate_request(self, sample: Sample) -> None:
        if sample.has_image_input():
            raise ValueError(
                f"modality={self.modality!r} rejects image-bearing requests; use an image-conditioned modality instead."
            )

    def build_inputs(self, sample: Sample) -> List[GenerateCall]:
        return self.input_adapter.build(sample)

    def build_response(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Sample:
        return self.output_adapter.build(sample, per_request)


@register_adapter("hi3_it2i")
class Hi3It2iAdapter(ModelAdapter):
    """HI3 image+text → AR recaption → DiT edited image."""

    deploy_config = "hunyuan_image3_it2i_rl.yaml"
    omni_mode = "text-to-image"
    ar_lora_passthrough = True
    clear_cuda_visible = True

    def __init__(self, config: Any, model_config: Any, *, strategy: Any = None, tokenize_fn: Any = None) -> None:
        super().__init__(config, model_config, strategy=strategy, tokenize_fn=tokenize_fn)
        self.input_adapter = Hi3InputAdapter(
            self.modality,
            tokenize_fn=self.tokenize_fn,
            task_key="it2i_think",
            output_modalities=("image",),
            stages=("ar", "dit"),
            image_input=True,
            bot_task_base="it2i",
        )
        self.output_adapter = Hi3ImageOutputAdapter(self.modality)

    def validate_request(self, sample: Sample) -> None:
        if not sample.has_image_input():
            raise ValueError(
                f"modality={self.modality!r} requires an image input Part (Images primitive) "
                "chained off the prompt (Part.input_child); none found."
            )

    def build_inputs(self, sample: Sample) -> List[GenerateCall]:
        return self.input_adapter.build(sample)

    def build_response(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Sample:
        return self.output_adapter.build(sample, per_request)


@register_adapter("hi3_i2t")
class Hi3I2tAdapter(ModelAdapter):
    """HI3 image+text → AR text."""

    deploy_config = "hunyuan_image3_i2t_rl.yaml"
    needs_sigmas = False
    ar_lora_passthrough = True
    clear_cuda_visible = True

    def __init__(self, config: Any, model_config: Any, *, strategy: Any = None, tokenize_fn: Any = None) -> None:
        super().__init__(config, model_config, strategy=strategy, tokenize_fn=tokenize_fn)
        self.input_adapter = Hi3InputAdapter(
            self.modality,
            tokenize_fn=self.tokenize_fn,
            task_key="i2t",
            output_modalities=("text",),
            stages=("ar",),
            image_input=True,
        )
        self.output_adapter = Hi3TextOutputAdapter(self.modality)

    def validate_request(self, sample: Sample) -> None:
        if not sample.has_image_input():
            raise ValueError(
                f"modality={self.modality!r} requires an image input Part (Images primitive) "
                "chained off the prompt (Part.input_child); none found."
            )

    def build_inputs(self, sample: Sample) -> List[GenerateCall]:
        return self.input_adapter.build(sample)

    def build_response(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Sample:
        return self.output_adapter.build(sample, per_request)


@register_adapter("hi3_t2t")
class Hi3T2tAdapter(ModelAdapter):
    """HI3 text → AR text."""

    deploy_config = "hunyuan_image3_ar_rl.yaml"
    needs_sigmas = False
    ar_lora_passthrough = True
    clear_cuda_visible = True

    def __init__(self, config: Any, model_config: Any, *, strategy: Any = None, tokenize_fn: Any = None) -> None:
        super().__init__(config, model_config, strategy=strategy, tokenize_fn=tokenize_fn)
        self.input_adapter = Hi3InputAdapter(
            self.modality,
            tokenize_fn=self.tokenize_fn,
            task_key="t2t",
            output_modalities=("text",),
            stages=("ar",),
        )
        self.output_adapter = Hi3TextOutputAdapter(self.modality)

    def validate_request(self, sample: Sample) -> None:
        if sample.has_image_input():
            raise ValueError(
                f"modality={self.modality!r} rejects image-bearing requests; use modality='hi3_i2t' instead."
            )

    def build_inputs(self, sample: Sample) -> List[GenerateCall]:
        return self.input_adapter.build(sample)

    def build_response(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Sample:
        return self.output_adapter.build(sample, per_request)


@register_adapter("hi3_ar_recaption")
class Hi3ArRecaptionAdapter(ModelAdapter):
    """Two-engine trainer's AR think/recaption producer."""

    deploy_config = "hunyuan_image3_ar_recaption_rl.yaml"
    needs_sigmas = False
    ar_lora_passthrough = True
    clear_cuda_visible = True
    lora_copy_transport = True

    def __init__(self, config: Any, model_config: Any, *, strategy: Any = None, tokenize_fn: Any = None) -> None:
        super().__init__(config, model_config, strategy=strategy, tokenize_fn=tokenize_fn)
        self.input_adapter = Hi3InputAdapter(
            self.modality,
            tokenize_fn=self.tokenize_fn,
            task_key="t2i_think",
            output_modalities=("image",),
            stages=("ar",),
            carries_target_size=True,
        )

    def build_inputs(self, sample: Sample) -> List[GenerateCall]:
        return self.input_adapter.build(sample)

    def build_response(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Sample:
        if not per_request or not any(per_request):
            raise ValueError("build_response: empty per-request outputs (Omni.generate returned nothing surfaceable).")
        segment = build_ar_segment(per_request)
        if segment is None:
            return sample
        decoded = decoded_text_from_ar(per_request)
        conditions = hi3_ar_fused_conditions(per_request)
        frontier = sample.frontier_gen_part(ARSamplingParams)
        return sample.replace_frontier(
            frontier.fill(
                segment=segment,
                primitives={"text": decoded},
                conditions=dict(conditions),
            )
        )


@register_adapter("hi3_dit_recaption")
class Hi3DitRecaptionAdapter(ModelAdapter):
    """Standalone HI3 DiT — the two-engine trainer's image half."""

    deploy_config = "hunyuan_image3_dit_recaption_rl.yaml"
    omni_mode = "text-to-image"
    clear_cuda_visible = True
    # HI3 two-engine stages are TP>1 — wake-time LoRA re-push must use the byte-copy transport.
    lora_copy_transport = True

    def __init__(self, config: Any, model_config: Any, *, strategy: Any = None, tokenize_fn: Any = None) -> None:
        super().__init__(config, model_config, strategy=strategy, tokenize_fn=tokenize_fn)
        self.output_adapter = Hi3DitRecaptionOutputAdapter(self.modality)

    def build_inputs(self, sample: Sample) -> List[GenerateCall]:
        if sample.has_image_input():
            raise ValueError(f"modality={self.modality!r} does not accept an image input Part")

        turns = sample.text_conditioning()
        texts = turns[0].content
        cot = turns[1].content
        gen_part = sample.frontier_gen_part(DiffusionSamplingParams)
        diff_params = gen_part.sampling_params
        sys_type = sample.parts[0].control.get("sys_type") or "en_unified"

        base_kwargs = core_diff_kwargs(diff_params)
        height = base_kwargs["height"]
        width = base_kwargs["width"]

        base_extra = sde_extra_args(diff_params)
        recipe_gids = gen_part.group_ids if diff_params.init_same_noise else list(gen_part.sample_ids)
        if recipe_gids and not diff_params.disable_driver_xt:
            base_extra["init_noise_group_ids"] = recipe_gids
            base_extra["init_noise_seed"] = diff_params.seed if diff_params.seed is not None else 0

        calls: List[GenerateCall] = []
        for idx, (sample_id, text, recap) in enumerate(zip(gen_part.sample_ids, texts.texts, cot.texts)):
            prompt = {
                "prompt": text,
                "height": height,
                "width": width,
                "use_system_prompt": sys_type,
                "extra": {"ar_generated_text": recap},
            }
            kwargs = dict(base_kwargs)
            kwargs["seed"] = seed_from_sample_id(sample_id)
            extra_args = dict(base_extra)
            if "init_noise_group_ids" in extra_args:
                extra_args["init_noise_group_ids"] = [recipe_gids[idx]]
            if extra_args:
                kwargs["extra_args"] = extra_args
            calls.append(
                GenerateCall(
                    prompts=[prompt],
                    sampling=[StageSampling(kind=STAGE_KIND_DIFFUSION, kwargs=kwargs)],
                    group_by_request_id=False,
                )
            )
        return calls

    def build_response(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Sample:
        return self.output_adapter.build(sample, per_request)


__all__ = [
    "Hi3ArRecaptionAdapter",
    "Hi3DitRecaptionAdapter",
    "Hi3DitRecaptionOutputAdapter",
    "Hi3I2tAdapter",
    "Hi3ImageOutputAdapter",
    "Hi3InputAdapter",
    "Hi3It2iAdapter",
    "Hi3T2iAdapter",
    "Hi3T2tAdapter",
    "Hi3TextOutputAdapter",
    "hi3_ar_fused_conditions",
    "hi3_fused_conditions",
]
