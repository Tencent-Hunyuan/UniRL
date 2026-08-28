"""SenseNova-U1.5 family: vLLM-Omni T2I request and response adapters."""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from unirl.models.sensenova_u1.conditions import SenseNovaU1Conditions
from unirl.models.sensenova_u1.diffusion import SenseNovaU1DiffusionParams
from unirl.rollout.engine.vllm_omni.adapters.base import ModelAdapter, register_adapter
from unirl.rollout.engine.vllm_omni.adapters.dit import (
    DitInputAdapter,
    DitOutputAdapter,
    _grouped_texts_from_sample,
)
from unirl.rollout.engine.vllm_omni.backends import GenerateCall, OmniRawResult, StageSampling
from unirl.rollout.engine.vllm_omni.utils import collect_dit_outputs
from unirl.sde.runtime import FlowMatchSchedulePolicy
from unirl.types.sample import Sample


class SenseNovaU1InputAdapter(DitInputAdapter):
    """Build one same-prompt batched request per GRPO group."""

    def __init__(self, modality: str, *, model_config: Any) -> None:
        super().__init__(modality)
        self.model_config = model_config

    def build_prompts(self, sample: Sample) -> List[Any]:
        grouped_texts, _ = _grouped_texts_from_sample(
            sample,
            caller=f"{self.modality}.build_prompts",
        )
        return [{"prompt": text} for text in grouped_texts]

    def build_sampling(self, sample: Sample):
        _, samples_per_prompt = _grouped_texts_from_sample(
            sample,
            caller=f"{self.modality}.build_sampling",
        )
        sampling = super().build_sampling(sample)
        params = sample.frontier_gen_part(SenseNovaU1DiffusionParams).sampling_params
        kwargs = sampling[0].kwargs
        kwargs["num_outputs_per_prompt"] = samples_per_prompt

        extra_args = kwargs.setdefault("extra_args", {})
        extra_args.update(
            {
                "batch_size": samples_per_prompt,
                "cfg_scale": float(params.guidance_scale),
                "cfg_norm": str(params.cfg_norm),
                "cfg_interval": [float(v) for v in params.cfg_interval],
                "timestep_shift": float(self.model_config.timestep_shift),
                "t_eps": float(params.t_eps),
                "think": False,
                "trajectory_precision": str(params.trajectory_precision),
            }
        )
        if params.sigmas is not None:
            # OmniDiffusionSamplingParams.sigmas follows Diffusers' T-entry
            # convention; replay needs UniRL's complete T+1 schedule.
            extra_args["unirl_sigmas"] = params.sigmas.detach().to("cpu", dtype=torch.float32).tolist()
        return sampling

    def build(self, sample: Sample) -> List[GenerateCall]:
        """Split distinct prompts because upstream SenseNova batches one prompt only."""
        prompts = self.build_prompts(sample)
        sampling = self.build_sampling(sample)[0]
        samples_per_prompt = int(sampling.kwargs["num_outputs_per_prompt"])
        calls: List[GenerateCall] = []
        for group_index, prompt in enumerate(prompts):
            start = group_index * samples_per_prompt
            end = start + samples_per_prompt
            kwargs = dict(sampling.kwargs)
            extra_args = dict(kwargs.get("extra_args") or {})
            if "initial_noise_batch" in extra_args:
                extra_args["initial_noise_batch"] = extra_args["initial_noise_batch"][start:end]
            if "init_noise_group_ids" in extra_args:
                extra_args["init_noise_group_ids"] = list(extra_args["init_noise_group_ids"][start:end])
            extra_args["sde_seed"] = (int(kwargs.get("seed", 0)) + 1_000_003 * group_index) % (2**31)
            kwargs["extra_args"] = extra_args
            calls.append(
                GenerateCall(
                    prompts=[prompt],
                    sampling=[StageSampling(kind=sampling.kind, kwargs=kwargs)],
                )
            )
        return calls


class SenseNovaU1OutputAdapter(DitOutputAdapter):
    """Rebuild replay-ready prefix-cache conditions from worker captures."""

    _CAPTURE_KEY = "sensenova_u1_capture"

    def build_conditions(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Dict[str, Any]:
        diff_outputs, _, _ = collect_dit_outputs(
            per_request,
            final_output_type=self.final_output_type,
            stage_id=self.stage_id,
            modality=self.modality,
        )
        captures = [(getattr(output, "custom_output", None) or {}).get(self._CAPTURE_KEY) for output in diff_outputs]
        if any(capture is None for capture in captures):
            raise RuntimeError(
                "build_response: SenseNova rollout returned no "
                f"{self._CAPTURE_KEY!r} on DiffusionOutput.custom_output. "
                "Check that RLSenseNovaU1Pipeline was installed by the stage YAML."
            )

        fields = {
            "prompts": [],
            "condition_caches": [],
            "uncondition_caches": [],
            "condition_image_indexes": [],
            "uncondition_image_indexes": [],
            "image_shapes": [],
        }
        for capture in captures:
            for name in fields:
                fields[name].extend(capture[name])

        conditions = SenseNovaU1Conditions(**fields)
        conditions.validate()
        expected = len(sample.frontier_gen_part(SenseNovaU1DiffusionParams).sample_ids)
        if conditions.batch_size != expected:
            raise RuntimeError(
                "build_response: SenseNova condition batch "
                f"{conditions.batch_size} != diffusion sample count {expected}."
            )
        return conditions.to_dict()


@register_adapter("sensenova_u1_t2i")
class SenseNovaU1T2IAdapter(ModelAdapter):
    """SenseNova-U1.5 text-to-image rollout on one vLLM-Omni diffusion stage."""

    stage_yaml = "sensenova_u1_t2i_rl.yaml"
    omni_mode = "text-to-image"
    needs_driver_tokenizer = False

    def __init__(self, config: Any, model_config: Any, *, strategy: Any = None, tokenize_fn: Any = None) -> None:
        super().__init__(config, model_config, strategy=strategy, tokenize_fn=tokenize_fn)
        self.input_adapter = SenseNovaU1InputAdapter(self.modality, model_config=model_config)
        self.output_adapter = SenseNovaU1OutputAdapter(self.modality)

    def schedule_policy(self) -> FlowMatchSchedulePolicy:
        return FlowMatchSchedulePolicy.static_only(float(self.model_config.timestep_shift))

    def validate(self) -> None:
        if self.model_config is None or not hasattr(self.model_config, "timestep_shift"):
            raise ValueError(
                f"SenseNovaU1T2IAdapter requires model_config.timestep_shift; got {type(self.model_config).__name__}."
            )

    def validate_request(self, sample: Sample) -> None:
        if sample.has_image_input():
            raise ValueError(
                f"modality={self.modality!r} rejects image-bearing requests; "
                "the initial integration supports text-to-image only."
            )

    def build_inputs(self, sample: Sample) -> List[GenerateCall]:
        return self.input_adapter.build(sample)

    def build_response(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Sample:
        return self.output_adapter.build(sample, per_request)


__all__ = [
    "SenseNovaU1InputAdapter",
    "SenseNovaU1OutputAdapter",
    "SenseNovaU1T2IAdapter",
]
