"""Trainside (in-process) rollout engine adapter."""

from __future__ import annotations

import threading
from dataclasses import replace
from typing import ClassVar, Dict, List, Optional, Sequence

import torch

from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.models.types.diffusion import DiffusionStage
from unirl.models.types.per_sample_replay import stamp_sde_index_per_sample
from unirl.models.types.pipeline import Pipeline
from unirl.rollout.engine.base import BaseRolloutEngine
from unirl.sde.runtime import FlowMatchSchedulePolicy, ensure_sample_sigmas
from unirl.types.sample import Part, Sample
from unirl.types.sampling import DiffusionSamplingParams


class TrainsideRolloutEngine(BaseRolloutEngine):
    """In-process rollout engine: the train actor's Pipeline IS the sampler."""

    _component_name = "trainside"
    supports_per_sample_sde_index: ClassVar[bool] = True

    def __init__(
        self,
        *,
        pipeline: Pipeline,
        stage_attrs: Sequence[str] = ("diffusion",),
        forward_batch_size: Optional[int] = None,
    ) -> None:
        self.pipeline = pipeline
        stages = [getattr(pipeline, a) for a in stage_attrs]
        self.stages = stages
        self.models = [s.trainable_module() for s in stages]
        if forward_batch_size is not None and forward_batch_size < 1:
            raise ValueError(
                f"TrainsideRolloutEngine.forward_batch_size must be >= 1 when set; got {forward_batch_size!r}"
            )
        self.forward_batch_size = forward_batch_size
        if any(isinstance(s, DiffusionStage) for s in stages):
            if hasattr(pipeline, "build_schedule_policy"):
                self.schedule_policy = pipeline.build_schedule_policy()
            else:
                self.schedule_policy = FlowMatchSchedulePolicy.from_pretrained(
                    pipeline.bundle.pretrained_path,
                    shift=pipeline.shift,
                )
        else:
            self.schedule_policy = None

        self.lock = threading.Lock()
        self.closed = False

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def generate(self, sample: Sample) -> Sample:
        """Generate one whole DP shard synchronously."""
        with self.lock:
            if self.closed:
                raise RuntimeError("TrainsideRolloutEngine.generate called after shutdown")
            sample.reject_row_canvases(engine="TrainsideRolloutEngine")
            if self.forward_batch_size is not None:
                gen_parts = [p for p in sample.parts if p.is_gen]
                if len(gen_parts) > 1:
                    raise ValueError(
                        "TrainsideRolloutEngine.forward_batch_size chunks pipeline.generate, which "
                        f"fills EVERY gen Part; this Sample has {len(gen_parts)} "
                        f"({[type(p.sampling_params).__name__ for p in gen_parts]}). Chunking would "
                        "re-run the interior stage(s) once per chunk, keep only the frontier, and "
                        "return the earlier stages as empty shells. Unset forward_batch_size, or "
                        "chunk inside the stage that needs it (as the PE recipes do via the SGLang "
                        "diffusion sub-engine's own forward_batch_size)."
                    )
            if self.schedule_policy is not None:
                ensure_sample_sigmas(sample, self.schedule_policy)
            prev_modes = [m.training for m in self.models]
            for m in self.models:
                m.eval()
            try:
                with torch.no_grad():
                    fbs = self.forward_batch_size
                    gen = sample.parts[-1]
                    bs = gen.batch_size
                    single_sde_index = None
                    if gen.sde_index_per_sample is not None:
                        steps = gen.sde_index_per_sample.unique().tolist()
                        if len(steps) == 1:
                            single_sde_index = steps[0]
                            if not isinstance(gen.sampling_params, DiffusionSamplingParams):
                                raise ValueError("Per-sample SDE requests require DiffusionSamplingParams.")
                            if not 0 <= single_sde_index < gen.sampling_params.num_inference_steps - 1:
                                raise ValueError(
                                    "Per-sample SDE steps must be in [0, num_inference_steps-2]; "
                                    "narrow the scheduler candidate pool."
                                )
                            gen.sampling_params = replace(gen.sampling_params, sde_indices=[single_sde_index])
                    if gen.sde_index_per_sample is not None and single_sde_index is None:
                        out = self._generate_per_sample_sde(sample)
                    elif fbs is None or bs <= fbs:
                        out = self.pipeline.generate(sample)
                    else:
                        input_parts = sample.parts[:-1]
                        gen_chunks: List[Part] = []
                        for start in range(0, bs, fbs):
                            end = min(start + fbs, bs)
                            chunk = self.pipeline.generate(Sample(parts=[*input_parts, gen.slice(start, end)]))
                            gen_chunks.append(chunk.parts[-1])
                        out = Sample(parts=[*input_parts, Part.concat(gen_chunks)])
                    if single_sde_index is not None:
                        stamp_sde_index_per_sample(out.parts[-1], single_sde_index)
                        out.parts[-1].sde_index_per_sample = None
            finally:
                for m, mode in zip(self.models, prev_modes):
                    m.train(mode)
            return self._stamp_output_version(out)

    def _generate_per_sample_sde(self, sample: Sample) -> Sample:
        """Group by SDE step, generate one row at a time, then restore request order; see trainer/README.md."""
        if self.forward_batch_size != 1:
            raise ValueError("Per-sample SDE requests require forward_batch_size=1 for FSDP forward lockstep.")
        if len(self.stages) != 1 or not getattr(self.stages[0], "supports_order_independent_sde_noise", False):
            raise ValueError(
                "Per-sample SDE requests require one stage with sample/step-keyed noise; use driver grouping."
            )
        gen = sample.parts[-1]
        params = gen.sampling_params
        if not isinstance(params, DiffusionSamplingParams):
            raise ValueError("Per-sample SDE requests require DiffusionSamplingParams.")
        positions_by_step: Dict[int, List[int]] = {}
        for position, step in enumerate(gen.sde_index_per_sample.tolist()):
            if not 0 <= step < params.num_inference_steps - 1:
                raise ValueError(
                    "Per-sample SDE steps must be in [0, num_inference_steps-2]; narrow the scheduler candidate pool."
                )
            positions_by_step.setdefault(step, []).append(position)
        rows: List[Optional[Part]] = [None] * gen.batch_size
        input_parts = sample.parts[:-1]
        for step in sorted(positions_by_step):
            for position in positions_by_step[step]:
                row = gen.slice(position, position + 1)
                row.sampling_params = replace(params, sde_indices=[step])
                chunk = self.pipeline.generate(Sample(parts=[*input_parts, row]))
                output = chunk.parts[-1]
                stamp_sde_index_per_sample(output, step)
                output.sde_index_per_sample = None
                rows[position] = output
        return Sample(parts=[*input_parts, Part.concat(rows)])

    def shutdown(self) -> None:
        with self.lock:
            self.closed = True

    def health_check(self) -> bool:
        return self.pipeline is not None and all(m is not None for m in self.models)


__all__ = ["TrainsideRolloutEngine"]
