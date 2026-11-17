"""Trainside (in-process) rollout engine adapter."""

from __future__ import annotations

import threading
from contextlib import nullcontext
from typing import List, Optional, Sequence

import torch

from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.models.types.diffusion import DiffusionStage
from unirl.models.types.pipeline import Pipeline
from unirl.rollout.engine.base import BaseRolloutEngine
from unirl.sde.runtime import FlowMatchSchedulePolicy, ensure_sample_sigmas
from unirl.types.sample import Part, Sample


class TrainsideRolloutEngine(BaseRolloutEngine):
    """In-process rollout engine: the train actor's Pipeline IS the sampler."""

    _component_name = "trainside"

    def __init__(
        self,
        *,
        pipeline: Pipeline,
        stage_attrs: Sequence[str] = ("diffusion",),
        forward_batch_size: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.pipeline = pipeline
        stages = [getattr(pipeline, a) for a in stage_attrs]
        self.models = [s.trainable_module() for s in stages]
        if forward_batch_size is not None and forward_batch_size < 1:
            raise ValueError(
                f"TrainsideRolloutEngine.forward_batch_size must be >= 1 when set; got {forward_batch_size!r}"
            )
        self.forward_batch_size = forward_batch_size
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError("TrainsideRolloutEngine.seed must be an integer when set.")
        self.seed = seed
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
                devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
                rng_scope = torch.random.fork_rng(devices=devices) if self.seed is not None else nullcontext()
                with torch.no_grad(), rng_scope:
                    if self.seed is not None:
                        rollout_ids = {row["rollout_id"] for row in sample.parts[0].metadata}
                        if len(rollout_ids) != 1:
                            raise ValueError("Seeded trainside generation requires one rollout_id per request.")
                        rank = self.rank_info
                        seed = self.seed + rollout_ids.pop() * rank.dp_size + rank.dp_rank
                        torch.random.default_generator.manual_seed(seed)
                        if devices:
                            torch.cuda.manual_seed(seed)
                    fbs = self.forward_batch_size
                    gen = sample.parts[-1]
                    bs = gen.batch_size
                    if fbs is None or bs <= fbs:
                        out = self.pipeline.generate(sample)
                    else:
                        input_parts = sample.parts[:-1]
                        gen_chunks: List[Part] = []
                        for start in range(0, bs, fbs):
                            end = min(start + fbs, bs)
                            chunk = self.pipeline.generate(Sample(parts=[*input_parts, gen.slice(start, end)]))
                            gen_chunks.append(chunk.parts[-1])
                        out = Sample(parts=[*input_parts, Part.concat(gen_chunks)])
            finally:
                for m, mode in zip(self.models, prev_modes):
                    m.train(mode)
            return self._stamp_output_version(out)

    def shutdown(self) -> None:
        with self.lock:
            self.closed = True

    def health_check(self) -> bool:
        return self.pipeline is not None and all(m is not None for m in self.models)


__all__ = ["TrainsideRolloutEngine"]
