"""Trainside (in-process) rollout engine adapter.

Wraps a materialized ``models`` :class:`Pipeline` plus the trainable
stage, and exposes them as a :class:`BaseRolloutEngine`.  Used in
direct-sampling mode where the training model IS the sampler (on-policy
RL) and rollout runs in the same Python process as training — so no
worker subprocess and no weight sync are needed.
"""

from __future__ import annotations

import threading
from typing import List, Optional, Sequence, Union

import torch

from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.models.types.ar import ARStage
from unirl.models.types.diffusion import DiffusionStage
from unirl.models.types.pipeline import Pipeline
from unirl.rollout.engine.synchronous import SyncRolloutEngine
from unirl.sde.runtime import FlowMatchSchedulePolicy, ensure_sample_sigmas
from unirl.types.sample import Part, Sample

Stage = Union[DiffusionStage, ARStage]


class TrainsideRolloutEngine(SyncRolloutEngine):
    """In-process rollout engine: the train actor's Pipeline IS the sampler.

    Args:
        pipeline: A materialized ``models`` pipeline whose
            ``generate(sample)`` fills the request ``Sample``'s gen Parts.
        stage: Optional pre-resolved trainable stage whose
            ``trainable_module()`` is the FSDP-wrapped model (the v1 train
            actor passes one). Takes precedence over ``stage_attrs``.
        stage_attrs: Stage attribute(s) to read off ``pipeline`` and
            eval-scope around ``generate``. A list so composed pipelines can
            drive more than one trainable module (e.g. PE's
            ``["diffusion", "ar"]``); defaults to ``("diffusion",)`` for the
            common single-diffusion engine.
        forward_batch_size: Optional intra-call chunk size for the
            ``pipeline.generate`` forward path. When set and the gen frontier
            exceeds this, ``generate`` slices the frontier Part via
            :meth:`Part.slice`, runs ``pipeline.generate`` per chunk, and
            concatenates the filled gen parts via :meth:`Part.concat`. Bounds
            stage peak memory (e.g. SD3 VAE decode) when there is no external
            inference runtime to chunk for us. **Single gen Part only** — chunking
            a multi-stage lineage would re-run the interior stage(s) per chunk and
            drop their output; ``generate`` rejects that combination.
    """

    _component_name = "trainside"

    def __init__(
        self,
        *,
        pipeline: Pipeline,
        stage: Optional[Stage] = None,
        stage_attrs: Sequence[str] = ("diffusion",),
        forward_batch_size: Optional[int] = None,
    ) -> None:
        self.pipeline = pipeline
        if stage is not None:
            stages = [stage]
        else:
            stages = [getattr(pipeline, a) for a in stage_attrs]
        self._models = [s.trainable_module() for s in stages]
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
                    getattr(pipeline.bundle, "pretrained_path", None),
                    shift=float(pipeline.shift),
                )
        else:
            self.schedule_policy = None

        self._weight_version = 0
        self._generate_lock = threading.Lock()
        self._shutdown_lock = threading.Lock()
        self._shutdown_requested = False
        self._shutdown_complete = False

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def generate(self, sample: Sample) -> Sample:
        """Generate one whole DP shard synchronously."""
        return self._generate_locked(sample)

    def _generate_locked(self, sample: Sample) -> Sample:
        with self._generate_lock:
            if self._shutdown_requested:
                raise RuntimeError("TrainsideRolloutEngine.generate called after shutdown")
            return self._stamp_weight_version(self._generate_core(sample))

    def _generate_core(self, sample: Sample) -> Sample:
        """Synchronous pipeline forward for one whole ``Sample``."""
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
            self._ensure_sample_sigmas(sample)
        prev_modes = [m.training for m in self._models]
        for m in self._models:
            m.eval()
        try:
            with torch.no_grad():
                fbs = self.forward_batch_size
                gen = sample.parts[-1]
                bs = int(gen.batch_size)
                if fbs is None or bs <= fbs:
                    return self.pipeline.generate(sample)
                input_parts = sample.parts[:-1]
                gen_chunks: List[Part] = []
                for start in range(0, bs, fbs):
                    end = min(start + fbs, bs)
                    chunk = self.pipeline.generate(Sample(parts=[*input_parts, gen.slice(start, end)]))
                    gen_chunks.append(chunk.parts[-1])
                return Sample(parts=[*input_parts, Part.concat(gen_chunks)])
        finally:
            for m, mode in zip(self._models, prev_modes):
                m.train(mode)

    def _ensure_sample_sigmas(self, sample: Sample) -> None:
        """Pin the σ schedule onto the gen part's ``DiffusionSamplingParams.sigmas``.

        Shared across the part's samples (one params object). Only reached when a
        diffusion stage is present (``schedule_policy is not None``).
        """
        ensure_sample_sigmas(sample, self.schedule_policy)

    def shutdown(self) -> None:
        with self._shutdown_lock:
            if self._shutdown_complete:
                return
            with self._generate_lock:
                self._shutdown_requested = True
            self._shutdown_complete = True

    def health_check(self) -> bool:
        return self.pipeline is not None and all(m is not None for m in self._models)


__all__ = ["TrainsideRolloutEngine"]
