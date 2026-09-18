"""Trainside (in-process) rollout engine adapter."""

from __future__ import annotations

import sys
import threading
from typing import Dict, List, Mapping, Optional, Sequence, Union

import torch

from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.models.types.ar import ARStage
from unirl.models.types.diffusion import DiffusionStage
from unirl.models.types.pipeline import Pipeline
from unirl.rollout.engine.base import BaseRolloutEngine
from unirl.rollout.engine.trainside.plan import (
    ExecutionKey,
    GeometryKey,
    assert_rank_uniform_schedule,
    plan_micro_batches,
    positive_int,
    prompt_group_sizes,
    schedule_token,
)
from unirl.sde.runtime import FlowMatchSchedulePolicy, ensure_sample_sigmas
from unirl.types.sample import Part, Sample
from unirl.types.sampling import DiffusionSamplingParams
from unirl.utils.distributed_utils import find_dtensor_mesh

Stage = Union[DiffusionStage, ARStage]

SCHEDULING_MODES = ("off", "shape_bucket", "continuous")
_UNBOUNDED_ROWS = sys.maxsize
_DEFAULT_BUCKET = "default"


class TrainsideRolloutEngine(BaseRolloutEngine):
    """In-process rollout engine: the train actor's Pipeline IS the sampler."""

    _component_name = "trainside"

    def __init__(
        self,
        *,
        pipeline: Pipeline,
        stage: Optional[Stage] = None,
        stage_attrs: Sequence[str] = ("diffusion",),
        forward_batch_size: Optional[int] = None,
        scheduling_mode: str = "off",
        bucket_batch_sizes: Optional[Mapping[str, int]] = None,
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
        self.scheduling_mode = str(scheduling_mode).strip().lower()
        if self.scheduling_mode not in SCHEDULING_MODES:
            raise ValueError(f"scheduling_mode must be one of {list(SCHEDULING_MODES)}; got {scheduling_mode!r}")
        if self.scheduling_mode == "continuous":
            raise NotImplementedError(
                "scheduling_mode='continuous' (step-level refill) is not implemented; see the trainside README. "
                "Use 'off' or 'shape_bucket'."
            )
        if self.scheduling_mode == "shape_bucket" and self.forward_batch_size is not None:
            raise ValueError(
                "shape_bucket and forward_batch_size are two independent chunk planners: forward_batch_size chunks "
                "by row count and can split a prompt group, while shape_bucket chunks by bucket capacity and never "
                "splits one. Set forward_batch_size=None and declare bucket_batch_sizes instead."
            )
        if bucket_batch_sizes and self.scheduling_mode != "shape_bucket":
            raise ValueError(
                f"bucket_batch_sizes requires scheduling_mode='shape_bucket'; got scheduling_mode "
                f"{self.scheduling_mode!r}. Drop the mapping or enable shape_bucket."
            )
        self._bucket_batch_sizes: Dict[str, int] = {
            str(name): positive_int(name=f"bucket_batch_sizes[{str(name)!r}]", value=value)
            for name, value in (bucket_batch_sizes or {}).items()
        }
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

        self._version = 0
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
            return self._stamp_output_version(self._generate_core(sample))

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
                if self.scheduling_mode == "shape_bucket":
                    return self._generate_bucketed(sample)
                return self._generate_chunked(sample)
        finally:
            for m, mode in zip(self._models, prev_modes):
                m.train(mode)

    def _generate_chunked(self, sample: Sample) -> Sample:
        """Forward the frontier in ``forward_batch_size`` row chunks (the legacy schedule)."""
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

    def _ensure_sample_sigmas(self, sample: Sample) -> None:
        """Pin the σ schedule onto the gen part's ``DiffusionSamplingParams.sigmas``."""
        ensure_sample_sigmas(sample, self.schedule_policy)

    def _generate_bucketed(self, sample: Sample) -> Sample:
        """Generate one frontier through shape-bucket micros that never split a prompt group."""
        gen_parts = [p for p in sample.parts if p.is_gen]
        if len(gen_parts) != 1:
            raise ValueError(
                "TrainsideRolloutEngine scheduling_mode='shape_bucket' supports exactly one gen frontier; "
                f"this Sample has {len(gen_parts)}. Use scheduling_mode='off'."
            )
        input_parts = sample.parts[:-1]
        gen = sample.parts[-1]
        if int(gen.batch_size) == 0:
            return self.pipeline.generate(sample)

        group_ids = gen.group_ids
        keys = self._execution_keys(sample, gen)
        plan = plan_micro_batches(group_ids=group_ids, keys=keys, capacity_rows=self._capacity_rows)
        assert_rank_uniform_schedule(plan, group_sizes=prompt_group_sizes(group_ids))
        if plan.is_passthrough:
            return self.pipeline.generate(sample)

        chunks: List[Part] = []
        for start, end in plan.micros:
            micro = gen.select(plan.permutation[start:end])
            chunks.append(self.pipeline.generate(Sample(parts=[*input_parts, micro])).parts[-1])
        self._assert_micro_shared_fields(chunks, gen)
        restored = Part.concat(chunks).select(plan.inverse)
        if restored.sample_ids != gen.sample_ids:
            raise RuntimeError(
                "shape_bucket: the inverse permutation did not restore the frontier's row order; refusing to return a "
                "reordered rollout (sample identity, GRPO siblings and advantages are order-coupled)."
            )
        return Sample(parts=[*input_parts, restored])

    def _execution_keys(self, sample: Sample, gen: Part) -> List[ExecutionKey]:
        """Per-row execution keys resolved from the geometry the driver already pinned."""
        params = gen.sampling_params
        if not isinstance(params, DiffusionSamplingParams):
            raise TypeError(
                "shape_bucket needs diffusion sampling params on the frontier; got "
                f"{type(params).__name__ if params is not None else 'None'}."
            )
        latent_shape = tuple(int(dim) for dim in (params.init_noise_latent_shape or []))
        if not latent_shape:
            raise ValueError(
                "shape_bucket needs the driver-pinned latent geometry: sampling_params.init_noise_latent_shape is "
                "unset (DISABLE_DRIVER_XT, or the pipeline declares no latent_shape()). Use scheduling_mode='off', "
                "or give the pipeline a latent_shape() classmethod."
            )
        key = ExecutionKey(
            geometry=GeometryKey(
                model_adapter=f"{type(self.pipeline).__module__}.{type(self.pipeline).__qualname__}",
                latent_shape=latent_shape,
                conditioning_layout=self._conditioning_layout(sample),
            ),
            dtype=f"{params.autocast_precision}/{params.trajectory_precision}",
            cfg_rows=2 if float(params.guidance_scale) > 1.0 else 1,
            schedule=schedule_token(params),
            weight_version=int(self._version),
            parallel=self._parallel_signature(),
        )
        return [key] * int(gen.batch_size)

    def _conditioning_layout(self, sample: Sample) -> str:
        """Structural digest of the conditioning chain every frontier row is generated from."""
        items: List[str] = []
        for part in sample.parts[:-1]:
            for name in sorted(part.primitives):
                primitive = part.primitives[name]
                items.append(f"{name}:{len(primitive) if hasattr(primitive, '__len__') else 0}")
        return "+".join(items) or "none"

    def _parallel_signature(self) -> str:
        """Collective-group signature of the model shard group, or the plain world when unwrapped."""
        import torch.distributed as dist

        world = int(dist.get_world_size()) if dist.is_available() and dist.is_initialized() else 1
        mesh = None
        for model in self._models:
            mesh = find_dtensor_mesh(model)
            if mesh is not None:
                break
        if mesh is None:
            return f"world{world}"
        names = tuple(mesh.mesh_dim_names or ())
        shard = int(mesh.size(names.index("dp_shard"))) if "dp_shard" in names else int(mesh.size(0))
        return f"world{world}/shard{shard}"

    def _capacity_rows(self, key: ExecutionKey) -> int:
        """Declared micro capacity in execution rows for one bucket, unbounded when none is declared."""
        if not self._bucket_batch_sizes:
            return _UNBOUNDED_ROWS
        declared = self._bucket_batch_sizes.get(key.geometry_token())
        if declared is None:
            declared = self._bucket_batch_sizes.get(_DEFAULT_BUCKET)
        if declared is None:
            raise ValueError(
                f"shape_bucket has no capacity for bucket {key.geometry_token()!r}; add it to bucket_batch_sizes or "
                f"add a {_DEFAULT_BUCKET!r} entry (configured buckets: {sorted(self._bucket_batch_sizes)})."
            )
        return declared

    @staticmethod
    def _assert_micro_shared_fields(chunks: Sequence[Part], gen: Part) -> None:
        """Reject a merge whose shared Part fields would be silently taken from the first micro."""
        for chunk in chunks:
            if chunk.sampling_params is not gen.sampling_params:
                raise ValueError(
                    "shape_bucket micros must keep the frontier's own sampling_params; one micro produced a different "
                    "shared params object, which Part.concat would silently resolve to the first micro."
                )
            if (chunk.role, chunk.harness_status) != (gen.role, gen.harness_status):
                raise ValueError(
                    "shape_bucket micros must keep the frontier's shared role/harness_status; Part.concat would "
                    "silently take the first micro's value."
                )

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
