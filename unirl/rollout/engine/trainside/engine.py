"""Trainside (in-process) rollout engine adapter.

Wraps a materialized ``models`` :class:`Pipeline` plus the trainable
stage, and exposes them as a :class:`BaseRolloutEngine`.  Used in
direct-sampling mode where the training model IS the sampler (on-policy
RL) and rollout runs in the same Python process as training — so no
worker subprocess and no weight sync are needed.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import torch

from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.models.types.ar import ARStage
from unirl.models.types.diffusion import DiffusionStage
from unirl.models.types.pipeline import Pipeline
from unirl.rollout.engine.base import BaseRolloutEngine
from unirl.sde.runtime import FlowMatchSchedulePolicy, ensure_req_sigmas
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp

Stage = Union[DiffusionStage, ARStage]


class TrainsideRolloutEngine(BaseRolloutEngine):
    """In-process rollout engine: the train actor's Pipeline IS the sampler.

    Args:
        pipeline: A materialized ``models`` pipeline whose
            ``generate(req)`` populates ``RolloutResp``.
        stage: Optional pre-resolved trainable stage whose
            ``trainable_module()`` is the FSDP-wrapped model (the v1 train
            actor passes one). Takes precedence over ``stage_attrs``.
        stage_attrs: Stage attribute(s) to read off ``pipeline`` and
            eval-scope around ``generate``. A list so composed pipelines can
            drive more than one trainable module (e.g. PE's
            ``["diffusion", "ar"]``); defaults to ``("diffusion",)`` for the
            common single-diffusion engine.
        forward_batch_size: Optional intra-call chunk size for the
            ``pipeline.generate`` forward path. When set and the request
            exceeds this, ``generate`` slices the request via
            :meth:`RolloutReq.slice`, runs ``pipeline.generate`` per chunk,
            and concatenates results via :meth:`RolloutResp.concat`.
            Bounds stage peak memory (e.g. SD3 VAE decode) when there is
            no external inference runtime to chunk for us.
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
        # Resolve the trainable module(s) to eval-scope around generate().
        # A pre-resolved ``stage`` (the v1 train actor passes one) wins;
        # otherwise resolve ``stage_attrs`` off the pipeline. ``stage_attrs``
        # is a list so composed pipelines eval-scope more than one trainable
        # module (e.g. PE's ["diffusion", "ar"]); the ("diffusion",) default
        # keeps the common single-diffusion case.
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
        # Build a σ-schedule only when a diffusion stage is present (PE wraps
        # both diffusion + ar, so check the resolved list, not the lone `stage`
        # param which is None on the stage_attrs path); AR-only needs none.
        if any(isinstance(s, DiffusionStage) for s in stages):
            if hasattr(pipeline, "build_schedule_policy"):
                self.schedule_policy = pipeline.build_schedule_policy()
            else:
                self.schedule_policy = FlowMatchSchedulePolicy.from_pretrained(
                    getattr(pipeline.bundle, "pretrained_path", None),
                    shift=float(pipeline.shift),
                )
        else:
            # AR stage — no diffusion schedule needed
            self.schedule_policy = None

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def generate(self, req: RolloutReq) -> RolloutResp:
        if self.schedule_policy is not None:
            ensure_req_sigmas(req, self.schedule_policy)
        prev_modes = [m.training for m in self._models]
        for m in self._models:
            m.eval()
        try:
            with torch.no_grad():
                fbs = self.forward_batch_size
                bs = int(req.batch_size)
                if fbs is None or bs <= fbs:
                    return self.pipeline.generate(req)
                outputs: List[RolloutResp] = []
                for start in range(0, bs, fbs):
                    end = min(start + fbs, bs)
                    outputs.append(self.pipeline.generate(req.slice(start, end)))
                    # LIN-387: no per-chunk empty_cache() — it forced allocator
                    # re-warm on the next chunk (decode 0.87s -> 2.76s spikes).
                    # Chunking alone bounds the live-tensor peak; cached blocks
                    # are reused, not leaked.
                return RolloutResp.concat(outputs)
        finally:
            for m, mode in zip(self._models, prev_modes):
                m.train(mode)

    def shutdown(self) -> None:
        pass

    # sleep / wake_up inherit BaseRolloutEngine's @distributed no-op default.

    def health_check(self) -> bool:
        return self.pipeline is not None and all(m is not None for m in self._models)


__all__ = ["TrainsideRolloutEngine"]
