"""Composable model pipeline interfaces.

A ``Pipeline`` is the top-level entrypoint that maps a ``RolloutReq`` (typed
inputs keyed by modality slot) into a ``RolloutResp`` (typed outputs:
per-track conditions used + segments produced + decoded primitives). Concrete
pipelines compose stage instances (``EmbedStage`` / ``EncodeStage`` /
``DiffusionStage`` / ``ARStage`` / ``DecodeStage``) for one model bundle.

The ``Pipeline`` Protocol itself is intentionally non-parametric — both
``RolloutReq`` and ``RolloutResp`` are universal in/out shapes shared
across every bundle, so per-model conditions typing happens *inside* the
pipeline (after ``RolloutReq.primitives`` are encoded into a typed
container, and before that container is repacked into
``RolloutResp.tracks[<slot>].conditions``).

Per-bundle contract documentation (which ``req.primitives`` keys are read,
which ``req.stage_params`` keys, and which ``resp.tracks[<slot>].{conditions,
segment, decoded}`` keys are produced) lives in each concrete ``Pipeline``'s
docstring so multiple bundles don't drift on the same key names.

σ schedule contract
-------------------
Diffusion pipelines no longer own σ construction. The engine adapter that
hosts the pipeline (``TrainsideRolloutEngine``, ``SGLangDiffusionRolloutEngine``,
``VLLMOmniRolloutEngine``) pins ``RolloutReq.sigmas`` via
:func:`unirl.sde.runtime.ensure_req_sigmas` BEFORE calling
``pipeline.generate(req)``; the pipeline reads ``req.sigmas`` and uses it
verbatim. This makes σ ownership explicit:

    Policy        → model checkpoint (loaded once via
                    ``FlowMatchSchedulePolicy.from_pretrained``)
    Params (T,H,W) → request (``req.stage_params['diffusion']``)
    σ tensor       → ``req.sigmas`` (engine-pinned, pipeline-consumed)
"""

from __future__ import annotations

from typing import Any, Protocol, Tuple

from unirl.distributed.group.remote import Remote
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp


class Pipeline(Remote):
    """Generate-time pipeline: ``RolloutReq → RolloutResp`` for one bundle."""

    def generate(self, req: RolloutReq) -> RolloutResp:
        raise NotImplementedError


class LatentShapeProvider(Protocol):
    """Optional Pipeline mixin: declare per-sample latent shape for
    driver-side noise pre-computation.

    The driver calls :meth:`latent_shape` BEFORE any actor is alive
    (in ``DiffusionTrainer._resolve_noise_latent_shape``, once at init) to
    produce a ``(C, *spatial)`` or ``(C, T, *spatial)`` tuple. That shape
    becomes the x_T RECIPE's ``RolloutReq.init_noise_latent_shape``; each
    engine then regenerates a byte-identical, seed-shared x_T from the recipe
    (``regen_initial_noise``) rather than the driver shipping a materialized
    tensor. This is the canonical path for GRPO group noise + resume
    determinism + rollout/replay consistency across engines.

    Pipelines that haven't been wired for driver-side noise pre-
    computation MUST raise ``NotImplementedError`` (driver then falls
    back to the engine's own RNG, accepting the determinism cost).
    """

    @classmethod
    def latent_shape(cls, *, model_config: Any, sampling_spec: Any) -> Tuple[int, ...]: ...


__all__ = ["LatentShapeProvider", "Pipeline"]
