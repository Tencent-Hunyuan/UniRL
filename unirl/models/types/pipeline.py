"""Composable model pipeline interfaces.

A ``Pipeline`` is the top-level entrypoint that maps a request ``Sample`` (an
ordered ``parts`` chain whose frontier is a pre-forked generation shell) into a
filled ``Sample`` (the same chain with the frontier Part's ``segment`` /
``primitives`` populated). Concrete pipelines compose stage instances
(``EmbedStage`` / ``EncodeStage`` / ``DiffusionStage`` / ``ARStage`` /
``DecodeStage``) for one model bundle.

The ``Pipeline`` Protocol itself is intentionally non-parametric — ``Sample`` is
the universal in/out shape shared across every bundle, so per-model conditions
typing happens *inside* the pipeline (after the prompt(s) surfaced by
``sample.conditioning()`` are encoded into a typed container, and before the
frontier Part is filled via ``Part.fill``).

Per-bundle contract documentation (which ``sample.conditioning()`` primitives are
read, which ``frontier.sampling_params`` fields, and what ``segment`` /
``primitives`` the frontier Part is filled with) lives in each concrete
``Pipeline``'s docstring so multiple bundles don't drift.

σ schedule contract
-------------------
Diffusion pipelines no longer own σ construction. The engine adapter that
hosts the pipeline (``TrainsideRolloutEngine``, ``SGLangDiffusionRolloutEngine``,
``VLLMOmniRolloutEngine``) pins the σ schedule onto the gen Part's
``DiffusionSamplingParams.sigmas`` BEFORE calling ``pipeline.generate(sample)``;
the pipeline reads ``params.sigmas`` and uses it verbatim. This makes σ
ownership explicit:

    Policy        → model checkpoint (loaded once via
                    ``FlowMatchSchedulePolicy.from_pretrained``)
    Params (T,H,W) → the gen Part's ``DiffusionSamplingParams``
    σ tensor       → ``params.sigmas`` (engine-pinned, pipeline-consumed)
"""

from __future__ import annotations

from typing import Any, Protocol, Tuple

from unirl.distributed.group.remote import Remote
from unirl.types.sample import Sample


class Pipeline(Remote):
    """Generate-time pipeline: ``Sample → Sample`` for one bundle."""

    def generate(self, sample: Sample) -> Sample:
        raise NotImplementedError


class LatentShapeProvider(Protocol):
    """Optional Pipeline mixin: declare per-sample latent shape for
    driver-side noise pre-computation.

    The driver calls :meth:`latent_shape` BEFORE any actor is alive
    (in ``DiffusionTrainer._resolve_noise_latent_shape``, once at init) to
    produce a ``(C, *spatial)`` or ``(C, T, *spatial)`` tuple. That shape
    becomes the x_T RECIPE's ``init_noise_latent_shape`` on the gen Part's
    ``DiffusionSamplingParams``; each engine then regenerates a byte-identical,
    seed-shared x_T from the recipe (``regen_initial_noise``) rather than the
    driver shipping a materialized tensor. This is the canonical path for GRPO
    group noise + resume determinism + rollout/replay consistency across engines.

    Pipelines that haven't been wired for driver-side noise pre-
    computation MUST raise ``NotImplementedError`` (driver then falls
    back to the engine's own RNG, accepting the determinism cost).
    """

    @classmethod
    def latent_shape(cls, *, model_config: Any, sampling_spec: Any) -> Tuple[int, ...]: ...


__all__ = ["LatentShapeProvider", "Pipeline"]
