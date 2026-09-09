"""Composable model pipeline interfaces."""

from __future__ import annotations

from typing import Any, Protocol, Tuple

from unirl.distributed.group.remote import Remote
from unirl.types.sample import Sample


class Pipeline(Remote):
    """Generate-time pipeline: ``Sample → Sample`` for one bundle."""

    def generate(self, sample: Sample) -> Sample:
        raise NotImplementedError


class LatentShapeProvider(Protocol):
    """Optional Pipeline mixin: per-sample latent shape ``(C, *spatial)`` or ``(C, T, *spatial)`` for the x_T recipe."""

    @classmethod
    def latent_shape(cls, *, model_config: Any, sampling_spec: Any) -> Tuple[int, ...]: ...


__all__ = ["LatentShapeProvider", "Pipeline"]
