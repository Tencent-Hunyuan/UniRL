"""Qwen3 pipeline wrapper that validates the experiment's frozen model contract."""

from __future__ import annotations

from unirl.models.qwen3.pipeline import Qwen3Pipeline

from .contract import validate_runtime_contract


class ParityQwen3Pipeline(Qwen3Pipeline):
    @classmethod
    def from_bundle(cls, bundle, **kwargs):
        validate_runtime_contract(
            bundle.transformer.config,
            dtype=bundle.dtype,
        )
        return super().from_bundle(bundle, **kwargs)


__all__ = ["ParityQwen3Pipeline"]
