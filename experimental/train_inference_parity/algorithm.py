"""Experimental algorithm wrapper that requires exact replay/rollout parity."""

from __future__ import annotations

from unirl.algorithms.grpo import GRPO


class TrainInferenceParityGRPO(GRPO):
    """GRPO with a fail-closed gradient-replay/rollout parity gate."""

    def __init__(
        self,
        *args,
        alignment_require_exact: bool = True,
        **kwargs,
    ) -> None:
        if not alignment_require_exact:
            raise ValueError("TrainInferenceParityGRPO requires alignment_require_exact=true")
        super().__init__(
            *args,
            alignment_require_exact=True,
            **kwargs,
        )


__all__ = ["TrainInferenceParityGRPO"]
