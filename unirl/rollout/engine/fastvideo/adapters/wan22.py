"""Wan2.2 A14B dual-expert adapter for FastVideo rollout."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from unirl.config.require import require
from unirl.rollout.engine.fastvideo.adapters.base import register_adapter
from unirl.rollout.engine.fastvideo.adapters.wan21 import Wan21FastVideoAdapter


@register_adapter("wan2.2", "wan22")
class Wan22FastVideoAdapter(Wan21FastVideoAdapter):
    """Add Wan2.2 dual-transformer routing to the shared Wan rollout contract."""

    engine_name = "fastvideo/wan2.2"

    def validate(self) -> None:
        super().validate()
        boundary_ratio = float(getattr(self.model_config, "boundary_ratio", 0.0))
        require(0.0 < boundary_ratio < 1.0, f"Wan2.2 boundary_ratio must be in (0, 1); got {boundary_ratio}")
        require(
            int(getattr(self.model_config, "num_train_timesteps", 1000)) == 1000,
            "Wan2.2 FastVideo rollout requires num_train_timesteps=1000",
        )

        model_path = Path(str(self.model_config.pretrained_model_ckpt_path)).expanduser()
        model_index = model_path / "model_index.json"
        if model_index.is_file():
            payload = json.loads(model_index.read_text())
            require(
                "transformer" in payload and "transformer_2" in payload,
                f"Wan2.2 A14B checkpoint must expose transformer and transformer_2: {model_index}",
            )

    def align_runtime_args(self, fastvideo_args: Any) -> None:
        super().align_runtime_args(fastvideo_args)
        pipeline_config = fastvideo_args.pipeline_config
        dit_config = getattr(pipeline_config, "dit_config", None)
        require(dit_config is not None, "Wan2.2 FastVideo pipeline has no dit_config")
        require(
            hasattr(dit_config, "boundary_ratio"),
            "Wan2.2 FastVideo requires a dual-expert pipeline config with dit_config.boundary_ratio",
        )
        boundary_ratio = float(self.model_config.boundary_ratio)
        pipeline_config.boundary_ratio = boundary_ratio
        dit_config.boundary_ratio = boundary_ratio

    def build_forward_batch(
        self,
        *,
        prompt: str,
        seed: int,
        params: Any,
        sigmas: Any,
        fastvideo_args: Any,
    ) -> Any:
        batch = super().build_forward_batch(
            prompt=prompt,
            seed=seed,
            params=params,
            sigmas=sigmas,
            fastvideo_args=fastvideo_args,
        )
        low_noise_guidance = getattr(params, "guidance_scale_2", None)
        if low_noise_guidance is None:
            low_noise_guidance = getattr(self.model_config, "guidance_scale_2", None)
        batch.guidance_scale_2 = (
            float(low_noise_guidance) if low_noise_guidance is not None else float(params.guidance_scale)
        )
        batch.boundary_ratio = float(self.model_config.boundary_ratio)
        return batch


__all__ = ["Wan22FastVideoAdapter"]
