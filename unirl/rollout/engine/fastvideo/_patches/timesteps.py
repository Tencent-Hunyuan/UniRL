"""Model-scoped custom-sigma adaptation for FastVideo schedulers."""

from __future__ import annotations

from functools import wraps

import numpy as np
import torch


def patch_timesteps() -> None:
    """Convert UniRL custom sigmas only when the selected adapter requests it."""

    from fastvideo.pipelines.stages.timestep_preparation import TimestepPreparationStage

    original_forward = TimestepPreparationStage.forward
    if getattr(original_forward, "_unirl_fastvideo_timesteps", False):
        return

    @wraps(original_forward)
    def forward(self, batch, fastvideo_args):
        original_sigmas = batch.sigmas
        dtype = getattr(fastvideo_args, "_unirl_custom_sigmas_dtype", None)
        if original_sigmas is not None and dtype == "float32":
            batch.sigmas = np.asarray(original_sigmas, dtype=np.float32)
        try:
            result = original_forward(self, batch, fastvideo_args)
            if original_sigmas is not None and dtype == "float32":
                # FlowUniPC stores custom sigmas exactly but truncates their
                # corresponding model timesteps to int64. Keep the scheduler
                # and denoising loop on exact float timesteps; the Wan-scoped
                # transformer wrapper casts only the model input to long.
                timesteps = self.scheduler.sigmas[:-1].to(
                    device=result.timesteps.device,
                    dtype=torch.float32,
                )
                timesteps = timesteps * float(self.scheduler.config.num_train_timesteps)
                self.scheduler.timesteps = timesteps
                result.timesteps = timesteps
            return result
        finally:
            batch.sigmas = original_sigmas

    setattr(forward, "_unirl_fastvideo_timesteps", True)
    TimestepPreparationStage.forward = forward


__all__ = ["patch_timesteps"]
