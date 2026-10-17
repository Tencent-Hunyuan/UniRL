"""RL-aware HunyuanVideo-1.5 pipeline subclass."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.hunyuan_video.pipeline_hunyuan_video_1_5 import (
    HunyuanVideo15Pipeline,
)
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from unirl.rollout.engine.vllm_omni.pipelines._shared.flow_match_sde_scheduler import (
    FlowMatchSDEDiscreteScheduler,
)
from unirl.rollout.engine.vllm_omni.pipelines._shared.interception import (
    detach_cpu,
    drain_trajectory_into,
    inject_latents,
    resolve_request_noise,
    single_request,
    slice_request_denoise_seed_keys,
    stamp_capture,
)


class RLHunyuanVideo15Pipeline(HunyuanVideo15Pipeline):
    """HunyuanVideo-1.5 pipeline with the RL interception protocol installed."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__(od_config=od_config, prefix=prefix)
        self.scheduler = FlowMatchSDEDiscreteScheduler.from_config(self.scheduler.config, eta=0.0)
        self.captured_conditioning: Optional[Dict[str, Any]] = None
        self.pending_initial_noise: Optional[torch.Tensor] = None

    def encode_prompt(self, *args: Any, **kw: Any) -> Any:  # type: ignore[override]
        """Capture the dual text-encoder embeddings on the first call per forward."""
        result = super().encode_prompt(*args, **kw)
        if self.captured_conditioning is None:
            (
                prompt_embeds,
                prompt_embeds_mask,
                prompt_embeds_2,
                prompt_embeds_mask_2,
                neg_embeds,
                neg_mask,
                neg_embeds_2,
                neg_mask_2,
            ) = result
            self.captured_conditioning = {
                "prompt_embeds": detach_cpu(prompt_embeds),
                "prompt_embeds_mask": detach_cpu(prompt_embeds_mask),
                "prompt_embeds_2": detach_cpu(prompt_embeds_2),
                "prompt_embeds_mask_2": detach_cpu(prompt_embeds_mask_2),
                "negative_prompt_embeds": detach_cpu(neg_embeds),
                "negative_prompt_embeds_mask": detach_cpu(neg_mask),
                "negative_prompt_embeds_2": detach_cpu(neg_embeds_2),
                "negative_prompt_embeds_mask_2": detach_cpu(neg_mask_2),
            }
        return result

    def prepare_latents(self, *args, **kwargs):  # type: ignore[override]
        """Initial-noise injection point: bypass upstream RNG when the driver supplied an x_T."""
        upstream = super().prepare_latents
        noise = self.pending_initial_noise
        if noise is not None:
            args, kwargs = inject_latents(upstream, args, kwargs, noise)
            self.pending_initial_noise = None
        return upstream(*args, **kwargs)

    def forward(self, req: DiffusionRequestBatch, **kwargs) -> DiffusionOutput:
        """Single-request batch in, single output out — see ``single_request``."""
        one = single_request(req, caller="RLHunyuanVideo15Pipeline.forward")
        extra = one.sampling_params.extra_args
        self.scheduler.arm(
            eta=one.sampling_params.eta,
            sde_indices=extra.get("sde_indices"),
            denoise_seed_keys=slice_request_denoise_seed_keys(one, caller="RLHunyuanVideo15Pipeline.forward"),
            denoise_base_seed=extra["denoise_base_seed"],
        )
        self.pending_initial_noise = resolve_request_noise(one, caller="RLHunyuanVideo15Pipeline.forward")
        self.captured_conditioning = None

        engine_sigmas = one.sampling_params.sigmas
        if engine_sigmas is not None:
            sched = self.scheduler
            orig_set_timesteps = sched.set_timesteps
            sched.set_timesteps = lambda *a, **kw: orig_set_timesteps(*a, **{**kw, "sigmas": engine_sigmas})  # type: ignore[assignment]
            try:
                out = super().forward(req, **kwargs)
            finally:
                del sched.set_timesteps
        else:
            out = super().forward(req, **kwargs)

        decoded = out.output
        if decoded is not None:
            stamp_capture(out, "rl_decoded_video", detach_cpu(decoded))

        drain_trajectory_into(out, self.scheduler)
        if self.captured_conditioning is not None:
            stamp_capture(out, "text_capture", self.captured_conditioning)
        return out


__all__ = ["RLHunyuanVideo15Pipeline"]
