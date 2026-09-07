"""RL-aware HunyuanVideo-1.5 pipeline subclass."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

import torch
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.hunyuan_video.pipeline_hunyuan_video_1_5 import (
    HunyuanVideo15Pipeline,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from unirl.rollout.engine.vllm_omni.pipelines._shared.flow_match_sde_scheduler import (
    FlowMatchSDEDiscreteScheduler,
)
from unirl.rollout.engine.vllm_omni.pipelines._shared.interception import (
    detach_cpu,
    drain_trajectory_into,
    inject_latents,
    make_sde_scheduler,
    resolve_request_noise,
    single_request,
    stamp_capture,
)


class RLHunyuanVideo15Pipeline(HunyuanVideo15Pipeline):
    """HunyuanVideo-1.5 pipeline with the RL interception protocol installed."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__(od_config=od_config, prefix=prefix)
        self._upstream_scheduler = self.scheduler
        self._captured_conditioning: Optional[Dict[str, Any]] = None
        self._conditioning_tap_installed: bool = False
        self._pending_initial_noise: Optional[torch.Tensor] = None

    def _install_sde_scheduler(self) -> None:
        """Swap in the trajectory-capturing SDE scheduler (from_config keeps the upstream schedule parameters)."""
        if isinstance(self.scheduler, FlowMatchSDEDiscreteScheduler):
            return
        self.scheduler = make_sde_scheduler(self._upstream_scheduler.config)

    def _install_conditioning_tap(self) -> None:
        """Wrap ``encode_prompt`` to capture the dual text-encoder embeddings."""
        if self._conditioning_tap_installed:
            return

        orig = self.encode_prompt
        pipeline_self = self

        def tapped(*args: Any, **kw: Any) -> Any:
            result = orig(*args, **kw)
            if pipeline_self._captured_conditioning is None:
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
                pipeline_self._captured_conditioning = {
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

        self.encode_prompt = tapped  # type: ignore[assignment]
        self._conditioning_tap_installed = True

    def _arm_sde(self, req: OmniDiffusionRequest) -> None:
        """This request's SDE strength + sparse step gate."""
        eta = float(getattr(req.sampling_params, "eta", 0.0) or 0.0)
        extra = getattr(req.sampling_params, "extra_args", None) or {}
        self.scheduler.arm(eta=eta, sde_indices=extra.get("sde_indices"))

    def _arm_initial_noise(self, req: OmniDiffusionRequest) -> None:
        """This request's driver-authored x_T (batch slice or recipe row)."""
        self._pending_initial_noise = resolve_request_noise(req, caller="RLHunyuanVideo15Pipeline._arm_initial_noise")

    def _arm_conditioning_tap(self) -> None:
        """Fresh capture buffer so the tap records THIS request's first encode."""
        self._captured_conditioning = None

    def prepare_latents(self, *args, **kwargs):  # type: ignore[override]
        """Initial-noise injection point: bypass upstream RNG when the driver supplied an x_T."""
        upstream = super().prepare_latents
        noise = self._pending_initial_noise
        if noise is not None:
            args, kwargs = inject_latents(upstream, args, kwargs, noise)
            self._pending_initial_noise = None
        return upstream(*args, **kwargs)

    @contextmanager
    def _sigma_override(self, req: OmniDiffusionRequest) -> Iterator[None]:
        """WORKAROUND: make upstream pick up the engine's σ schedule."""
        engine_sigmas = getattr(req.sampling_params, "sigmas", None)
        if engine_sigmas is None:
            yield
            return

        sched = self.scheduler
        orig_set_timesteps = sched.set_timesteps

        def _set_timesteps_with_engine_sigmas(*args: Any, **kw: Any) -> Any:
            kw["sigmas"] = engine_sigmas
            return orig_set_timesteps(*args, **kw)

        sched.set_timesteps = _set_timesteps_with_engine_sigmas  # type: ignore[assignment]
        try:
            yield
        finally:
            del sched.set_timesteps

    def _harvest_trajectory(self, out: DiffusionOutput) -> None:
        if isinstance(self.scheduler, FlowMatchSDEDiscreteScheduler):
            drain_trajectory_into(out, self.scheduler)

    def _harvest_conditioning(self, out: DiffusionOutput) -> None:
        if self._captured_conditioning is not None:
            stamp_capture(out, "text_capture", self._captured_conditioning)

    def forward(self, req: DiffusionRequestBatch, **kwargs) -> DiffusionOutput:
        """Single-request batch in, single output out — see ``single_request``."""
        one = single_request(req, caller="RLHunyuanVideo15Pipeline.forward")
        self._install_sde_scheduler()
        self._install_conditioning_tap()

        self._arm_sde(one)
        self._arm_initial_noise(one)
        self._arm_conditioning_tap()

        with self._sigma_override(one):
            out = super().forward(req, **kwargs)

        decoded = getattr(out, "output", None)
        if decoded is not None:
            stamp_capture(out, "rl_decoded_video", detach_cpu(decoded))

        self._harvest_trajectory(out)
        self._harvest_conditioning(out)
        return out


__all__ = ["RLHunyuanVideo15Pipeline"]
