"""RL-aware Qwen-Image pipeline subclass."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image import QwenImagePipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.utils.size_utils import normalize_min_aligned_size

from unirl.rollout.engine.vllm_omni.pipelines._shared.flow_match_sde_scheduler import (
    FlowMatchSDEDiscreteScheduler,
)
from unirl.rollout.engine.vllm_omni.pipelines._shared.interception import (
    detach_cpu,
    drain_trajectory_into,
    inject_latents,
    make_sde_scheduler,
    resolve_request_noise,
    stamp_custom_output,
)


class RLQwenImagePipeline(QwenImagePipeline):
    """Qwen-Image pipeline with the RL interception protocol installed."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__(od_config=od_config, prefix=prefix)
        self._upstream_scheduler: FlowMatchEulerDiscreteScheduler = self.scheduler
        self._captured_conditioning: Optional[Dict[str, Any]] = None
        self._conditioning_tap_installed: bool = False
        self._pending_initial_noise: Optional[torch.Tensor] = None
        self._harvest_hw: Optional[Tuple[int, int]] = None

    def _install_sde_scheduler(self) -> None:
        """Swap in the trajectory-capturing SDE scheduler via ``from_config``; always installed, even at eta=0."""
        if isinstance(self.scheduler, FlowMatchSDEDiscreteScheduler):
            return
        self.scheduler = make_sde_scheduler(self._upstream_scheduler.config)

    def _install_conditioning_tap(self) -> None:
        """Wrap ``encode_prompt`` to capture the text conditioning."""
        if self._conditioning_tap_installed:
            return

        orig = self.encode_prompt
        pipeline_self = self

        def tapped(*args: Any, **kw: Any) -> Any:
            result = orig(*args, **kw)
            cap = pipeline_self._captured_conditioning
            if cap is not None:
                prompt_embeds, prompt_embeds_mask = result
                if "prompt_embeds" not in cap:
                    cap["prompt_embeds"] = detach_cpu(prompt_embeds)
                    cap["prompt_embeds_mask"] = detach_cpu(prompt_embeds_mask)
                elif "negative_prompt_embeds" not in cap:
                    cap["negative_prompt_embeds"] = detach_cpu(prompt_embeds)
                    cap["negative_prompt_embeds_mask"] = detach_cpu(prompt_embeds_mask)
            return result

        self.encode_prompt = tapped  # type: ignore[assignment]
        self._conditioning_tap_installed = True

    def _arm_sde(self, req: OmniDiffusionRequest) -> None:
        """This request's SDE strength + sparse step gate."""
        eta = float(getattr(req.sampling_params, "eta", 0.0) or 0.0)
        extra = getattr(req.sampling_params, "extra_args", None) or {}
        self.scheduler.arm(eta=eta, sde_indices=extra.get("sde_indices"))

    def _arm_initial_noise(self, req: OmniDiffusionRequest) -> None:
        """This request's driver-authored x_T, still spatial ``[1, C, H, W]``; packing happens at injection."""
        self._pending_initial_noise = resolve_request_noise(req, caller="RLQwenImagePipeline._arm_initial_noise")

    def _arm_conditioning_tap(self) -> None:
        """Fresh capture buffer so the tap records THIS request's encodes."""
        self._captured_conditioning = {}

    # run-phase interception — upstream-called name, cannot be renamed

    def prepare_latents(self, *args, **kwargs):  # type: ignore[override]
        """Initial-noise injection: the driver's ``[B, C, H, W]`` x_T is packed to ``[B, S, C*4]``. Consume-once."""
        noise = self._pending_initial_noise
        if noise is not None:
            self._pending_initial_noise = None
            args, kwargs = inject_latents(args, kwargs, self._pack_pending_noise(noise, args))
        return super().prepare_latents(*args, **kwargs)

    def _pack_pending_noise(self, noise: torch.Tensor, args: tuple) -> torch.Tensor:
        """Spatial ``[B, C, h, w]`` x_T → packed ``[B, S, C*4]``, validated against the call site's grid geometry."""
        if len(args) < 4:
            raise RuntimeError(
                "RLQwenImagePipeline._pack_pending_noise: expected upstream's "
                f"fully positional prepare_latents call; got {len(args)} positional args."
            )
        batch, channels = int(args[0]), int(args[1])
        grid_h = 2 * (int(args[2]) // (self.vae_scale_factor * 2))
        grid_w = 2 * (int(args[3]) // (self.vae_scale_factor * 2))
        if tuple(noise.shape) != (batch, channels, grid_h, grid_w):
            raise RuntimeError(
                "RLQwenImagePipeline: driver x_T shape "
                f"{tuple(noise.shape)} does not match the worker latent grid "
                f"[{batch}, {channels}, {grid_h}, {grid_w}] for "
                f"{int(args[2])}x{int(args[3])} px — check the recipe's "
                "init_noise_latent_shape / initial_noise_batch."
            )
        return self._pack_latents(noise, batch, channels, grid_h, grid_w)

    def _harvest_trajectory(self, out: DiffusionOutput) -> None:
        if not isinstance(self.scheduler, FlowMatchSDEDiscreteScheduler):
            return
        drain_trajectory_into(out, self.scheduler)
        if out.trajectory_latents is not None:
            out.trajectory_latents = self._unpack_trajectory(out.trajectory_latents)

    def _unpack_trajectory(self, packed: torch.Tensor) -> torch.Tensor:
        """Packed ``[B, T+1, S, C*4]`` trajectory to spatial ``[B, T+1, C, H, W]``, the ``LatentSegment`` shape."""
        if self._harvest_hw is None:
            raise RuntimeError(
                "RLQwenImagePipeline._unpack_trajectory: no stashed H/W — forward() did not run before harvest."
            )
        height, width = self._harvest_hw
        b, t1 = packed.shape[0], packed.shape[1]
        flat = self._unpack_latents(packed.reshape(b * t1, *packed.shape[2:]), height, width, self.vae_scale_factor)
        flat = flat.squeeze(2)
        return flat.reshape(b, t1, *flat.shape[1:])

    def _harvest_conditioning(self, out: DiffusionOutput) -> None:
        if self._captured_conditioning:
            stamp_custom_output(out, "text_capture", self._captured_conditioning)

    def forward(self, req: OmniDiffusionRequest, **kwargs) -> DiffusionOutput:
        self._install_sde_scheduler()
        self._install_conditioning_tap()

        self._arm_sde(req)
        self._arm_initial_noise(req)
        self._arm_conditioning_tap()
        height = req.sampling_params.height or self.default_sample_size * self.vae_scale_factor
        width = req.sampling_params.width or self.default_sample_size * self.vae_scale_factor
        height, width = normalize_min_aligned_size(height, width, self.vae_scale_factor * 2)
        self._harvest_hw = (int(height), int(width))

        out = super().forward(req, **kwargs)

        self._harvest_trajectory(out)
        self._harvest_conditioning(out)
        return out


__all__ = ["RLQwenImagePipeline"]
