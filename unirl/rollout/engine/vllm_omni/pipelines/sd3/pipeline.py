"""RL-aware StableDiffusion 3.5 pipeline subclass."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.sd3.pipeline_sd3 import StableDiffusion3Pipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest

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


class RLStableDiffusion3Pipeline(StableDiffusion3Pipeline):
    """SD3.5 pipeline with the RL interception protocol installed."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__(od_config=od_config, prefix=prefix)
        self._upstream_scheduler: FlowMatchEulerDiscreteScheduler = self.scheduler
        self._captured_conditioning: Optional[Dict[str, Any]] = None
        self._conditioning_tap_installed: bool = False
        self._t5_workaround_installed: bool = False
        self._pending_initial_noise: Optional[torch.Tensor] = None

    def _install_sde_scheduler(self) -> None:
        """Swap in the trajectory-capturing SDE scheduler via ``from_config``; always installed, even at eta=0."""
        if isinstance(self.scheduler, FlowMatchSDEDiscreteScheduler):
            return
        self.scheduler = make_sde_scheduler(self._upstream_scheduler.config)

    def _install_conditioning_tap(self) -> None:
        """Capture ``encode_prompt``: joint ``[B, L, D]`` for cross-attn, pooled ``[B, D_pooled]`` for AdaLN."""
        if self._conditioning_tap_installed:
            return

        orig = self.encode_prompt
        pipeline_self = self

        def tapped(*args: Any, **kw: Any) -> Any:
            result = orig(*args, **kw)
            if pipeline_self._captured_conditioning is None:
                prompt_embeds, pooled_prompt_embeds = result
                pipeline_self._captured_conditioning = {
                    "prompt_embeds": detach_cpu(prompt_embeds),
                    "pooled_prompt_embeds": detach_cpu(pooled_prompt_embeds),
                }
            return result

        self.encode_prompt = tapped  # type: ignore[assignment]
        self._conditioning_tap_installed = True

    def _install_t5_truncation_workaround(self) -> None:
        """WORKAROUND: replace ``_get_t5_prompt_embeds`` to skip a cross-device warning check."""
        if self._t5_workaround_installed:
            return

        pipeline_self = self

        def patched_get_t5_prompt_embeds(
            prompt: Any,
            num_images_per_prompt: int = 1,
            max_sequence_length: int = 256,
            dtype: Optional[torch.dtype] = None,
        ) -> torch.Tensor:
            prompt_list = [prompt] if isinstance(prompt, str) else prompt
            batch_size = len(prompt_list)

            if pipeline_self.text_encoder_3 is None:
                dtype_fallback = dtype or getattr(pipeline_self.transformer, "dtype", torch.float32)
                return torch.zeros(
                    (
                        batch_size,
                        max_sequence_length,
                        pipeline_self.transformer.joint_attention_dim,
                    ),
                    device=pipeline_self.device,
                    dtype=dtype_fallback,
                )

            dtype = dtype or pipeline_self.text_encoder_3.dtype
            text_inputs = pipeline_self.tokenizer_3(
                prompt_list,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            ).to(pipeline_self.device)
            text_input_ids = text_inputs.input_ids

            prompt_embeds = pipeline_self.text_encoder_3(text_input_ids.to(pipeline_self.device))[0]
            prompt_embeds = prompt_embeds.to(
                dtype=pipeline_self.text_encoder_3.dtype,
                device=pipeline_self.device,
            )
            _, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
            return prompt_embeds

        self._get_t5_prompt_embeds = patched_get_t5_prompt_embeds  # type: ignore[assignment]
        self._t5_workaround_installed = True

    def _arm_sde(self, req: OmniDiffusionRequest) -> None:
        """This request's SDE strength + sparse step gate."""
        eta = float(getattr(req.sampling_params, "eta", 0.0) or 0.0)
        extra = getattr(req.sampling_params, "extra_args", None) or {}
        self.scheduler.arm(eta=eta, sde_indices=extra.get("sde_indices"))

    def _arm_initial_noise(self, req: OmniDiffusionRequest) -> None:
        """This request's driver-authored x_T (batch slice or recipe row)."""
        self._pending_initial_noise = resolve_request_noise(req, caller="RLStableDiffusion3Pipeline._arm_initial_noise")

    def _arm_conditioning_tap(self) -> None:
        """Fresh capture buffer so the tap records THIS request's first encode."""
        self._captured_conditioning = None

    # run-phase interception — upstream-called name, cannot be renamed

    def prepare_latents(self, *args, **kwargs):  # type: ignore[override]
        """Initial-noise injection point: bypass upstream RNG when the driver supplied an x_T."""
        noise = self._pending_initial_noise
        if noise is not None:
            args, kwargs = inject_latents(args, kwargs, noise)
            self._pending_initial_noise = None
        return super().prepare_latents(*args, **kwargs)

    def _harvest_trajectory(self, out: DiffusionOutput) -> None:
        if isinstance(self.scheduler, FlowMatchSDEDiscreteScheduler):
            drain_trajectory_into(out, self.scheduler)

    def _harvest_conditioning(self, out: DiffusionOutput) -> None:
        if self._captured_conditioning is not None:
            stamp_custom_output(out, "text_capture", self._captured_conditioning)

    def forward(self, req: OmniDiffusionRequest, **kwargs) -> DiffusionOutput:
        self._install_sde_scheduler()
        self._install_conditioning_tap()
        self._install_t5_truncation_workaround()

        self._arm_sde(req)
        self._arm_initial_noise(req)
        self._arm_conditioning_tap()

        out = super().forward(req, **kwargs)

        self._harvest_trajectory(out)
        self._harvest_conditioning(out)
        return out


__all__ = ["RLStableDiffusion3Pipeline"]
