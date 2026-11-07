"""RL-aware StableDiffusion 3.5 pipeline subclass."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.sd3.pipeline_sd3 import StableDiffusion3Pipeline
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from unirl.rollout.engine.vllm_omni.pipelines._shared.flow_match_sde_scheduler import (
    FlowMatchSDEDiscreteScheduler,
)
from unirl.rollout.engine.vllm_omni.pipelines._shared.interception import (
    detach_cpu,
    drain_trajectory_into,
    finalize_output,
    inject_latents,
    resolve_request_noise,
    single_request,
    stamp_capture,
)


class RLStableDiffusion3Pipeline(StableDiffusion3Pipeline):
    """SD3.5 pipeline with the RL interception protocol installed."""

    # Upstream opts into request batching; RL rollout does not — see ``single_request``.
    supports_request_batch = False

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__(od_config=od_config, prefix=prefix)
        self.scheduler = FlowMatchSDEDiscreteScheduler.from_config(self.scheduler.config, eta=0.0)
        self.captured_conditioning: Optional[Dict[str, Any]] = None
        self.pending_initial_noise: Optional[torch.Tensor] = None

    def encode_prompt(self, *args: Any, **kw: Any) -> Any:  # type: ignore[override]
        """Capture joint ``[B, L, D]`` and pooled ``[B, D_pooled]`` text conditioning."""
        result = super().encode_prompt(*args, **kw)
        if self.captured_conditioning is None:
            prompt_embeds, pooled_prompt_embeds = result
            self.captured_conditioning = {
                "prompt_embeds": detach_cpu(prompt_embeds),
                "pooled_prompt_embeds": detach_cpu(pooled_prompt_embeds),
            }
        return result

    def _get_t5_prompt_embeds(
        self,
        prompt: Any,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 256,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Override upstream T5 embed helper to skip a cross-device warning check."""
        prompt_list = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt_list)

        if self.text_encoder_3 is None:
            dtype_fallback = dtype or self.od_config.dtype
            return torch.zeros(
                (
                    batch_size,
                    max_sequence_length,
                    self.transformer.joint_attention_dim,
                ),
                device=self.device,
                dtype=dtype_fallback,
            )

        text_inputs = self.tokenizer_3(
            prompt_list,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).to(self.device)
        text_input_ids = text_inputs.input_ids

        prompt_embeds = self.text_encoder_3(text_input_ids.to(self.device))[0]
        prompt_embeds = prompt_embeds.to(
            dtype=self.text_encoder_3.dtype,
            device=self.device,
        )
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        return prompt_embeds

    def prepare_latents(self, *args, **kwargs):  # type: ignore[override]
        """Initial-noise injection point: bypass upstream RNG when the driver supplied an x_T."""
        upstream = super().prepare_latents
        noise = self.pending_initial_noise
        if noise is not None:
            args, kwargs = inject_latents(upstream, args, kwargs, noise)
            self.pending_initial_noise = None
        return upstream(*args, **kwargs)

    def forward(self, req: DiffusionRequestBatch, **kwargs) -> list[DiffusionOutput]:
        """Single-request batch in, one-element list out."""
        one = single_request(req, caller="RLStableDiffusion3Pipeline.forward")
        extra = one.sampling_params.extra_args
        self.scheduler.arm(eta=one.sampling_params.eta, sde_indices=extra.get("sde_indices"))
        self.pending_initial_noise = resolve_request_noise(one, caller="RLStableDiffusion3Pipeline.forward")
        self.captured_conditioning = None

        outs = super().forward(req, **kwargs)

        out = outs[0]
        drain_trajectory_into(out, self.scheduler)
        if self.captured_conditioning is not None:
            stamp_capture(out, "text_capture", self.captured_conditioning)
        finalize_output(out)
        return outs


__all__ = ["RLStableDiffusion3Pipeline"]
