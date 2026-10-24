"""RL-aware HunyuanImage3 pipeline — ``trajectory_timesteps`` carries the true ``[0, 1]`` sigma schedule."""

from __future__ import annotations

from typing import Any, Dict, Optional

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import (
    HunyuanImage3Pipeline,
)
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from unirl.rollout.engine.vllm_omni.pipelines._shared.flow_match_sde_scheduler import (
    FlowMatchSDEDiscreteScheduler,
)
from unirl.rollout.engine.vllm_omni.pipelines._shared.interception import (
    detach_cpu,
    drain_trajectory_into,
    finalize_output,
    single_request,
    stamp_capture,
)
from unirl.types.noise_recipe import NoiseRecipe


class RLHunyuanImage3Pipeline(HunyuanImage3Pipeline):
    """HunyuanImage3 pipeline with the RL interception protocol installed."""

    def __init__(self, od_config: OmniDiffusionConfig) -> None:
        super().__init__(od_config)
        self.captured_conditioning: Optional[Dict[str, Any]] = None
        self.pending_initial_noise_recipe: Optional[NoiseRecipe] = None

    def forward(self, req: DiffusionRequestBatch, **kwargs) -> DiffusionOutput:
        """Single-request batch in, single output out — see ``single_request``."""
        one = single_request(req, caller="RLHunyuanImage3Pipeline.forward")
        inner = self.pipeline
        if not isinstance(self.scheduler, FlowMatchSDEDiscreteScheduler):
            sde = FlowMatchSDEDiscreteScheduler(
                num_train_timesteps=1000,
                shift=self.generation_config.flow_shift,
                use_dynamic_shifting=False,
                base_shift=0.5,
                max_shift=1.15,
                time_shift_type="exponential",
                stochastic_sampling=False,
                eta=0.0,
            )
            self.scheduler = sde
            inner.set_scheduler(sde)

            transformer = inner.model
            orig_prepare_inputs = transformer.prepare_inputs_for_generation
            pipeline_self = self

            def tapped(*args: Any, **kw: Any) -> Any:
                if pipeline_self.captured_conditioning is None:
                    input_ids = args[0] if args else kw.get("input_ids")
                    # Do not capture engine-specific RoPE tables; replay rebuilds them.
                    pipeline_self.captured_conditioning = {
                        "input_ids": detach_cpu(input_ids),
                        "attention_mask": detach_cpu(kw.get("attention_mask")),
                        "position_ids": detach_cpu(kw.get("position_ids")),
                        "gen_image_mask": detach_cpu(kw.get("image_mask")),
                        "gen_timestep_scatter_index": detach_cpu(kw.get("gen_timestep_scatter_index")),
                    }
                return orig_prepare_inputs(*args, **kw)

            transformer.prepare_inputs_for_generation = tapped

            orig_prepare_latents = inner.prepare_latents

            def injecting(batch_size, latent_channel, image_size, dtype, device, generator, latents=None):
                recipe = pipeline_self.pending_initial_noise_recipe
                if latents is None and recipe is not None:
                    lsf = inner.latent_scale_factor
                    if lsf is None:
                        factors = (1,) * len(image_size)
                    elif isinstance(lsf, int):
                        factors = (lsf,) * len(image_size)
                    else:
                        factors = tuple(lsf)
                    per_sample_shape = (
                        latent_channel,
                        *[s // f for s, f in zip(image_size, factors)],
                    )
                    # Reject mismatched noise-group IDs instead of broadcasting the first ID.
                    gids = recipe.noise_group_ids
                    if gids and len(gids) != batch_size:
                        raise RuntimeError(
                            f"RLHunyuanImage3Pipeline.prepare_latents: x_T recipe carries "
                            f"{len(gids)} gid(s) but this DiT call has batch_size={batch_size}. "
                            f"The engine must ship gids aligned to the per-call batch (see "
                            f"VLLMOmniRolloutEngine.generate's dit_recaption per-prompt slice)."
                        )
                    latents = recipe.for_batch(batch_size, latent_shape=per_sample_shape).resolve(
                        device=device, dtype=dtype
                    )
                return orig_prepare_latents(
                    batch_size, latent_channel, image_size, dtype, device, generator, latents=latents
                )

            inner.prepare_latents = injecting

        extra = one.sampling_params.extra_args
        self.scheduler.arm(eta=one.sampling_params.eta, sde_indices=extra.get("sde_indices"))
        gids = extra.get("init_noise_group_ids")
        self.pending_initial_noise_recipe = (
            NoiseRecipe(noise_group_ids=list(gids), base_seed=extra["init_noise_seed"]) if gids else None
        )
        self.captured_conditioning = None

        out = super().forward(req, **kwargs)

        drain_trajectory_into(out, self.scheduler)
        if self.captured_conditioning is not None:
            stamp_capture(out, "fused_mm_capture", self.captured_conditioning)
        finalize_output(out)
        return out


__all__ = ["RLHunyuanImage3Pipeline"]
