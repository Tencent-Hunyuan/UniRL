"""RL-aware HunyuanImage3 pipeline — ``trajectory_timesteps`` carries the true ``[0, 1]`` sigma schedule."""

from __future__ import annotations

from typing import Any, Dict, Optional

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import (
    HunyuanImage3Pipeline,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
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
        self._upstream_scheduler = None
        self._captured_conditioning: Optional[Dict[str, Any]] = None
        self._conditioning_tap_installed: bool = False
        self._pending_initial_noise_recipe: Optional[NoiseRecipe] = None
        self._initial_noise_injector_installed: bool = False

    def _install_sde_scheduler(self) -> None:
        """Swap in the trajectory-capturing SDE scheduler."""
        _ = self.pipeline

        if self._upstream_scheduler is None:
            self._upstream_scheduler = self.scheduler

        if isinstance(self.scheduler, FlowMatchSDEDiscreteScheduler):
            return

        sde = FlowMatchSDEDiscreteScheduler(
            num_train_timesteps=1000,
            shift=float(self.generation_config.flow_shift),
            use_dynamic_shifting=False,
            base_shift=0.5,
            max_shift=1.15,
            time_shift_type="exponential",
            stochastic_sampling=False,
            eta=0.0,
        )
        self.scheduler = sde
        if self._pipeline is not None:
            self._pipeline.set_scheduler(sde)

    def _install_conditioning_tap(self) -> None:
        """Wrap ``transformer.prepare_inputs_for_generation`` to capture the fused multimodal conditioning."""
        if self._conditioning_tap_installed:
            return

        _ = self.pipeline
        transformer = self._pipeline.model

        orig = transformer.prepare_inputs_for_generation
        pipeline_self = self

        def tapped(*args: Any, **kw: Any) -> Any:
            if pipeline_self._captured_conditioning is None:
                input_ids = args[0] if args else kw.get("input_ids")
                # Do not capture engine-specific RoPE tables; replay rebuilds them.
                pipeline_self._captured_conditioning = {
                    "input_ids": detach_cpu(input_ids),
                    "attention_mask": detach_cpu(kw.get("attention_mask")),
                    "position_ids": detach_cpu(kw.get("position_ids")),
                    "gen_image_mask": detach_cpu(kw.get("image_mask")),
                    "gen_timestep_scatter_index": detach_cpu(kw.get("gen_timestep_scatter_index")),
                }
            return orig(*args, **kw)

        transformer.prepare_inputs_for_generation = tapped
        self._conditioning_tap_installed = True

    def _install_initial_noise_injector(self) -> None:
        """Wrap the inner pipeline's ``prepare_latents`` to inject the driver-authored x_T recipe."""
        if self._initial_noise_injector_installed:
            return
        _ = self.pipeline
        inner = self._pipeline
        orig = inner.prepare_latents
        pipeline_self = self

        def injecting(batch_size, latent_channel, image_size, dtype, device, generator, latents=None):
            recipe = pipeline_self._pending_initial_noise_recipe
            if latents is None and recipe is not None:
                lsf = getattr(inner, "latent_scale_factor", None)
                if lsf is None:
                    factors = (1,) * len(image_size)
                elif isinstance(lsf, int):
                    factors = (lsf,) * len(image_size)
                else:
                    factors = tuple(lsf)
                per_sample_shape = (
                    int(latent_channel),
                    *[int(s) // int(f) for s, f in zip(image_size, factors)],
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
            return orig(batch_size, latent_channel, image_size, dtype, device, generator, latents=latents)

        inner.prepare_latents = injecting
        self._initial_noise_injector_installed = True

    def _arm_sde(self, req: OmniDiffusionRequest) -> None:
        """This request's SDE strength + sparse step gate."""
        eta = float(getattr(req.sampling_params, "eta", 0.0) or 0.0)
        extra = getattr(req.sampling_params, "extra_args", None) or {}
        self.scheduler.arm(eta=eta, sde_indices=extra.get("sde_indices"))

    def _arm_initial_noise(self, req: OmniDiffusionRequest) -> None:
        """This request's x_T RECIPE (seed + per-sample gids, no shape — AR-dynamic; the injector fills it later)."""
        extra = getattr(req.sampling_params, "extra_args", None) or {}
        gids = extra.get("init_noise_group_ids")
        self._pending_initial_noise_recipe = (
            NoiseRecipe(noise_group_ids=[str(g) for g in gids], base_seed=int(extra.get("init_noise_seed", 0)))
            if gids
            else None
        )

    def _arm_conditioning_tap(self) -> None:
        """Fresh capture buffer so the tap records THIS request's first call."""
        self._captured_conditioning = None

    def _harvest_trajectory(self, out: DiffusionOutput) -> None:
        if isinstance(self.scheduler, FlowMatchSDEDiscreteScheduler):
            drain_trajectory_into(out, self.scheduler)

    def _harvest_conditioning(self, out: DiffusionOutput) -> None:
        if self._captured_conditioning is not None:
            stamp_capture(out, "fused_mm_capture", self._captured_conditioning)

    def forward(self, req: DiffusionRequestBatch, **kwargs) -> DiffusionOutput:
        """Single-request batch in, single output out — see ``single_request``."""
        one = single_request(req, caller="RLHunyuanImage3Pipeline.forward")
        # Installs materialize the inner pipeline; they must precede arming.
        self._install_sde_scheduler()
        self._install_conditioning_tap()
        self._install_initial_noise_injector()

        self._arm_sde(one)
        self._arm_initial_noise(one)
        self._arm_conditioning_tap()

        out = super().forward(req, **kwargs)

        self._harvest_trajectory(out)
        self._harvest_conditioning(out)
        finalize_output(out)
        return out


__all__ = ["RLHunyuanImage3Pipeline"]
