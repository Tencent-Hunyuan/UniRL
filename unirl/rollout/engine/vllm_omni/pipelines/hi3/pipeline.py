"""RL-aware HunyuanImage3 pipeline subclass.

``forward`` follows the RL interception protocol (see
``pipelines/_shared/interception.py``): **install** (once) → **arm** (every
request) → run (upstream) → **harvest**. The interceptions, mapped to
upstream's stages
(``vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py:300``):

- SDE scheduler swap (behavior policy + dense-trajectory recorder), built
  with explicit HI3 kwargs and routed through the inner pipeline's
  ``set_scheduler`` hook (``hunyuan_image3_transformer.py:2547-2548``, which
  calls ``register_modules`` so the diffusers component graph stays
  consistent). Installed regardless of eta — ``resp_to_samples`` requires
  ``segment.latents`` and only this scheduler captures the trajectory.
- A conditioning **tap** on the transformer's
  ``prepare_inputs_for_generation``: captures the fused multimodal tensors
  (``input_ids`` / ``attention_mask`` / ``position_ids`` / ``rope_cache`` /
  ``gen_image_mask`` / ``gen_timestep_scatter_index``) on the **first**
  per-request call — subsequent steps under KV-cache reuse pass the
  gathered-down ``L'`` slice which is not what training-side replay needs.
  Read back driver-side as ``conditions["fused"]`` for
  ``HunyuanImage3DiffusionConditions.from_dict``.
- An initial-noise **injection** wrapping the inner pipeline's
  ``prepare_latents``: HI3's DiT latent shape is AR-dynamic (only known once
  upstream resolves ``image_size`` post-AR), so the driver ships a RECIPE
  (seed + per-sample gids), not a tensor; the injector fills the resolved
  shape and regenerates byte-identical x_T via ``NoiseRecipe``.

``trajectory_timesteps`` carries the **true [0, 1] sigma schedule** (what
replay indexes as ``segment.sigmas``); the 1000-scale per-step timesteps are
dropped (regenerable). Exports ride ``trajectory_*`` + ``custom_output``
only — plain runtime attrs on ``DiffusionOutput`` are filtered during the
worker→parent IPC.

Everything else — system-prompt resolution, AR-bridged
``ar_generated_text``, it2i conditioning via ``batch_cond_image_info``,
generator/seed/CFG, the denoise loop itself — is handled by upstream's
``forward`` at ``pipeline_hunyuan_image3.py:1262-1347``.

This class is loaded inside vLLM-Omni's worker subprocess via
``custom_pipeline_args.pipeline_class`` injected from our static stage
configs (``stage_configs/hunyuan_image3_t2i_rl.yaml`` and ``..._it2i_rl.yaml``).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import (
    HunyuanImage3Pipeline,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest

from unirl.rollout.engine.vllm_omni.pipelines._shared.flow_match_sde_scheduler import (
    FlowMatchSDEDiscreteScheduler,
)
from unirl.rollout.engine.vllm_omni.pipelines._shared.interception import (
    detach_cpu,
    detach_cpu_pair,
    drain_trajectory_into,
    stamp_custom_output,
)
from unirl.types.noise_recipe import NoiseRecipe


class RLHunyuanImage3Pipeline(HunyuanImage3Pipeline):
    """HunyuanImage3 pipeline with the RL interception protocol installed."""

    def __init__(self, od_config: OmniDiffusionConfig) -> None:
        super().__init__(od_config)
        # Stashed by the first ``_install_sde_scheduler`` (it has to
        # materialize ``self.pipeline`` first).
        self._upstream_scheduler = None
        # Conditioning-tap state: armed (reset) every request, filled by the
        # tap's first per-request call; the flag keeps the install idempotent.
        self._captured_conditioning: Optional[Dict[str, Any]] = None
        self._conditioning_tap_installed: bool = False
        # Driver-authored x_T recipe (seed + per-sample gids) for THIS
        # request, armed every forward; the injector consumes it. ``None`` →
        # upstream RNG.
        self._pending_initial_noise_recipe: Optional[NoiseRecipe] = None
        self._initial_noise_injector_installed: bool = False

    # ------------------------------------------------------------------ #
    # install — once per pipeline lifetime, idempotent
    # ------------------------------------------------------------------ #

    def _install_sde_scheduler(self) -> None:
        """Swap in the trajectory-capturing SDE scheduler.

        First call materializes ``self._pipeline`` via upstream's
        ``pipeline`` property (``pipeline_hunyuan_image3.py:429-443``), which
        sets ``self.scheduler`` to the upstream Euler scheduler — stash it,
        then swap through the inner pipeline's ``set_scheduler`` hook.
        Explicit HI3 kwargs (no ``from_config`` — the inner pipeline owns the
        flow_shift); per-request eta rides ``_arm_sde``.
        """
        # Force the inner pipeline + upstream scheduler into existence.
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
        """Wrap ``transformer.prepare_inputs_for_generation`` to capture the
        fused multimodal conditioning.

        First-call-only per request: the tap writes ``_captured_conditioning``
        only while it's ``None`` (re-armed each ``forward``) — the first call
        carries the full sequence length ``L``; the gathered-down ``L'``
        calls that follow under KV-cache reuse are ignored.

        The inner ``HunyuanImage3Text2ImagePipeline`` holds the MoE backbone
        on ``.model`` (upstream ``hunyuan_image3_transformer.py:2797``
        dispatches ``self.model.prepare_inputs_for_generation(...)``).
        Despite the diffusers convention of ``pipeline.transformer``, the t2i
        pipeline class doesn't expose that alias — only the outer vllm-omni
        wrapper aliases ``self.transformer = self.model``, a different
        object. Field map matches our own per-step kernel
        (``models/hunyuan_image3/diffusion.py:212-231``).
        """
        if self._conditioning_tap_installed:
            return

        # Force inner pipeline materialization so the model ref is live.
        _ = self.pipeline
        transformer = self._pipeline.model

        orig = transformer.prepare_inputs_for_generation
        pipeline_self = self

        def tapped(*args: Any, **kw: Any) -> Any:
            if pipeline_self._captured_conditioning is None:
                # ``input_ids`` is the only positional arg upstream passes.
                input_ids = args[0] if args else kw.get("input_ids")
                pipeline_self._captured_conditioning = {
                    "input_ids": detach_cpu(input_ids),
                    "attention_mask": detach_cpu(kw.get("attention_mask")),
                    "position_ids": detach_cpu(kw.get("position_ids")),
                    "rope_cache": detach_cpu_pair(kw.get("custom_pos_emb")),
                    "gen_image_mask": detach_cpu(kw.get("image_mask")),
                    "gen_timestep_scatter_index": detach_cpu(kw.get("gen_timestep_scatter_index")),
                }
            return orig(*args, **kw)

        transformer.prepare_inputs_for_generation = tapped
        self._conditioning_tap_installed = True

    def _install_initial_noise_injector(self) -> None:
        """Wrap the inner pipeline's ``prepare_latents`` to inject the
        driver-authored x_T recipe.

        HI3's DiT latent shape is AR-dynamic — only known when upstream calls
        ``prepare_latents`` with the resolved ``image_size`` /
        ``latent_channel`` — so the driver cannot ship a materialized x_T
        tensor; it ships only a RECIPE (seed + per-sample gids) via
        ``sampling_params.extra_args``. The injector recomputes the
        per-sample latent shape exactly as upstream does (mirrors
        ``hunyuan_image3_transformer.py:2489`` — ``latent_scale_factor``
        applied to ``image_size``), regenerates byte-identical noise via
        ``NoiseRecipe.for_batch(...).resolve(...)`` (CPU-fp32 → device), and
        feeds it as ``latents`` so upstream skips its RNG draw. No recipe
        armed → pass-through.

        Note: ``prepare_latents`` lives on the INNER t2i pipeline
        (``hunyuan_image3_transformer.py:2489``), NOT on
        ``self._pipeline.model`` (the MoE backbone, which holds
        ``prepare_inputs_for_generation``).
        """
        if self._initial_noise_injector_installed:
            return
        _ = self.pipeline
        inner = self._pipeline
        orig = inner.prepare_latents
        pipeline_self = self

        def injecting(batch_size, latent_channel, image_size, dtype, device, generator, latents=None):
            recipe = pipeline_self._pending_initial_noise_recipe
            if latents is None and recipe is not None:
                # Resolve the per-sample latent shape from the (post-AR)
                # prepare_latents args, mirroring upstream's own arithmetic —
                # (latent_channel, *[image_size // latent_scale_factor]).
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
                # The recipe must arrive already aligned to THIS call's batch: the
                # engine ships one gid per single-prompt dit_recaption generate
                # (batch_size=1) and one gid per prompt for batched modalities
                # (batch_size=N). ``batch_size`` here is the un-doubled prompt count
                # (any CFG expansion happens later, inside the denoise loop), so a
                # length mismatch means the engine dispatch forwarded the wrong gid
                # slice — fail loud rather than let for_batch silently slice gids[0]
                # onto every image (the x_T-collapse class of bug).
                gids = recipe.noise_group_ids
                if gids and len(gids) != batch_size:
                    raise RuntimeError(
                        f"RLHunyuanImage3Pipeline.prepare_latents: x_T recipe carries "
                        f"{len(gids)} gid(s) but this DiT call has batch_size={batch_size}. "
                        f"The engine must ship gids aligned to the per-call batch (see "
                        f"VLLMOmniRolloutEngine.generate's dit_recaption per-prompt slice)."
                    )
                # Fill the post-AR shape; gids already match the batch (asserted),
                # so for_batch is a no-op slice. Resolution is shared with the
                # trainside pipelines — only the shape's fill site differs.
                latents = recipe.for_batch(batch_size, latent_shape=per_sample_shape).resolve(
                    device=device, dtype=dtype
                )
            return orig(batch_size, latent_channel, image_size, dtype, device, generator, latents=latents)

        inner.prepare_latents = injecting
        self._initial_noise_injector_installed = True

    # ------------------------------------------------------------------ #
    # arm — every request (stale-leak guards)
    # ------------------------------------------------------------------ #

    def _arm_sde(self, req: OmniDiffusionRequest) -> None:
        """This request's SDE strength + sparse step gate."""
        eta = float(getattr(req.sampling_params, "eta", 0.0) or 0.0)
        extra = getattr(req.sampling_params, "extra_args", None) or {}
        self.scheduler.arm(eta=eta, sde_indices=extra.get("sde_indices"))

    def _arm_initial_noise(self, req: OmniDiffusionRequest) -> None:
        """This request's x_T RECIPE (seed + per-sample gids; no shape —
        AR-dynamic, the injector fills it once upstream reveals it)."""
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

    # ------------------------------------------------------------------ #
    # harvest — export onto the wire
    # ------------------------------------------------------------------ #

    def _harvest_trajectory(self, out: DiffusionOutput) -> None:
        if isinstance(self.scheduler, FlowMatchSDEDiscreteScheduler):
            drain_trajectory_into(out, self.scheduler)

    def _harvest_conditioning(self, out: DiffusionOutput) -> None:
        # ``None`` if upstream's ``prepare_inputs_for_generation`` was never
        # called (unexpected for image modalities) — the driver side treats
        # absence as "old-style" empty conditions and raises with the
        # pipeline_class diagnosis.
        if self._captured_conditioning is not None:
            stamp_custom_output(out, "fused_mm_capture", self._captured_conditioning)

    # ------------------------------------------------------------------ #
    # the protocol
    # ------------------------------------------------------------------ #

    def forward(self, req: OmniDiffusionRequest, **kwargs) -> DiffusionOutput:
        # Installs materialize the inner pipeline; they must precede arming.
        self._install_sde_scheduler()
        self._install_conditioning_tap()
        self._install_initial_noise_injector()

        self._arm_sde(req)
        self._arm_initial_noise(req)
        self._arm_conditioning_tap()

        # Delegate everything else (prompt construction, system prompt,
        # AR-bridged cot_text, batch_cond_image_info, prepare_model_inputs,
        # _generate) to upstream forward at pipeline_hunyuan_image3.py:1262;
        # the installed tap/injector fire inside.
        out = super().forward(req, **kwargs)

        self._harvest_trajectory(out)
        self._harvest_conditioning(out)
        return out


__all__ = ["RLHunyuanImage3Pipeline"]
