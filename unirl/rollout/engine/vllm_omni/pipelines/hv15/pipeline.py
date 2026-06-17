"""RL-aware HunyuanVideo-1.5 pipeline subclass.

``forward`` follows the RL interception protocol (see
``pipelines/_shared/interception.py``): **install** (once) → **arm** (every
request) → run (upstream) → **harvest**. The interceptions, mapped to
upstream's stages
(``vllm_omni/diffusion/models/hunyuan_video/pipeline_hunyuan_video_1_5.py``):

- SDE scheduler swap (behavior policy + dense-trajectory recorder) in place
  of the upstream scheduler; installed regardless of eta — at eta=0 the SDE
  math is dormant but the per-step ``prev_sample`` capture still fires
  (``resp_to_samples`` requires ``segment.latents``).
- A conditioning **tap** on ``encode_prompt``: captures the dual
  text-encoder embeddings (Qwen2.5-VL MLLM + ByT5 glyph, 8 tensors) for the
  trainer-side ``HunyuanVideo15Conditions`` reconstruction.
- An initial-noise **injection** through the ``prepare_latents`` override
  (driver-authored x_T slice or recipe row replaces upstream's RNG draw).
- A σ-schedule **workaround**: upstream HV1.5 ignores
  ``req.sampling_params.sigmas`` (see :meth:`_sigma_override`).

This class is loaded inside vLLM-Omni's worker subprocess via
``custom_pipeline_args.pipeline_class`` injected from
``stage_configs/hunyuan_video15_t2v_rl.yaml``.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

import torch
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.hunyuan_video.pipeline_hunyuan_video_1_5 import (
    HunyuanVideo15Pipeline,
)
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


class RLHunyuanVideo15Pipeline(HunyuanVideo15Pipeline):
    """HunyuanVideo-1.5 pipeline with the RL interception protocol installed."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__(od_config=od_config, prefix=prefix)
        # Config donor for the SDE swap.
        self._upstream_scheduler = self.scheduler
        # Conditioning-tap state: armed (reset) every request, filled by the
        # tap's first call; the flag keeps the install idempotent.
        self._captured_conditioning: Optional[Dict[str, Any]] = None
        self._conditioning_tap_installed: bool = False
        # Per-request x_T hand-off (same pattern as SD3).
        self._pending_initial_noise: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ #
    # install — once per pipeline lifetime, idempotent
    # ------------------------------------------------------------------ #

    def _install_sde_scheduler(self) -> None:
        """Swap in the trajectory-capturing SDE scheduler (from_config keeps
        the upstream schedule parameters). Per-request eta rides ``_arm_sde``."""
        if isinstance(self.scheduler, FlowMatchSDEDiscreteScheduler):
            return
        self.scheduler = make_sde_scheduler(self._upstream_scheduler.config)

    def _install_conditioning_tap(self) -> None:
        """Wrap ``encode_prompt`` to capture the dual text-encoder embeddings.

        HunyuanVideo-1.5 returns 8 values from ``encode_prompt``:
        ``(prompt_embeds, prompt_embeds_mask, prompt_embeds_2,
        prompt_embeds_mask_2, negative_*, …)``. First-call-only per request
        (the buffer is re-armed each ``forward``).
        """
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
                    # MLLM (Qwen2.5-VL) text encoder output
                    "prompt_embeds": detach_cpu(prompt_embeds),
                    "prompt_embeds_mask": detach_cpu(prompt_embeds_mask),
                    # ByT5 glyph encoder output
                    "prompt_embeds_2": detach_cpu(prompt_embeds_2),
                    "prompt_embeds_mask_2": detach_cpu(prompt_embeds_mask_2),
                    # Negative (for CFG)
                    "negative_prompt_embeds": detach_cpu(neg_embeds),
                    "negative_prompt_embeds_mask": detach_cpu(neg_mask),
                    "negative_prompt_embeds_2": detach_cpu(neg_embeds_2),
                    "negative_prompt_embeds_mask_2": detach_cpu(neg_mask_2),
                }
            return result

        self.encode_prompt = tapped  # type: ignore[assignment]
        self._conditioning_tap_installed = True

    # ------------------------------------------------------------------ #
    # arm — every request (stale-leak guards)
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    # run-phase interceptions
    # ------------------------------------------------------------------ #

    def prepare_latents(self, *args, **kwargs):  # type: ignore[override]
        """Initial-noise injection point (consume-once; upstream signature:
        ``(batch_size, height, width, num_frames, dtype, device, generator,
        latents)`` — same dtype@4/device@5/latents@7 slots as SD3)."""
        noise = self._pending_initial_noise
        if noise is not None:
            args, kwargs = inject_latents(args, kwargs, noise)
            self._pending_initial_noise = None
        return super().prepare_latents(*args, **kwargs)

    @contextmanager
    def _sigma_override(self, req: OmniDiffusionRequest) -> Iterator[None]:
        """WORKAROUND: make upstream pick up the engine's σ schedule.

        Upstream HunyuanVideo15Pipeline hardcodes
        ``sigmas = np.linspace(1.0, 0.0, num_steps + 1)[:-1]`` before its
        single ``scheduler.set_timesteps(sigmas=sigmas, ...)`` call, ignoring
        ``req.sampling_params.sigmas`` (every other vllm-omni model does
        ``sigmas = req.sampling_params.sigmas or sigmas`` — see qwen_image,
        sd3, flux*, z_image — HV1.5 is the outlier). Without this the worker
        runs an unshifted linear σ schedule while the engine sent a shift=5.0
        flow-match schedule, tripping ``sigma_verify`` (max-abs-diff ~0.38)
        and aborting rollout.

        Patches the scheduler's ``set_timesteps`` for the duration of ONE
        ``forward`` and always restores — the closure must never leak across
        requests. Delete once upstream honors the request sigmas.
        """
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

    # ------------------------------------------------------------------ #
    # harvest — export onto the wire
    # ------------------------------------------------------------------ #

    def _harvest_trajectory(self, out: DiffusionOutput) -> None:
        if isinstance(self.scheduler, FlowMatchSDEDiscreteScheduler):
            drain_trajectory_into(out, self.scheduler)

    def _harvest_conditioning(self, out: DiffusionOutput) -> None:
        if self._captured_conditioning is not None:
            stamp_custom_output(out, "text_capture", self._captured_conditioning)

    # ------------------------------------------------------------------ #
    # the protocol
    # ------------------------------------------------------------------ #

    def forward(self, req: OmniDiffusionRequest, **kwargs) -> DiffusionOutput:
        self._install_sde_scheduler()
        self._install_conditioning_tap()

        self._arm_sde(req)
        self._arm_initial_noise(req)
        self._arm_conditioning_tap()

        # Delegate to upstream (encode, latent prep, denoise loop, VAE
        # decode); the installed tap/injector fire inside.
        with self._sigma_override(req):
            out = super().forward(req, **kwargs)

        # The engine post-processes out.output into PIL frames, but for video
        # those do NOT survive the worker->client wire — only tensors carried in
        # custom_output / trajectory_* cross; PIL image lists are dropped, so the
        # trainer-side response would see empty ``images`` (LIN-382). Stamp the
        # decoded video tensor (CHW-by-frame, [B, C, F, H, W]) onto custom_output
        # so ``collect_dit_outputs`` can recover frames for the reward.
        decoded = getattr(out, "output", None)
        if decoded is not None:
            stamp_custom_output(out, "rl_decoded_video", detach_cpu(decoded))

        self._harvest_trajectory(out)
        self._harvest_conditioning(out)
        return out


__all__ = ["RLHunyuanVideo15Pipeline"]
