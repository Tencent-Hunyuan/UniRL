"""RL-aware HunyuanVideo-1.5 pipeline subclass for vllm-omni.

Three behaviors on top of upstream ``HunyuanVideo15Pipeline``
(``vllm_omni/diffusion/models/hunyuan_video/pipeline_hunyuan_video_1_5.py``):

1. Before the denoise loop: install :class:`FlowMatchSDEDiscreteScheduler`
   in place of the upstream scheduler (captures dense latent trajectory for
   SDE log-prob replay on the trainer side; at eta=0 degenerates to ODE).
2. After the denoise loop: drain trajectory (latents, sigmas, log_probs)
   off the scheduler and stamp into ``DiffusionOutput.trajectory_*``.
3. Capture dual text-encoder embeddings (Qwen2.5-VL MLLM + ByT5 glyph)
   from the first ``encode_prompt`` call and stamp into
   ``DiffusionOutput.custom_output["text_capture"]`` for the trainer-side
   ``HunyuanVideo15Conditions`` reconstruction.

This class is loaded inside vLLM-Omni's worker subprocess via
``custom_pipeline_args.pipeline_class`` injected from
``stage_configs/hunyuan_video15_t2v_rl.yaml``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.hunyuan_video.pipeline_hunyuan_video_1_5 import (
    HunyuanVideo15Pipeline,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest

from unirl.rollout.engine.vllm_omni._shared.flow_match_sde_scheduler import (
    FlowMatchSDEDiscreteScheduler,
)


def _detach_cpu(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Detach + move to CPU for IPC transport. None passthrough."""
    if t is None:
        return None
    return t.detach().to("cpu")


class RLHunyuanVideo15Pipeline(HunyuanVideo15Pipeline):
    """HunyuanVideo-1.5 pipeline with SDE trajectory + text-condition capture for RL rollout."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__(od_config=od_config, prefix=prefix)
        # Stash upstream scheduler for SDE scheduler construction via from_config.
        self._upstream_scheduler = self.scheduler
        # Text-encoder capture state: reset per-forward, filled on first encode_prompt call.
        self._text_capture: Optional[Dict[str, Any]] = None
        self._encode_prompt_patched: bool = False
        # Per-request initial-noise hand-off (same pattern as SD3).
        self._pending_request_noise: Optional[torch.Tensor] = None

    def _ensure_scheduler_for_eta(self, eta: float) -> None:
        """Install our trajectory-capturing scheduler unconditionally.

        Even at eta=0, we need the scheduler installed to capture the dense
        latent trajectory (required by ``resp_to_samples``). The SDE math is
        dormant when no step is SDE-gated; the per-step ``prev_sample``
        capture still fires.
        """
        if isinstance(self.scheduler, FlowMatchSDEDiscreteScheduler):
            self.scheduler._eta = float(eta)
            return
        sde = FlowMatchSDEDiscreteScheduler.from_config(
            self._upstream_scheduler.config,
            eta=float(eta),
        )
        self.scheduler = sde

    def _install_encode_prompt_hook(self) -> None:
        """Wrap ``encode_prompt`` to capture dual text-encoder embeddings.

        HunyuanVideo-1.5 returns 8 values from encode_prompt:
          (prompt_embeds, prompt_embeds_mask,
           prompt_embeds_2, prompt_embeds_mask_2,
           negative_prompt_embeds, negative_prompt_embeds_mask,
           negative_prompt_embeds_2, negative_prompt_embeds_mask_2)

        We capture the first 4 (positive branch) on the first call per request.
        """
        if self._encode_prompt_patched:
            return

        orig = self.encode_prompt
        pipeline_self = self

        def wrapped(*args: Any, **kw: Any) -> Any:
            result = orig(*args, **kw)
            if pipeline_self._text_capture is None:
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
                pipeline_self._text_capture = {
                    # MLLM (Qwen2.5-VL) text encoder output
                    "prompt_embeds": _detach_cpu(prompt_embeds),
                    "prompt_embeds_mask": _detach_cpu(prompt_embeds_mask),
                    # ByT5 glyph encoder output
                    "prompt_embeds_2": _detach_cpu(prompt_embeds_2),
                    "prompt_embeds_mask_2": _detach_cpu(prompt_embeds_mask_2),
                    # Negative (for CFG)
                    "negative_prompt_embeds": _detach_cpu(neg_embeds),
                    "negative_prompt_embeds_mask": _detach_cpu(neg_mask),
                    "negative_prompt_embeds_2": _detach_cpu(neg_embeds_2),
                    "negative_prompt_embeds_mask_2": _detach_cpu(neg_mask_2),
                }
            return result

        self.encode_prompt = wrapped  # type: ignore[assignment]
        self._encode_prompt_patched = True

    def _resolve_pending_noise(self, req: "OmniDiffusionRequest") -> None:
        """Look up this request's pre-computed x_T slice from initial_noise_batch."""
        extra = getattr(req.sampling_params, "extra_args", None) or {}
        noise_batch = extra.get("initial_noise_batch")
        if noise_batch is None:
            self._pending_request_noise = None
            return
        rid = str(getattr(req, "request_id", "") or "")
        try:
            idx = int(rid.split("_", 1)[0])
        except ValueError:
            raise RuntimeError(
                f"RLHunyuanVideo15Pipeline._resolve_pending_noise: cannot parse batch index from request_id={rid!r}."
            )
        if idx < 0 or idx >= int(noise_batch.shape[0]):
            raise IndexError(
                f"RLHunyuanVideo15Pipeline._resolve_pending_noise: index "
                f"{idx} out of bounds for noise_batch.shape[0]="
                f"{int(noise_batch.shape[0])}."
            )
        self._pending_request_noise = noise_batch[idx : idx + 1].clone()

    def prepare_latents(self, *args, **kwargs):  # type: ignore[override]
        """Bypass upstream RNG when driver supplied an x_T tensor."""
        noise = self._pending_request_noise
        if noise is not None:
            # HunyuanVideo15Pipeline.prepare_latents signature:
            # (batch_size, height, width, num_frames, dtype, device, generator, latents)
            dtype = args[4] if len(args) > 4 else kwargs.get("dtype")
            device = args[5] if len(args) > 5 else kwargs.get("device")
            if dtype is not None:
                noise = noise.to(dtype=dtype)
            if device is not None:
                noise = noise.to(device=device)
            if len(args) >= 8:
                args = (*args[:7], noise, *args[8:])
            else:
                kwargs["latents"] = noise
            self._pending_request_noise = None
        return super().prepare_latents(*args, **kwargs)

    def forward(self, req: OmniDiffusionRequest, **kwargs) -> DiffusionOutput:
        # Read eta from sampling params.
        eta = float(getattr(req.sampling_params, "eta", 0.0) or 0.0)
        self._ensure_scheduler_for_eta(eta)

        # Install SDE step gate on scheduler.
        if isinstance(self.scheduler, FlowMatchSDEDiscreteScheduler):
            extra = getattr(req.sampling_params, "extra_args", None) or {}
            sde_indices = extra.get("sde_indices")
            self.scheduler._sde_indices_set = (
                frozenset(int(i) for i in sde_indices) if sde_indices is not None else None
            )

        # Resolve pre-computed initial noise.
        self._resolve_pending_noise(req)

        # Reset and install text-encoder capture hook.
        self._text_capture = None
        self._install_encode_prompt_hook()

        # Inject engine-supplied sigmas. Upstream HunyuanVideo15Pipeline
        # hardcodes ``sigmas = np.linspace(1.0, 0.0, num_steps + 1)[:-1]``
        # before its single ``scheduler.set_timesteps(sigmas=sigmas, ...)``
        # call, ignoring ``req.sampling_params.sigmas`` (every other
        # vllm-omni model does ``sigmas = req.sampling_params.sigmas or
        # sigmas`` — see qwen_image, sd3, flux*, z_image — HV1.5 is the
        # outlier). Without this swap the worker runs an unshifted linear σ
        # schedule while the engine sent a shift=5.0 flow-match schedule,
        # which trips ``sigma_verify`` with a max-abs-diff ~0.38 and aborts
        # rollout.
        #
        # We monkey-patch the scheduler's ``set_timesteps`` for the
        # duration of this call so upstream's call site transparently
        # picks up our σ.
        engine_sigmas = getattr(req.sampling_params, "sigmas", None)
        sigma_patch_active = False
        if engine_sigmas is not None:
            sched = self.scheduler
            orig_set_timesteps = sched.set_timesteps

            def _set_timesteps_with_engine_sigmas(*args, **kw):
                kw["sigmas"] = engine_sigmas
                return orig_set_timesteps(*args, **kw)

            sched.set_timesteps = _set_timesteps_with_engine_sigmas  # type: ignore[assignment]
            sigma_patch_active = True

        try:
            # Delegate to upstream (encode, latent prep, denoise loop, VAE decode).
            out = super().forward(req, **kwargs)
        finally:
            if sigma_patch_active:
                # Restore so we never leak the closure across requests.
                del self.scheduler.set_timesteps

        # Drain trajectory from our scheduler.
        if isinstance(self.scheduler, FlowMatchSDEDiscreteScheduler):
            traj = self.scheduler.drain_trajectory()
            if traj is not None:
                latents, sigmas, _timesteps, log_probs = traj
                out.trajectory_latents = latents
                out.trajectory_timesteps = sigmas
                out.trajectory_log_probs = log_probs
                sde_step_indices = self.scheduler.last_sde_step_indices
                if out.custom_output is None:
                    out.custom_output = {}
                out.custom_output["sde_step_indices"] = sde_step_indices

        # Surface captured text embeds for trainer-side conditions reconstruction.
        if self._text_capture is not None:
            if out.custom_output is None:
                out.custom_output = {}
            out.custom_output["text_capture"] = self._text_capture

        return out


__all__ = ["RLHunyuanVideo15Pipeline"]
