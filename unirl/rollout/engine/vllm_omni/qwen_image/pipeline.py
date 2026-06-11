"""RL-aware Qwen-Image pipeline subclass for vllm-omni.

Three behaviors on top of upstream ``QwenImagePipeline``
(``vllm_omni/diffusion/models/qwen_image/pipeline_qwen_image.py``), mirroring
:class:`RLStableDiffusion3Pipeline` and :class:`RLHunyuanVideo15Pipeline`:

1. Before the denoise loop: install :class:`FlowMatchSDEDiscreteScheduler` in
   place of upstream's ``FlowMatchEulerDiscreteScheduler`` (captures the dense
   latent trajectory for SDE log-prob replay; at eta=0 degenerates to ODE).
2. After the denoise loop: drain trajectory (latents, sigmas, log_probs) off the
   scheduler, **unpack** the packed token latents back to image form, and stamp
   into ``DiffusionOutput.trajectory_*`` + ``custom_output["sde_step_indices"]``.
3. Capture the single Qwen2.5-VL text-encoder embeddings (``prompt_embeds`` +
   ``prompt_embeds_mask``) from the first ``encode_prompt`` call and stamp into
   ``DiffusionOutput.custom_output["text_capture"]`` for the trainer-side
   :class:`QwenImageConditions` reconstruction.

Why packed→image-form unpack matters
-------------------------------------
Qwen-Image's transformer is a sequence model: ``QwenImagePipeline`` packs latents
to ``[B, S, C_packed=64]`` (16 channels × a 2×2 patch) for the whole denoise loop,
so the trajectory captured by the scheduler is packed too. The trainer-side
``QwenImageDiffusionStage.replay`` expects image-form ``[B, K, C=16, latent_h,
latent_w]`` (it runs its own pack/unpack at the transformer boundary). We unpack
here — using upstream's own ``_unpack_latents`` (the exact inverse of what packed
the loop) — so the generic ``response.py::_build_image_segment`` stays
model-agnostic and the segment matches what replay reads.

σ handling
----------
Upstream ``forward`` already does ``sigmas = req.sampling_params.sigmas or sigmas``
(``pipeline_qwen_image.py:978``) and routes them through ``set_timesteps(sigmas=
...)``. So — unlike HunyuanVideo-1.5 — NO ``set_timesteps`` monkey-patch is
needed; the engine-pinned schedule flows in directly. Our
``FlowMatchSDEDiscreteScheduler.set_timesteps`` neutralizes the diffusers
static-shift double-application when ``sigmas`` is passed (see that class).

CFG scope
---------
Commissioned for CFG-off only (``true_cfg_scale == 1`` → ``do_true_cfg=False``,
single transformer forward), matching the trainside Qwen recipes (guidance=1).
Upstream's CFG-on path applies a one-sided ``cfg_normalize`` clamp, which is NOT
the trainside norm-corrected blend in ``QwenImageDiffusionStep`` — so
guidance>1 would drift the GRPO ratio. The request translator pins
``true_cfg_scale = guidance_scale`` to keep CFG off; do not enable CFG here
without aligning the math.

Step-execution note
-------------------
``QwenImagePipeline`` declares ``supports_step_execution=True``, but the engine
only drives ``denoise_step``/``step_scheduler`` when ``od_config.step_execution``
is True (default False; see ``diffusion_model_runner.py:151`` +
``diffusion/data.py:521``). Our stage YAML leaves it off, so the engine uses the
monolithic ``execute_model`` → ``pipeline.forward(req)`` path and this ``forward``
override's scheduler swap captures the trajectory as intended.

This class is loaded inside vLLM-Omni's worker subprocess via
``custom_pipeline_args.pipeline_class`` injected from
``stage_configs/qwen_image_t2i_rl.yaml``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image import QwenImagePipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest

from unirl.rollout.engine.vllm_omni._shared.flow_match_sde_scheduler import (
    FlowMatchSDEDiscreteScheduler,
)


def _detach_cpu(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Detach + move to CPU for IPC transport. None passthrough."""
    if t is None:
        return None
    return t.detach().to("cpu")


class RLQwenImagePipeline(QwenImagePipeline):
    """Qwen-Image pipeline with SDE trajectory + text-condition capture for RL rollout."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__(od_config=od_config, prefix=prefix)
        # Upstream ``__init__`` builds ``self.scheduler`` (FlowMatchEulerDiscrete);
        # stash it so ``_ensure_scheduler_for_eta`` can rebuild our SDE subclass
        # via ``from_config`` with the same dynamic-shift params.
        self._upstream_scheduler = self.scheduler
        # Text-encoder capture state: reset per-forward, filled on first
        # encode_prompt call.
        self._text_capture: Optional[Dict[str, Any]] = None
        self._encode_prompt_patched: bool = False
        # Per-request initial-noise hand-off (same pattern as SD3 / HV15).
        self._pending_request_noise: Optional[torch.Tensor] = None

    def _ensure_scheduler_for_eta(self, eta: float) -> None:
        """Install our trajectory-capturing scheduler unconditionally.

        Even at eta=0 the scheduler must be installed so the dense latent
        trajectory is captured (``resp_to_samples`` requires non-empty
        ``segment.latents``); the SDE math stays dormant via
        ``_sde_indices_set`` gating.
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
        """Wrap ``encode_prompt`` to capture the single-encoder embeddings.

        Qwen-Image's ``encode_prompt`` returns ``(prompt_embeds,
        prompt_embeds_mask)`` (``pipeline_qwen_image.py:490``). Qwen uses ONE
        text encoder (Qwen2.5-VL) and produces NO pooled vector — the mask is
        load-bearing because prompts are variable-length. We capture both on
        the FIRST call per request (the positive prompt); the negative-prompt
        call (CFG-on only, not commissioned here) is left untouched.

        The 34-token chat-template prefix strip is already applied inside
        upstream ``_get_qwen_prompt_embeds``, so the captured ``prompt_embeds``
        align byte-for-byte with the trainside ``QwenImageTextEmbedStage``
        output (and hence ``QwenImageConditions.text``).
        """
        if self._encode_prompt_patched:
            return

        orig = self.encode_prompt
        pipeline_self = self

        def wrapped(*args: Any, **kw: Any) -> Any:
            result = orig(*args, **kw)
            if pipeline_self._text_capture is None:
                prompt_embeds, prompt_embeds_mask = result
                pipeline_self._text_capture = {
                    "prompt_embeds": _detach_cpu(prompt_embeds),
                    "prompt_embeds_mask": _detach_cpu(prompt_embeds_mask),
                }
            return result

        self.encode_prompt = wrapped  # type: ignore[assignment]
        self._encode_prompt_patched = True

    def _resolve_pending_noise(self, req: "OmniDiffusionRequest") -> None:
        """Look up this request's pre-computed x_T slice from ``initial_noise_batch``.

        ``extra_args["initial_noise_batch"]`` is a single ``[B, C, H_lat, W_lat]``
        image-form tensor shared across the prompt batch; each request is keyed
        by the ``i_`` prefix on ``request_id``. ``None`` (key absent) leaves
        ``_pending_request_noise`` alone so upstream RNG fires as before.
        """
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
                f"RLQwenImagePipeline._resolve_pending_noise: cannot parse batch index from request_id={rid!r}."
            )
        if idx < 0 or idx >= int(noise_batch.shape[0]):
            raise IndexError(
                f"RLQwenImagePipeline._resolve_pending_noise: index {idx} out of "
                f"bounds for noise_batch.shape[0]={int(noise_batch.shape[0])}."
            )
        # Keep the [1, C, H_lat, W_lat] slice (image form); ``prepare_latents``
        # packs it below.
        self._pending_request_noise = noise_batch[idx : idx + 1].clone()

    def prepare_latents(self, *args, **kwargs):  # type: ignore[override]
        """Bypass upstream RNG when the driver supplied an x_T tensor.

        Upstream ``prepare_latents(batch_size, num_channels_latents, height,
        width, dtype, device, generator, latents=None)`` returns early with
        ``latents.to(device, dtype)`` when its ``latents`` kwarg is non-None,
        THEN the caller does not re-pack it — but our pre-shipped tensor is
        image-form ``[1, C, H_lat, W_lat]`` while the loop expects packed
        ``[1, S, C*4]``. Upstream's early-return path (line 535-536) does NOT
        pack, so we must pass a tensor that is ALREADY packed.

        We therefore pack our image-form slice here via the pipeline's own
        ``_pack_latents`` (the exact format the loop consumes) before handing
        it to upstream as the ``latents`` override. Idempotent: the slice is
        consumed after one call so a CFG-driven second call (not used in the
        CFG-off path) falls back to upstream RNG.
        """
        noise = self._pending_request_noise
        if noise is not None:
            # Sniff dtype/device off the call site (positional then kw).
            dtype = args[4] if len(args) > 4 else kwargs.get("dtype")
            device = args[5] if len(args) > 5 else kwargs.get("device")
            if dtype is not None:
                noise = noise.to(dtype=dtype)
            if device is not None:
                noise = noise.to(device=device)
            # Pack image-form [1, C, H_lat, W_lat] -> [1, S, C*4] so it matches
            # what the upstream early-return path returns to the loop.
            b, c, h, w = noise.shape
            packed = self._pack_latents(noise, b, c, h, w)
            if len(args) >= 8:
                args = (*args[:7], packed, *args[8:])
            else:
                kwargs["latents"] = packed
            self._pending_request_noise = None
        return super().prepare_latents(*args, **kwargs)

    def forward(self, req: OmniDiffusionRequest, **kwargs) -> DiffusionOutput:
        # Read eta off the typed field; install our scheduler unconditionally
        # (eta=0 still installs, SDE branch dormant — see method docstring).
        eta = float(getattr(req.sampling_params, "eta", 0.0) or 0.0)
        self._ensure_scheduler_for_eta(eta)

        # Install (or reset) the sparse-SDE step gate on the scheduler. The
        # request translator writes the per-request sparse step list into
        # ``extra_args["sde_indices"]``; re-install every forward so a stale set
        # from a previous request can't leak.
        if isinstance(self.scheduler, FlowMatchSDEDiscreteScheduler):
            extra = getattr(req.sampling_params, "extra_args", None) or {}
            sde_indices = extra.get("sde_indices")
            self.scheduler._sde_indices_set = (
                frozenset(int(i) for i in sde_indices) if sde_indices is not None else None
            )

        # Resolve this request's pre-computed x_T (consumed by prepare_latents).
        self._resolve_pending_noise(req)

        # Reset text capture and install the encode_prompt hook lazily.
        self._text_capture = None
        self._install_encode_prompt_hook()

        # Delegate the full denoise pipeline (encode, latent prep, timestep
        # build, diffuse loop, VAE decode) to upstream. No set_timesteps patch:
        # upstream forward already routes req.sampling_params.sigmas in.
        out = super().forward(req, **kwargs)

        # Drain trajectory off our scheduler and unpack packed tokens -> image
        # form so the generic response builder gets [B, T+1, C, H_lat, W_lat].
        if isinstance(self.scheduler, FlowMatchSDEDiscreteScheduler):
            traj = self.scheduler.drain_trajectory()
            if traj is not None:
                latents, sigmas, _timesteps, log_probs = traj
                # latents: [B, T+1, S, C_packed]. Unpack each step to image form.
                height = req.sampling_params.height or self.default_sample_size * self.vae_scale_factor
                width = req.sampling_params.width or self.default_sample_size * self.vae_scale_factor
                out.trajectory_latents = self._unpack_trajectory(latents, int(height), int(width))
                # ``trajectory_timesteps`` carries the true [0,1] sigma schedule
                # (drained from the SDE scheduler); the response layer reads it
                # back as ``LatentSegment.sigmas``.
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

    def _unpack_trajectory(self, packed: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """``[B, T+1, S, C_packed]`` packed tokens → ``[B, T+1, C, H_lat, W_lat]``.

        Uses the pipeline's own ``_unpack_latents`` (the exact inverse of the
        ``_pack_latents`` the loop ran), applied per trajectory step. The
        resulting image-form latents match what ``QwenImageDiffusionStage.replay``
        reads on the trainer side.
        """
        B, T1 = packed.shape[0], packed.shape[1]
        flat = packed.reshape(B * T1, packed.shape[2], packed.shape[3])
        # upstream _unpack_latents(latents, height, width, vae_scale_factor)
        # returns [B*T1, C, 1, H_lat, W_lat] (5D, video-VAE layout, T=1).
        unpacked = self._unpack_latents(flat, height, width, self.vae_scale_factor)
        # Drop the singleton temporal dim and restore the [B, T+1, ...] axis.
        if unpacked.dim() == 5:
            unpacked = unpacked[:, :, 0]
        c, lh, lw = unpacked.shape[1], unpacked.shape[2], unpacked.shape[3]
        return unpacked.reshape(B, T1, c, lh, lw).contiguous()


__all__ = ["RLQwenImagePipeline"]
