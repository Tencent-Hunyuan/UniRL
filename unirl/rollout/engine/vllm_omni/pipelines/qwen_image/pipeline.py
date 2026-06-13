"""RL-aware Qwen-Image pipeline subclass.

``forward`` follows the RL interception protocol (see
``pipelines/_shared/interception.py``): **install** (once) → **arm** (every
request) → run (upstream) → **harvest**. The interceptions, mapped to
upstream's stages (``vllm_omni/diffusion/models/qwen_image/pipeline_qwen_image.py``):

- SDE scheduler swap (the behavior policy + dense-trajectory recorder) in
  place of the upstream ``FlowMatchEulerDiscreteScheduler``, installed
  regardless of eta (at eta=0 it degenerates to pure Euler ODE). Upstream's
  ``prepare_timesteps`` reads the dynamic-shift μ constants
  (``base_image_seq_len`` / ``max_shift`` / …) off ``self.scheduler.config``
  and calls ``set_timesteps(sigmas=..., mu=mu)`` — both survive the swap:
  ``from_config`` preserves the config keys and our scheduler accepts ``mu``
  while treating externally pinned σ as final values.
- A conditioning **tap** on ``encode_prompt`` capturing
  ``(prompt_embeds, prompt_embeds_mask)`` for the trainer-side
  ``QwenImageConditions.text``. Upstream calls ``encode_prompt`` once for
  the positive prompt and a second time only under ``do_true_cfg``
  (``_prepare_generation_context``) — the tap routes call 1 to the positive
  slot and call 2 to the negative slot.
- An initial-noise **injection** through the ``prepare_latents`` override —
  the driver-authored x_T (batch slice or recipe) replaces upstream's RNG
  draw. Unlike SD3, the injected tensor must be **packed** first: upstream's
  ``latents is not None`` early-return hands the tensor straight to the
  denoise loop, which runs in the transformer's patchified
  ``[B, S, C*4]`` layout.

Packed-latent boundary
----------------------
The whole upstream denoise loop — and therefore every latent our SDE
scheduler records — lives in packed ``[B, S, C*4]`` space, while the
driver/trainer contract (``LatentSegment.latents``, the x_T recipe, the
trainside ``models/qwen_image/diffusion.py`` replay) is the spatial
``[B, C, H, W]`` shape. This subclass owns both crossings: it packs the
driver's x_T before injection and unpacks the harvested trajectory back to
``[B, T+1, C, H, W]`` (upstream's ``_unpack_latents`` emits a 5D
``[B, C, 1, H, W]`` video-VAE shape; the singleton frame dim is squeezed).

Everything else — prompt encoding (Qwen2.5-VL chat template + 34-token
prefix strip), latent prep, the dynamic-shift timestep build, the diffusion
loop (norm-corrected true CFG when armed), VAE decode — is handled by
upstream's ``forward``.

This class is loaded inside vLLM-Omni's worker subprocess via
``custom_pipeline_args.pipeline_class`` injected from
``stage_configs/qwen_image_t2i_rl.yaml``.
"""

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
        # Upstream ``__init__`` constructs ``self.scheduler``; stash it as
        # the config donor for the SDE swap. We never swap back — our
        # scheduler is installed for the lifetime of this pipeline instance.
        self._upstream_scheduler: FlowMatchEulerDiscreteScheduler = self.scheduler
        # Conditioning-tap state: armed (reset to a fresh dict) every
        # request, filled by the tap's first/second call; the flag keeps the
        # install idempotent.
        self._captured_conditioning: Optional[Dict[str, Any]] = None
        self._conditioning_tap_installed: bool = False
        # Per-request x_T hand-off: armed every request, consumed once by the
        # ``prepare_latents`` override. ``None`` = upstream RNG fires.
        self._pending_initial_noise: Optional[torch.Tensor] = None
        # The request's normalized pixel H/W, stashed by ``forward`` for the
        # harvest-side trajectory unpack.
        self._harvest_hw: Optional[Tuple[int, int]] = None

    # ------------------------------------------------------------------ #
    # install — once per pipeline lifetime, idempotent
    # ------------------------------------------------------------------ #

    def _install_sde_scheduler(self) -> None:
        """Swap in the trajectory-capturing SDE scheduler (the from_config
        path keeps the dynamic-shift config keys ``prepare_timesteps`` reads
        for μ). Always installed, even for eta=0 flows (NFT) — per-request
        eta rides ``_arm_sde``."""
        if isinstance(self.scheduler, FlowMatchSDEDiscreteScheduler):
            return
        self.scheduler = make_sde_scheduler(self._upstream_scheduler.config)

    def _install_conditioning_tap(self) -> None:
        """Wrap ``encode_prompt`` to capture the text conditioning.

        Upstream returns ``(prompt_embeds, prompt_embeds_mask)`` — the
        Qwen2.5-VL last hidden states after the chat-template prefix strip
        plus the matching attention mask (variable-length per prompt; no
        pooled vector exists for Qwen-Image). Call routing per request:
        the first call fills the positive slot, the second (fired by
        upstream only under ``do_true_cfg``) the negative slot; later calls
        are observed but not recorded.
        """
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

    # ------------------------------------------------------------------ #
    # arm — every request (stale-leak guards)
    # ------------------------------------------------------------------ #

    def _arm_sde(self, req: OmniDiffusionRequest) -> None:
        """This request's SDE strength + sparse step gate."""
        eta = float(getattr(req.sampling_params, "eta", 0.0) or 0.0)
        extra = getattr(req.sampling_params, "extra_args", None) or {}
        self.scheduler.arm(eta=eta, sde_indices=extra.get("sde_indices"))

    def _arm_initial_noise(self, req: OmniDiffusionRequest) -> None:
        """This request's driver-authored x_T (batch slice or recipe row),
        still in the spatial ``[1, C, H, W]`` shape — packing happens at the
        injection point where upstream's grid geometry is in hand."""
        self._pending_initial_noise = resolve_request_noise(req, caller="RLQwenImagePipeline._arm_initial_noise")

    def _arm_conditioning_tap(self) -> None:
        """Fresh capture buffer so the tap records THIS request's encodes."""
        self._captured_conditioning = {}

    # ------------------------------------------------------------------ #
    # run-phase interception — upstream-called name, cannot be renamed
    # ------------------------------------------------------------------ #

    def prepare_latents(self, *args, **kwargs):  # type: ignore[override]
        """Initial-noise injection point: bypass upstream RNG when the driver
        supplied an x_T. Upstream's ``latents is not None`` early-return
        skips both the RNG draw and the packing, so the driver's spatial
        ``[B, C, H, W]`` noise is packed here first (the denoise loop runs
        in the transformer's ``[B, S, C*4]`` patch layout). Consume-once.
        """
        noise = self._pending_initial_noise
        if noise is not None:
            self._pending_initial_noise = None
            args, kwargs = inject_latents(args, kwargs, self._pack_pending_noise(noise, args))
        return super().prepare_latents(*args, **kwargs)

    def _pack_pending_noise(self, noise: torch.Tensor, args: tuple) -> torch.Tensor:
        """Spatial ``[B, C, h, w]`` x_T → packed ``[B, S, C*4]``, validated
        against the call site's grid geometry. Upstream calls
        ``prepare_latents(batch_size, num_channels_latents, height, width,
        dtype, device, generator, latents)`` with all args positional and
        pixel-space H/W; the latent grid is ``2 * (px // (vae_sf * 2))``
        per side (the divisible-by-2 packing constraint).
        """
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

    # ------------------------------------------------------------------ #
    # harvest — export onto the wire
    # ------------------------------------------------------------------ #

    def _harvest_trajectory(self, out: DiffusionOutput) -> None:
        if not isinstance(self.scheduler, FlowMatchSDEDiscreteScheduler):
            return
        drain_trajectory_into(out, self.scheduler)
        if out.trajectory_latents is not None:
            out.trajectory_latents = self._unpack_trajectory(out.trajectory_latents)

    def _unpack_trajectory(self, packed: torch.Tensor) -> torch.Tensor:
        """Packed ``[B, T+1, S, C*4]`` trajectory → spatial
        ``[B, T+1, C, H, W]`` (the trainer's ``LatentSegment`` shape).

        Upstream's ``_unpack_latents`` takes pixel H/W and returns the 5D
        video-VAE shape ``[N, C, 1, h, w]``; the singleton frame dim is
        squeezed out to match the trainside spatial convention
        (``models/qwen_image/diffusion.py`` keeps ``[B, K, C, H, W]``).
        """
        if self._harvest_hw is None:
            raise RuntimeError(
                "RLQwenImagePipeline._unpack_trajectory: no stashed H/W — forward() did not run before harvest."
            )
        height, width = self._harvest_hw
        b, t1 = packed.shape[0], packed.shape[1]
        flat = self._unpack_latents(packed.reshape(b * t1, *packed.shape[2:]), height, width, self.vae_scale_factor)
        flat = flat.squeeze(2)  # [B*(T+1), C, 1, h, w] → [B*(T+1), C, h, w]
        return flat.reshape(b, t1, *flat.shape[1:])

    def _harvest_conditioning(self, out: DiffusionOutput) -> None:
        if self._captured_conditioning:
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
        # Mirror upstream forward's H/W resolution (defaults + 16-alignment)
        # so the harvest unpack uses the exact grid the loop ran on.
        height = req.sampling_params.height or self.default_sample_size * self.vae_scale_factor
        width = req.sampling_params.width or self.default_sample_size * self.vae_scale_factor
        height, width = normalize_min_aligned_size(height, width, self.vae_scale_factor * 2)
        self._harvest_hw = (int(height), int(width))

        # Delegate the entire denoise pipeline (prompt encoding, latent prep,
        # timestep build, diffusion loop, VAE decode) to upstream; the
        # installed tap/injector fire inside.
        out = super().forward(req, **kwargs)

        self._harvest_trajectory(out)
        self._harvest_conditioning(out)
        return out


__all__ = ["RLQwenImagePipeline"]
