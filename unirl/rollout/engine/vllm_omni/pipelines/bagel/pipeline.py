"""RL-aware BAGEL-7B-MoT pipeline subclass.

``forward`` follows the RL interception protocol (see
``pipelines/_shared/interception.py``): **install** (once) → **arm** (every
request) → run (upstream) → **harvest**. BAGEL differs structurally from the
SD3 / Qwen-Image RL pipelines, so the protocol maps onto upstream
(``vllm_omni/diffusion/models/bagel/pipeline_bagel.py`` →
``bagel_transformer.py::Bagel.generate_image``) differently:

- **Scheduler.** Upstream ``generate_image`` ALREADY threads an optional
  ``scheduler`` through its hand-rolled denoise loop and calls
  ``scheduler.step(v_t, timesteps[i], x_t, dts[i])`` reading ``.prev_sample`` /
  ``.log_prob`` — a different contract from diffusers' ``step(...)[0]``. So this
  pipeline does NOT swap a diffusers scheduler; it sets ``self.scheduler`` to the
  BAGEL-specific :class:`...bagel_flow_match_sde_scheduler.BagelFlowSDEScheduler`,
  whose SDE math is byte-identical to the trainside ``FlowSDEStrategy`` so the
  GRPO on-policy ratio stays ≈ 1. The scheduler is the single source of the
  correctly-shaped trajectory (``[1, T+1, seq, C]`` + reconstructed σ + sparse
  SDE-index echo); upstream's own ``trajectory_*`` capture is overwritten on
  harvest.

- **Initial noise.** Upstream draws ``packed_init_noises`` inside
  ``bagel.prepare_vae_latent`` (``torch.randn(num_image_tokens, C·p²)``). The
  driver-authored x_T (per-sample :class:`NoiseRecipe`, packed ``[seq, C]``)
  replaces it via a tap on ``prepare_vae_latent`` — the BAGEL analogue of SD3's
  ``prepare_latents`` override.

- **Conditioning.** BAGEL's conditioning is opaque KV-cache contexts built from
  the prompt through the (frozen) und/text path — NOT a dense tensor, and NOT
  transportable across the worker→driver IPC boundary. So there is NO
  conditioning tap here: the driver-side adapter ships the PROMPTS, and the
  trainer rebuilds the KV contexts on its own bundle at replay time (the und
  path is frozen, so rebuilt contexts are identical regardless of the gen-LoRA
  state). This is the load-bearing design difference from SD3/Qwen, which ship
  dense text embeds.

Everything else — the three-branch CFG context build, the navit packed forward,
the velocity CFG combine, VAE decode — is upstream's ``forward``.

σ contract: BAGEL builds its σ schedule internally from ``(num_timesteps,
timestep_shift)`` and loops ``num_timesteps - 1`` steps. The driver-side adapter
therefore sends ``num_inference_steps = trainside_steps + 1`` (and BAGEL hardwires
``timestep_shift = 3.0`` == the trainside shift) so the worker schedule equals the
engine-pinned ``req.sigmas``; the scheduler reconstructs the visited σ and the
response layer's ``verify_engine_used_sigmas`` asserts the match.

This class is loaded inside vLLM-Omni's worker subprocess via
``custom_pipeline_args.pipeline_class`` injected from
``stage_configs/bagel_t2i_rl.yaml``.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.bagel.pipeline_bagel import BagelPipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest

from unirl.rollout.engine.vllm_omni.pipelines._shared.bagel_flow_match_sde_scheduler import (
    BagelFlowSDEScheduler,
)
from unirl.rollout.engine.vllm_omni.pipelines._shared.interception import (
    drain_trajectory_into,
    resolve_request_noise,
)
from unirl.utils.dtypes import parse_torch_dtype


class RLBagelPipeline(BagelPipeline):
    """BAGEL pipeline with the RL interception protocol installed."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__(od_config=od_config, prefix=prefix)
        # The trajectory-capturing SDE scheduler. ``generate_image`` consumes it
        # via ``self.scheduler`` (upstream reads the attribute directly).
        self._sde_scheduler = BagelFlowSDEScheduler()
        self._sde_scheduler_installed = False
        self._noise_tap_installed = False
        self._rope_fp32_patched = False
        # Per-request x_T hand-off: armed every request, consumed once by the
        # ``prepare_vae_latent`` tap. ``None`` = upstream RNG draw fires.
        self._pending_initial_noise: Optional[torch.Tensor] = None
        # Stored trajectory dtype (matches the trainside trajectory_precision so
        # the SDE log-prob's storage round-trip is identical on both sides).
        self._trajectory_dtype: torch.dtype = torch.float32

    # ------------------------------------------------------------------ #
    # install — once per pipeline lifetime, idempotent
    # ------------------------------------------------------------------ #

    def _install_sde_scheduler(self) -> None:
        """Point ``self.scheduler`` at the SDE scheduler ``generate_image`` reads.

        Always installed (even eta=0 / NFT) — the trainer's clean-latents replay
        requires a non-empty captured trajectory regardless of SDE choice, and
        only this scheduler captures it in the right shape. ``scheduler_kwargs``
        stays empty: the BAGEL loop passes ``(v_t, t, x_t, dt)`` positionally.
        """
        if self._sde_scheduler_installed:
            return
        self.scheduler = self._sde_scheduler
        self.scheduler_kwargs = {}
        self._sde_scheduler_installed = True

    def _install_rope_fp32(self) -> None:
        """Force the rotary cos/sin into fp32 — match the trainside RoPE exactly.

        rollout↔replay forward use SEPARATE codebases. The trainside vendored
        ``Qwen2RotaryEmbedding`` (``vendor/modeling/qwen2/modeling_qwen2.py``)
        computes ``freqs = inv_freq @ position_ids`` + ``cos``/``sin`` inside a
        ``torch.autocast(enabled=False)`` block — forced fp32 (the HF
        transformers#29285 fix). The worker's ``BagelRotaryEmbedding.forward``
        (``vllm_omni/.../bagel_transformer.py``) has NO such guard, and the whole
        ``generate_image`` runs under ``autocast(bf16)``, so its
        ``inv_freq_expanded @ position_ids_expanded`` matmul (despite ``.float()``
        on the operands) gets downcast to bf16 and ``cos()``/``sin()`` run in bf16
        → a per-head rotary that diverges from trainside on EVERY step. That
        systematic rotary gap is a prime suspect for the rollout↔replay log-prob
        drift.

        This wraps the worker rotary instance's ``forward`` so the freqs/trig run
        in an ``autocast(enabled=False)`` block, then casts the result back to the
        input dtype (identical to trainside). Idempotent (sentinel-guarded), and
        scoped to this BAGEL worker — no global vllm patch.
        """
        if self._rope_fp32_patched:
            return
        try:
            rotary = self.bagel.language_model.model.rotary_emb
        except AttributeError:
            # Topology changed under us (e.g. an und-only build); skip rather than
            # crash — the SDE scheduler / noise tap are the load-bearing installs.
            self._rope_fp32_patched = True
            return

        if getattr(rotary, "_unirl_fp32_forward", False):
            self._rope_fp32_patched = True
            return

        orig_forward = rotary.forward

        @torch.no_grad()
        def fp32_forward(x: torch.Tensor, position_ids: torch.Tensor):
            inv_freq_expanded = rotary.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
            position_ids_expanded = position_ids[:, None, :].float()
            device_type = x.device.type
            device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
            # The trainside-matching part: force fp32 for the matmul + trig
            # (autocast off), exactly like vendored Qwen2RotaryEmbedding.
            with torch.autocast(device_type=device_type, enabled=False):
                freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos()
                sin = emb.sin()
            cos = cos * rotary.attention_scaling
            sin = sin * rotary.attention_scaling
            return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

        rotary.forward = fp32_forward  # type: ignore[assignment]
        rotary._unirl_fp32_forward = True  # type: ignore[attr-defined]
        # Keep a handle for debugging / potential revert; never restored in-run.
        rotary._unirl_orig_forward = orig_forward  # type: ignore[attr-defined]
        self._rope_fp32_patched = True

    def _install_noise_tap(self) -> None:
        """Wrap ``bagel.prepare_vae_latent`` to inject the driver-authored x_T.

        Upstream draws ``packed_init_noises`` (``[seq, C·p²]``) inside
        ``prepare_vae_latent``; the tap swaps in the pending driver noise when
        present (consume-once), leaving the rest of the packed inputs untouched.
        Idempotent — the bound method is wrapped once for the worker's lifetime.
        """
        if self._noise_tap_installed:
            return

        orig = self.bagel.prepare_vae_latent
        pipeline_self = self

        def tapped(*args: Any, **kw: Any) -> Any:
            out = orig(*args, **kw)
            noise = pipeline_self._pending_initial_noise
            if noise is not None:
                pipeline_self._pending_initial_noise = None
                ref = out.get("packed_init_noises")
                if ref is None:
                    raise RuntimeError(
                        "RLBagelPipeline noise tap: prepare_vae_latent returned no "
                        "'packed_init_noises' to override."
                    )
                # Driver x_T arrives as [1, seq, C] ([1, ...] slice/recipe row);
                # BAGEL's packed_init_noises is unbatched [seq, C]. Squeeze the
                # leading 1 and validate the packed geometry matches the worker's.
                if noise.dim() == ref.dim() + 1 and int(noise.shape[0]) == 1:
                    noise = noise.squeeze(0)
                if tuple(noise.shape) != tuple(ref.shape):
                    raise RuntimeError(
                        "RLBagelPipeline noise tap: driver x_T shape "
                        f"{tuple(noise.shape)} != worker packed_init_noises shape "
                        f"{tuple(ref.shape)} — check the recipe's "
                        "init_noise_latent_shape (bagel_latent_shape) vs the "
                        "request's height/width."
                    )
                # Match the worker draw's dtype/device; upstream moves the whole
                # dict to device after this returns, so CPU is fine too.
                out["packed_init_noises"] = noise.to(dtype=ref.dtype, device=ref.device)
            return out

        self.bagel.prepare_vae_latent = tapped  # type: ignore[assignment]
        self._noise_tap_installed = True

    # ------------------------------------------------------------------ #
    # arm — every request (stale-leak guards)
    # ------------------------------------------------------------------ #

    def _arm_sde(self, req: OmniDiffusionRequest) -> None:
        """This request's SDE strength + sparse step gate + σ_max + storage dtype."""
        sp = req.sampling_params
        eta = float(getattr(sp, "eta", 0.0) or 0.0)
        extra = getattr(sp, "extra_args", None) or {}
        traj_dtype_name = extra.get("trajectory_precision")
        traj_dtype = (
            parse_torch_dtype(traj_dtype_name, field_name="trajectory_precision")
            if traj_dtype_name
            else self._trajectory_dtype
        )
        # σ_max: the trainside schedule[1] the adapter forwarded. Load-bearing for
        # the first SDE step's std_dev_t clamp — must match trainside or the ratio
        # drifts off 1 (see BagelFlowSDEScheduler.set_for_request).
        sigma_max = extra.get("sigma_max")
        self._sde_scheduler.set_for_request(
            eta=eta,
            sde_indices=extra.get("sde_indices"),
            sigma_max=float(sigma_max) if sigma_max is not None else None,
            trajectory_dtype=traj_dtype,
        )

    def _arm_initial_noise(self, req: OmniDiffusionRequest) -> None:
        """This request's driver-authored x_T (batch slice or recipe row)."""
        self._pending_initial_noise = resolve_request_noise(
            req, caller="RLBagelPipeline._arm_initial_noise"
        )

    # ------------------------------------------------------------------ #
    # harvest — export onto the wire
    # ------------------------------------------------------------------ #

    def _harvest_trajectory(self, out: DiffusionOutput) -> None:
        """Overwrite upstream's trajectory capture with the SDE scheduler's.

        ``drain_trajectory_into`` sets ``trajectory_latents`` ([1, T+1, seq, C]),
        ``trajectory_timesteps`` (the reconstructed [T+1] σ echo) and
        ``trajectory_log_probs`` ([1, K]), and stamps the sparse SDE step ids on
        ``custom_output['sde_step_indices']`` — the exact wire contract
        ``build_image_segment`` consumes (shared with SD3 / Qwen-Image).
        """
        drain_trajectory_into(out, self._sde_scheduler)

    # ------------------------------------------------------------------ #
    # the protocol
    # ------------------------------------------------------------------ #

    def forward(self, req: OmniDiffusionRequest, **kwargs) -> DiffusionOutput:
        self._install_sde_scheduler()
        self._install_noise_tap()
        self._install_rope_fp32()

        self._arm_sde(req)
        self._arm_initial_noise(req)

        # Delegate the full pipeline (CFG context build, navit denoise loop with
        # our scheduler, VAE decode) to upstream; the noise tap fires inside, and
        # the scheduler captures the trajectory as the loop runs.
        out = super().forward(req, **kwargs)

        self._harvest_trajectory(out)
        return out


__all__ = ["RLBagelPipeline"]
