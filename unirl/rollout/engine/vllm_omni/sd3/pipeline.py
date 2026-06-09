"""RL-aware StableDiffusion 3.5 pipeline subclass.

Three behaviors on top of upstream ``StableDiffusion3Pipeline``
(``vllm_omni/diffusion/models/sd3/pipeline_sd3.py:132``):

1. Before ``super().forward(req)``: unconditionally install
   :class:`FlowMatchSDEDiscreteScheduler` in place of the upstream
   ``FlowMatchEulerDiscreteScheduler``, regardless of eta. Our scheduler
   captures the dense latent trajectory (required by
   ``resp_to_samples``); when no step is SDE-gated it degenerates to
   pure Euler ODE, so installing it at eta=0 has no behavioural cost.
   The pipeline then propagates ``req.sampling_params.extra_args["sde_indices"]``
   onto the scheduler so per-step SDE/ODE gating runs.
2. After ``super().forward(req)`` returns: drain the captured
   trajectory off the scheduler and stamp into
   ``DiffusionOutput.trajectory_{latents,timesteps,log_probs}``.
3. Capture the encoded text embeddings (``prompt_embeds`` +
   ``pooled_prompt_embeds``) on the first ``encode_prompt`` call per
   request and stamp into ``DiffusionOutput.custom_output["text_capture"]``.
   Required by the training side: ``SD3DiffusionStage.replay`` consumes
   ``SD3Conditions.text`` (a typed ``TextEmbedCondition``) to recompute
   per-step log-probs, but the rollout actor runs in a separate process
   and can't share the encoder. Mirrors the HI3 fused-MM capture pattern
   (see ``hi3/pipeline.py:_install_prepare_inputs_hook``).

Everything else — prompt encoding (CLIP-L + CLIP-G + T5), latent prep,
dynamic-shift timestep build, the diffusion loop itself, VAE decode
with shift_factor — is handled by upstream's ``forward`` at
``pipeline_sd3.py:610-737``.

This class is loaded inside vLLM-Omni's worker subprocess via
``custom_pipeline_args.pipeline_class`` injected from
``stage_configs/sd35_t2i_rl.yaml``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.sd3.pipeline_sd3 import StableDiffusion3Pipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest

from unirl.rollout.engine.vllm_omni._shared.flow_match_sde_scheduler import (
    FlowMatchSDEDiscreteScheduler,
)
from unirl.types.noise_recipe import NoiseRecipe


def _detach_cpu(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Detach + move to CPU for IPC transport. None passthrough."""
    if t is None:
        return None
    return t.detach().to("cpu")


class RLStableDiffusion3Pipeline(StableDiffusion3Pipeline):
    """SD3.5 pipeline with SDE trajectory + text-condition capture for RL rollout."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__(od_config=od_config, prefix=prefix)
        # Upstream ``__init__`` constructs ``self.scheduler`` at
        # ``pipeline_sd3.py:191``; stash it so ``_ensure_scheduler_for_eta``
        # can build our subclass via ``from_config(...)`` (same dynamic-shift
        # parameters). We never swap back — our scheduler is installed for
        # the lifetime of this pipeline instance.
        self._upstream_scheduler: FlowMatchEulerDiscreteScheduler = self.scheduler
        # Text-encoder capture state. ``_text_capture`` is reset to ``None``
        # at the top of every ``forward`` and filled by the first
        # ``encode_prompt`` call; ``_encode_prompt_patched`` is the
        # idempotent install flag.
        self._text_capture: Optional[Dict[str, Any]] = None
        self._encode_prompt_patched: bool = False
        # T5 truncation-warning fix flag (see ``_install_t5_truncation_fix``).
        self._t5_truncation_fix_installed: bool = False
        # Per-request initial-noise hand-off. ``forward(req)`` resolves
        # ``req.sampling_params.extra_args["initial_noise_batch"]`` (a
        # ``[B, C, H_lat, W_lat]`` tensor shared across the batch) and
        # picks this request's row by ``request_id`` index prefix, stashing
        # it here; the overridden ``prepare_latents`` then returns it
        # instead of calling upstream's RNG. ``None`` is the "no override"
        # marker — upstream RNG kicks in as before.
        self._pending_request_noise: Optional[torch.Tensor] = None

    def _ensure_scheduler_for_eta(self, eta: float) -> None:
        """Install our trajectory-capturing scheduler regardless of ``eta``.

        Reuses the upstream scheduler's config so dynamic shifting (read
        by ``prepare_timesteps`` at ``pipeline_sd3.py:507``) continues to
        work — ``from_config`` on a ``SchedulerMixin`` subclass re-invokes
        ``__init__`` with the same kwargs the parent was built with,
        plus our ``eta``. SD3 has a single ``self.scheduler`` attribute
        (no inner ``self._pipeline.scheduler`` like HI3); a plain
        reassignment is sufficient.

        Rationale for always installing (even at ``eta == 0``):
        ``resp_to_samples`` requires ``segment.latents`` to be non-empty. The
        only thing that captures the dense latent trajectory inside the
        worker is THIS scheduler — upstream's
        ``FlowMatchEulerDiscreteScheduler`` doesn't stash anything. So
        DiffusionNFT / forward-process flows (which want ``eta == 0`` + no SDE
        log-prob capture) still need us installed: the SDE math is
        dormant (gated on ``_sde_indices_set``), but the per-step
        ``prev_sample`` capture still fires.
        """
        if isinstance(self.scheduler, FlowMatchSDEDiscreteScheduler):
            # Already installed — just retune eta in place.
            self.scheduler._eta = float(eta)
            return
        sde = FlowMatchSDEDiscreteScheduler.from_config(
            self._upstream_scheduler.config,
            eta=float(eta),
        )
        self.scheduler = sde

    def _install_encode_prompt_hook(self) -> None:
        """Idempotently wrap ``self.encode_prompt`` to capture text embeds.

        Runs in front of every ``encode_prompt`` call but only writes
        ``self._text_capture`` when it's ``None`` — reset to ``None`` at
        the top of every ``forward(req)``. Net effect: capture exactly the
        **first** call per request (the positive prompt encode). Upstream
        may call ``encode_prompt`` a second time for negative prompts when
        CFG is active; that path is left untouched.

        Upstream returns ``(prompt_embeds, pooled_prompt_embeds)`` (see
        ``pipeline_sd3.py:418``). Both are needed: ``prompt_embeds`` is the
        joint CLIP-L+CLIP-G+T5 sequence ([B, L, D]) used as the cross-attn
        K/V on the DiT; ``pooled_prompt_embeds`` is the pooled scalar
        condition ([B, D_pooled]) used by the AdaLN modulation.
        """
        if self._encode_prompt_patched:
            return

        orig = self.encode_prompt
        pipeline_self = self

        def wrapped(*args: Any, **kw: Any) -> Any:
            result = orig(*args, **kw)
            if pipeline_self._text_capture is None:
                prompt_embeds, pooled_prompt_embeds = result
                pipeline_self._text_capture = {
                    "prompt_embeds": _detach_cpu(prompt_embeds),
                    "pooled_prompt_embeds": _detach_cpu(pooled_prompt_embeds),
                }
            return result

        self.encode_prompt = wrapped  # type: ignore[assignment]
        self._encode_prompt_patched = True

    def _install_t5_truncation_fix(self) -> None:
        """Replace ``_get_t5_prompt_embeds`` to skip a cross-device warning check.

        Upstream builds the truncated token ids on ``self.device`` but leaves
        the untruncated ids on CPU before calling ``torch.equal`` for a warning
        about prompt truncation. The warning path is only informational and can
        crash long-prompt rollouts, so this patch drops that branch while
        preserving the embedding path.
        """
        if self._t5_truncation_fix_installed:
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
        self._t5_truncation_fix_installed = True

    def _resolve_pending_noise(self, req: "OmniDiffusionRequest") -> None:
        """Look up this request's pre-computed x_T slice and stash on ``self``.

        ``extra_args["initial_noise_batch"]`` is a single ``[B, C, H_lat, W_lat]``
        tensor shared across the prompt batch (Omni broadcasts the same
        ``OmniDiffusionSamplingParams`` to every request in a generate call).
        Each request is keyed by ``request_id = f"{i}_{uuid4()}"`` in
        ``vllm_omni/.../entrypoints/omni.py:115``; we parse the ``i``
        prefix to pick this request's row. ``None`` (key absent) leaves
        ``_pending_request_noise`` alone — upstream RNG fires as before.
        """
        extra = getattr(req.sampling_params, "extra_args", None) or {}
        noise_batch = extra.get("initial_noise_batch")
        if noise_batch is None:
            # Driver ships the x_T RECIPE — if it rode in, regenerate THIS request's
            # row on CPU-fp32 (generate_shared_noise keys each gid by its own seeded
            # generator, so regenerating only gids[idx] reproduces row idx exactly).
            recipe_gids = extra.get("init_noise_group_ids")
            if recipe_gids:
                rid = str(getattr(req, "request_id", "") or "")
                try:
                    idx = int(rid.split("_", 1)[0])
                except ValueError:
                    raise RuntimeError(
                        f"RLStableDiffusion3Pipeline._resolve_pending_noise: cannot "
                        f"parse batch index from request_id={rid!r}. Expected "
                        f"Omni's ``f'{{i}}_{{uuid}}'`` shape."
                    )
                if idx < 0 or idx >= len(recipe_gids):
                    raise IndexError(
                        f"RLStableDiffusion3Pipeline._resolve_pending_noise: index "
                        f"{idx} out of bounds for init_noise_group_ids len={len(recipe_gids)}."
                    )
                self._pending_request_noise = NoiseRecipe(
                    noise_group_ids=[str(recipe_gids[idx])],
                    base_seed=int(extra.get("init_noise_seed", 0)),
                    latent_shape=tuple(extra["init_noise_latent_shape"]),
                ).resolve()  # [1, C, H, W] — matches the noise_batch[idx:idx+1] slice shape
                return
            self._pending_request_noise = None
            return
        rid = str(getattr(req, "request_id", "") or "")
        try:
            idx = int(rid.split("_", 1)[0])
        except ValueError:
            raise RuntimeError(
                f"RLStableDiffusion3Pipeline._resolve_pending_noise: cannot "
                f"parse batch index from request_id={rid!r}. Expected "
                f"Omni's ``f'{{i}}_{{uuid}}'`` shape."
            )
        if idx < 0 or idx >= int(noise_batch.shape[0]):
            raise IndexError(
                f"RLStableDiffusion3Pipeline._resolve_pending_noise: index "
                f"{idx} out of bounds for noise_batch.shape[0]="
                f"{int(noise_batch.shape[0])}."
            )
        # Keep the [1, C, H, W] slice (don't squeeze) — upstream's
        # ``prepare_latents`` shape ``(B, C, H, W)`` is what the denoise
        # loop expects, and B == 1 per request here (``num_outputs_per_prompt=1``).
        self._pending_request_noise = noise_batch[idx : idx + 1].clone()

    def prepare_latents(self, *args, **kwargs):  # type: ignore[override]
        """Bypass upstream RNG when the driver has supplied an x_T tensor.

        Upstream ``prepare_latents`` only calls ``randn_tensor`` when its
        ``latents`` kwarg is ``None`` (see vllm-omni's
        ``diffusion/models/sd3/pipeline_sd3.py``); by injecting our
        pre-computed slice in place of upstream's ``None`` we skip the
        draw and otherwise let upstream's body run unchanged. (Upstream
        does NOT do the diffusers-style
        ``latents *= scheduler.init_noise_sigma`` scaling — Flow-Match
        schedulers already produce unit-variance noise at t=1, so the
        tensor we pass in IS the start-of-denoise state.) Idempotent:
        ``_pending_request_noise`` is cleared after one consumption so
        a follow-up call (e.g. CFG negative branch) doesn't reuse the
        slice.

        Argument-shape note: SD3 upstream calls
        ``self.prepare_latents(batch_size, num_channels_latents, height,
        width, dtype, device, generator, latents)`` with **all eight
        args positional** (``pipeline_sd3.py:709-718``). Writing
        ``kwargs["latents"] = noise`` here while ``latents`` is already
        positional at ``args[7]`` raises
        ``TypeError: prepare_latents() got multiple values for argument
        'latents'``. We detect the positional case and replace
        ``args[7]`` in-place; only when fewer than 8 positionals were
        passed do we fall back to ``kwargs``.
        """
        noise = self._pending_request_noise
        if noise is not None:
            # Sniff dtype/device off the call site (positional first,
            # then keyword fallback) so we match upstream's expected
            # tensor placement before handing the override over.
            dtype = args[4] if len(args) > 4 else kwargs.get("dtype")
            device = args[5] if len(args) > 5 else kwargs.get("device")
            if dtype is not None:
                noise = noise.to(dtype=dtype)
            if device is not None:
                noise = noise.to(device=device)
            if len(args) >= 8:
                # latents was passed positionally — overwrite the slot
                # rather than adding a keyword (which would double-bind).
                args = (*args[:7], noise, *args[8:])
            else:
                # Older / partial call — safe to use keyword.
                kwargs["latents"] = noise
            # Consume the slot so a CFG-driven second call falls back to
            # upstream's RNG / its own latents (we only want to control x_T,
            # not any subsequent latent re-draws).
            self._pending_request_noise = None
        return super().prepare_latents(*args, **kwargs)

    def forward(self, req: OmniDiffusionRequest, **kwargs) -> DiffusionOutput:
        # Read eta off the typed field (``OmniDiffusionSamplingParams.eta``).
        # ``_ensure_scheduler_for_eta`` installs our scheduler unconditionally
        # (eta=0 still installs but the SDE branch never fires; see the
        # method docstring for why).
        eta = float(getattr(req.sampling_params, "eta", 0.0) or 0.0)
        self._ensure_scheduler_for_eta(eta)

        # Install (or reset) the sparse-SDE step gate on the scheduler.
        # The trainer-side request builder (``rollout/engine/vllm_omni/request.py``)
        # writes the per-request sparse step list into
        # ``OmniDiffusionSamplingParams.extra_args["sde_indices"]``; the gate
        # MUST be re-installed (or cleared) every forward so a stale set from
        # a previous request can't leak into this one.
        if isinstance(self.scheduler, FlowMatchSDEDiscreteScheduler):
            extra = getattr(req.sampling_params, "extra_args", None) or {}
            sde_indices = extra.get("sde_indices")
            self.scheduler._sde_indices_set = (
                frozenset(int(i) for i in sde_indices) if sde_indices is not None else None
            )

        # Resolve and stash this request's pre-computed initial latent
        # (the override on ``prepare_latents`` consumes it). Re-evaluated
        # every forward so a stale slice from a previous request can't
        # leak in.
        self._resolve_pending_noise(req)

        # Reset text capture for this request; install the hook lazily.
        self._text_capture = None
        self._install_encode_prompt_hook()
        self._install_t5_truncation_fix()

        # Delegate the entire denoise pipeline (prompt encoding, latent
        # prep, timestep build, diffusion loop, VAE decode) to upstream.
        out = super().forward(req, **kwargs)

        # Drain trajectory off our scheduler. ``isinstance`` check is
        # belt-and-braces — ``_ensure_scheduler_for_eta`` always installs
        # us, but a future subclass override could in theory swap it out.
        if isinstance(self.scheduler, FlowMatchSDEDiscreteScheduler):
            traj = self.scheduler.drain_trajectory()
            if traj is not None:
                latents, sigmas, _timesteps, log_probs = traj
                out.trajectory_latents = latents
                # ``trajectory_timesteps`` carries the true [0, 1] sigma
                # schedule — see module docstring. The rollout response
                # layer reads it back as ``LatentSegment.sigmas``.
                out.trajectory_timesteps = sigmas
                out.trajectory_log_probs = log_probs
                # Stash the real sparse SDE step indices on custom_output so
                # the rollout response can echo them to the trainer as the
                # segment's ``sde_indices``. We can't use a plain runtime
                # attr — vllm-omni's IPC path filters those out (see the
                # text_capture comment below).
                sde_step_indices = self.scheduler.last_sde_step_indices
                if out.custom_output is None:
                    out.custom_output = {}
                out.custom_output["sde_step_indices"] = sde_step_indices

        # Surface the captured text embeds via ``DiffusionOutput.custom_output``
        # — a dataclass-declared dict that vllm-omni explicitly forwards into
        # ``OmniRequestOutput.custom_output`` (plain runtime attrs on
        # ``DiffusionOutput`` get filtered during IPC).
        if self._text_capture is not None:
            if out.custom_output is None:
                out.custom_output = {}
            out.custom_output["text_capture"] = self._text_capture
        return out


__all__ = ["RLStableDiffusion3Pipeline"]
