"""Make LTX-2's custom AV denoising/decoding stages RL-rollout-complete.

LTX-2 overrides the shared sglang stages (``LTX2SigmaPreparationStage``,
``LTX2DenoisingStage``, ``LTX2AVDecodingStage``) for its audiovisual pipeline, and
each override drops a piece of the rollout contract the base stages provide. This
patch restores the four missing pieces so the LTX-2 rollout matches what the
trainside replays (and the GRPO ratio stays ~1):

  (1) σ-schedule alignment — ``LTX2SigmaPreparationStage`` recomputes ``batch.sigmas``
      from its own schedule, clobbering the driver-pinned ``req.sigmas``; keep the
      pinned schedule in rollout (``_patch_sigma_alignment``).
  (2) AV-decode output carry — ``LTX2AVDecodingStage`` builds its ``OutputBatch``
      without the ``rollout_trajectory_data`` / text-embed conditions / co-denoised
      audio trajectory the base ``DecodingStage`` + ``patch_conditions`` provide;
      pass all three through (``_patch_av_decode_carry``).
  (3)+(4) SDE log-prob bridge — ``LTX2DenoisingStage._run_denoising_step`` advances
      the video latents without passing ``batch``/``generator``, so the RL flow-match
      scheduler silently takes the plain ODE branch (no log-prob capture, "Generator
      must be provided"); stash them on the scheduler and inject them into
      ``step()`` (``_patch_sde_logprob_bridge``).

All wraps are idempotent (sentinel-guarded) and import-safe (sglang imported
locally). Applied via the hijack ``_safe_apply`` list, after ``patch_ltx2_scheduler``
installs the RL-mixin video scheduler.
"""

from __future__ import annotations

_SENTINEL = "_unirl_ltx2_rollout_sde"


def patch_ltx2_rollout_sde() -> None:
    _patch_sigma_alignment()
    _patch_av_decode_carry()
    _patch_sde_logprob_bridge()


def _patch_sigma_alignment() -> None:
    """Keep the driver-pinned σ schedule in rollout.

    ``LTX2SigmaPreparationStage`` OVERWRITES ``batch.sigmas`` with its own
    ``build_official_ltx2_sigmas`` (a linear-then-shifted schedule), clobbering the
    driver-pinned ``req.sigmas`` the trainside uses. So sglang rolled out on
    [1.0, 0.9, 0.8, …] while the driver pinned [1.0, 0.958, 0.910, …] →
    ``verify_engine_used_sigmas`` fails and the curves diverge. In ROLLOUT, honor the
    already-set sigmas (mirrors the base ``TimestepPreparationStage``).
    """
    from sglang.multimodal_gen.runtime.pipelines.ltx_2_pipeline import (
        LTX2SigmaPreparationStage,
    )

    orig = LTX2SigmaPreparationStage.forward
    if getattr(orig, _SENTINEL, False):
        return

    def forward(self, batch, server_args):
        pinned = getattr(batch, "sigmas", None)
        if getattr(batch, "rollout", False) and pinned is not None and len(pinned) > 0:
            return batch  # keep the driver-pinned σ schedule; do not recompute.
        return orig(self, batch, server_args)

    forward._unirl_ltx2_rollout_sde = True  # type: ignore[attr-defined]
    LTX2SigmaPreparationStage.forward = forward


def _patch_av_decode_carry() -> None:
    """Carry the rollout trajectory, audio trajectory, and text conditions onto
    ``LTX2AVDecodingStage``'s OutputBatch (the base ``DecodingStage`` includes them).

    Three fields the unirl adapter reads off the result never reach it for LTX-2,
    because ``LTX2AVDecodingStage`` overrides ``forward`` and builds a bare OutputBatch:

    * ``rollout_trajectory_data`` — base ``DecodingStage`` sets it from ``batch``;
      ``RawResult.trajectory_latents`` reads ``…dit_trajectory.latents`` off it
      ("SGLang result missing trajectory_latents" without this).
    * text-embed conditions — ``patch_conditions`` wraps only the BASE
      ``DecodingStage.forward``; mirror that copy here ("missing prompt_embeds").
    * co-denoised AUDIO trajectory — LTX-2's video DiT cross-attends to an audio
      latent, and the trainside replay (``LTX2DiffusionStage.replay``) needs the SAME
      per-step audio (``segment.aux_latents``) or it raises "aux_latents (audio
      trajectory) missing" and the curve diverges. The denoising stage stacks it onto
      ``batch.trajectory_audio_latents`` ([B_out, T+1, …]); ride it inside
      ``dit_trajectory`` (set post-hoc — not a declared field) so the existing
      per-output concat/slice + RawResult plumbing carry it alongside the video.
    """
    from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding_av import (
        LTX2AVDecodingStage,
    )
    from unirl.rollout.engine.sglang_diffusion._patches.patch_conditions import (
        _copy_conditions,
    )

    orig = LTX2AVDecodingStage.forward
    if getattr(orig, _SENTINEL, False):
        return

    def forward(self, batch, server_args):
        out = orig(self, batch, server_args)
        rtd = getattr(out, "rollout_trajectory_data", None)
        if rtd is None:
            rtd = getattr(batch, "rollout_trajectory_data", None)
            if rtd is not None:
                out.rollout_trajectory_data = rtd
        audio_traj = getattr(batch, "trajectory_audio_latents", None)
        dit_traj = getattr(rtd, "dit_trajectory", None)
        if dit_traj is not None and audio_traj is not None and getattr(dit_traj, "audio_latents", None) is None:
            dit_traj.audio_latents = audio_traj.detach().cpu()
        _copy_conditions(batch, out)
        return out

    forward._unirl_ltx2_rollout_sde = True  # type: ignore[attr-defined]
    LTX2AVDecodingStage.forward = forward


def _patch_sde_logprob_bridge() -> None:
    """Feed ``batch`` + per-step SDE generators into the RL flow-match ``step()``.

    ``LTX2DenoisingStage._run_denoising_step`` advances the video latents with
    ``scheduler.step(..., return_dict=False)`` — WITHOUT ``batch``. The RL scheduler
    runs ``flow_sde_sampling`` (the per-step log-prob capture) only ``if batch is not
    None and batch.rollout``, so LTX-2's batch-less call silently takes the plain ODE
    branch → nothing is appended → ``collect_rollout_log_probs`` stacks an empty list.
    The base ``DenoisingStage`` passes ``batch``, so SD3 / WAN / HunyuanVideo capture
    fine; only LTX-2's override misses it. Bridge it without re-vendoring the large
    ``_run_denoising_step``: stash the rollout ``batch`` + SDE generators on the
    scheduler around the step, then inject them into ``step()`` when absent. Only the
    VIDEO scheduler is bridged — for T2V the audio scheduler keeps the plain step.
    ``patch_denoising`` builds the per-sample generators for the base path; LTX-2
    bypasses it, so reuse the same derivation (else ``generator=None``).
    """
    from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteScheduler,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.ltx_2_denoising import (
        LTX2DenoisingStage,
    )
    from unirl.rollout.engine.sglang_diffusion._patches.patch_denoising import (
        _make_step_generators,
        _resolve_base_seed,
    )

    orig_stage_step = LTX2DenoisingStage._run_denoising_step
    if not getattr(orig_stage_step, _SENTINEL, False):

        def _run_denoising_step(self, ctx, step, batch, server_args):
            sched = getattr(ctx, "scheduler", None)
            gen = None
            denoise_seeds = getattr(batch, "denoise_seeds", None)
            if getattr(batch, "rollout", False) and denoise_seeds is not None:
                base_seed = _resolve_base_seed(batch)
                if base_seed is not None:
                    gen = _make_step_generators(
                        base_seed, int(step.step_index), ctx.latents.device, list(denoise_seeds)
                    )
            if sched is not None:
                sched._unirl_rollout_batch = batch
                sched._unirl_rollout_generator = gen
            # Trajectory capture is handled by the BASE DenoisingStage.forward loop
            # (LTX2AVDenoisingStage.forward delegates to super().forward()), gated on
            # return_trajectory_latents. Do NOT append it here too — double-recording
            # corrupts the stacked tensor.
            try:
                return orig_stage_step(self, ctx, step, batch, server_args)
            finally:
                if sched is not None:
                    sched._unirl_rollout_batch = None
                    sched._unirl_rollout_generator = None

        _run_denoising_step._unirl_ltx2_rollout_sde = True  # type: ignore[attr-defined]
        LTX2DenoisingStage._run_denoising_step = _run_denoising_step

    orig_step = FlowMatchEulerDiscreteScheduler.step
    if not getattr(orig_step, _SENTINEL, False):

        def step(self, *args, **kwargs):
            if kwargs.get("batch") is None:
                stashed = getattr(self, "_unirl_rollout_batch", None)
                if stashed is not None:
                    kwargs["batch"] = stashed
                    gen = getattr(self, "_unirl_rollout_generator", None)
                    if gen is not None and kwargs.get("generator") is None:
                        kwargs["generator"] = gen
            return orig_step(self, *args, **kwargs)

        step._unirl_ltx2_rollout_sde = True  # type: ignore[attr-defined]
        FlowMatchEulerDiscreteScheduler.step = step
