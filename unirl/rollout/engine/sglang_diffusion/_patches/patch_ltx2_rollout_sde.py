"""Make LTX-2's custom AV denoising/decoding stages RL-rollout-complete."""

from __future__ import annotations

_SENTINEL = "_unirl_ltx2_rollout_sde"


def patch_ltx2_rollout_sde() -> None:
    _patch_all_valid_prompt_mask()
    _patch_rope_precision_alignment()
    _patch_scheduler()
    _patch_sigma_alignment()
    _patch_audio_trajectory_alignment()
    _patch_av_decode_carry()
    _patch_sde_logprob_bridge()


def _patch_all_valid_prompt_mask() -> None:
    """Turn LTX's post-connector all-valid prompt mask into ``None``."""
    import torch
    from sglang.multimodal_gen.runtime.pipelines_core.stages.ltx_2_denoising import (
        LTX2DenoisingStage,
    )

    orig = LTX2DenoisingStage._get_ltx_prompt_attention_mask
    if getattr(orig, "_unirl_drop_all_valid_mask", False):
        return

    def _get_ltx_prompt_attention_mask(batch, *, is_ltx23_variant: bool, negative: bool = False):
        mask = orig(batch, is_ltx23_variant=is_ltx23_variant, negative=negative)
        if isinstance(mask, torch.Tensor) and bool(mask.all()):
            return None
        return mask

    _get_ltx_prompt_attention_mask._unirl_drop_all_valid_mask = True  # type: ignore[attr-defined]
    LTX2DenoisingStage._get_ltx_prompt_attention_mask = staticmethod(_get_ltx_prompt_attention_mask)


def _patch_rope_precision_alignment() -> None:
    """Use diffusers' fp32 LTX-2 rotary multiply in the SGLang DiT."""
    import sglang.multimodal_gen.runtime.models.dits.ltx_2 as module
    import torch

    if getattr(module.apply_split_rotary_emb, "_unirl_fp32_rope", False):
        return

    def apply_interleaved_rotary_emb(x, freqs):
        cos, sin = freqs
        x_real, x_imag = x.unflatten(2, (-1, 2)).unbind(-1)
        x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(2)
        return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

    def apply_split_rotary_emb(x, freqs):
        cos, sin = freqs
        x_dtype = x.dtype
        needs_reshape = False
        if x.ndim != 4 and cos.ndim == 4:
            batch, heads, tokens, _ = cos.shape
            x = x.reshape(batch, tokens, heads, -1).swapaxes(1, 2)
            needs_reshape = True

        last = x.shape[-1]
        if last % 2:
            raise ValueError(f"Expected an even rotary dim, got {last}")
        half = last // 2
        split_x = x.reshape(*x.shape[:-1], 2, half).float()
        first_x = split_x[..., :1, :]
        second_x = split_x[..., 1:, :]
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
        out = split_x * cos
        first_out = out[..., :1, :]
        second_out = out[..., 1:, :]
        first_out.addcmul_(-sin, second_x)
        second_out.addcmul_(sin, first_x)
        out = out.reshape(*out.shape[:-2], last)
        if needs_reshape:
            out = out.swapaxes(1, 2).reshape(batch, tokens, -1)
        return out.to(dtype=x_dtype)

    apply_interleaved_rotary_emb._unirl_fp32_rope = True  # type: ignore[attr-defined]
    apply_split_rotary_emb._unirl_fp32_rope = True  # type: ignore[attr-defined]
    module.apply_interleaved_rotary_emb = apply_interleaved_rotary_emb
    module.apply_split_rotary_emb = apply_split_rotary_emb


def _patch_scheduler() -> None:
    """Replace LTX-2's stock scheduler with SGLang's rollout-aware variant."""
    from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteScheduler,
    )
    from sglang.multimodal_gen.runtime.pipelines.ltx_2_pipeline import _BaseLTX2Pipeline

    orig = _BaseLTX2Pipeline.initialize_pipeline
    if getattr(orig, "_unirl_flowmatch_scheduler", False):
        return

    def initialize_pipeline(self, server_args) -> None:
        orig(self, server_args)
        scheduler = self.get_module("scheduler")
        self.modules["scheduler"] = FlowMatchEulerDiscreteScheduler.from_config(scheduler.config)

    initialize_pipeline._unirl_flowmatch_scheduler = True  # type: ignore[attr-defined]
    _BaseLTX2Pipeline.initialize_pipeline = initialize_pipeline


def _patch_sigma_alignment() -> None:
    """Keep the driver-pinned σ schedule in rollout."""
    from sglang.multimodal_gen.runtime.pipelines.ltx_2_pipeline import (
        LTX2SigmaPreparationStage,
    )

    orig = LTX2SigmaPreparationStage.forward
    if getattr(orig, _SENTINEL, False):
        return

    def forward(self, batch, server_args):
        pinned = getattr(batch, "sigmas", None)
        if getattr(batch, "rollout", False) and pinned is not None and len(pinned) > 0:
            result = orig(self, batch, server_args)
            result.sigmas = pinned
            return result
        return orig(self, batch, server_args)

    forward._unirl_ltx2_rollout_sde = True  # type: ignore[attr-defined]
    LTX2SigmaPreparationStage.forward = forward


def _patch_av_decode_carry() -> None:
    """Carry the rollout trajectory, audio trajectory, and text conditions onto the LTX-2 AV OutputBatch."""
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


def _patch_audio_trajectory_alignment() -> None:
    """Prepend audio x_T so auxiliary trajectory indices match video T+1."""
    from sglang.multimodal_gen.runtime.pipelines_core.stages.ltx_2_denoising import (
        LTX2DenoisingStage,
    )

    orig = LTX2DenoisingStage._before_denoising_loop
    if getattr(orig, "_unirl_audio_traj_align", False):
        return

    def _before_denoising_loop(self, ctx, batch, server_args):
        result = orig(self, ctx, batch, server_args)
        if (
            getattr(batch, "return_trajectory_latents", False)
            and getattr(ctx, "audio_latents", None) is not None
            and not getattr(ctx, "trajectory_audio_latents", None)
        ):
            ctx.trajectory_audio_latents.append(ctx.audio_latents)
        return result

    _before_denoising_loop._unirl_audio_traj_align = True  # type: ignore[attr-defined]
    LTX2DenoisingStage._before_denoising_loop = _before_denoising_loop


def _patch_sde_logprob_bridge() -> None:
    """Feed ``batch`` + per-step SDE generators into the RL flow-match ``step()``."""
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
