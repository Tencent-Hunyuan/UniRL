"""Make LTX-2's custom AV denoising/decoding stages RL-rollout-complete."""

from __future__ import annotations

_SENTINEL = "_unirl_ltx2_rollout_sde"


def patch_ltx2_rollout_sde() -> None:
    import sglang.multimodal_gen.runtime.models.dits.ltx_2 as rope_module
    import torch
    from sglang.multimodal_gen.runtime.pipelines.ltx_2_pipeline import (
        LTX2SigmaPreparationStage,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.decoding_av import (
        LTX2AVDecodingStage,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.denoising import (
        LTX2DenoisingStage,
    )

    from unirl.rollout.engine.sglang_diffusion._patches.patch_conditions import (
        _copy_conditions,
    )
    from unirl.sde.noise import make_denoise_step_generators

    orig_mask = LTX2DenoisingStage._get_ltx_prompt_attention_mask
    if not getattr(orig_mask, "_unirl_drop_all_valid_mask", False):

        def _get_ltx_prompt_attention_mask(batch, *, is_ltx23_variant: bool, negative: bool = False):
            mask = orig_mask(batch, is_ltx23_variant=is_ltx23_variant, negative=negative)
            if isinstance(mask, torch.Tensor) and mask.all():
                return None
            return mask

        _get_ltx_prompt_attention_mask._unirl_drop_all_valid_mask = True  # type: ignore[attr-defined]
        LTX2DenoisingStage._get_ltx_prompt_attention_mask = staticmethod(_get_ltx_prompt_attention_mask)

    if not getattr(rope_module.apply_split_rotary_emb, "_unirl_fp32_rope", False):

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
        rope_module.apply_interleaved_rotary_emb = apply_interleaved_rotary_emb
        rope_module.apply_split_rotary_emb = apply_split_rotary_emb

    orig_sigma = LTX2SigmaPreparationStage.forward
    if not getattr(orig_sigma, _SENTINEL, False):

        def forward_sigma(self, batch, server_args):
            pinned = batch.sigmas
            result = orig_sigma(self, batch, server_args)
            if batch.rollout and pinned:
                result.sigmas = pinned
            return result

        forward_sigma._unirl_ltx2_rollout_sde = True  # type: ignore[attr-defined]
        LTX2SigmaPreparationStage.forward = forward_sigma

    orig_before_loop = LTX2DenoisingStage._before_denoising_loop
    if not getattr(orig_before_loop, "_unirl_audio_traj_align", False):

        def _before_denoising_loop(self, ctx, batch, server_args):
            result = orig_before_loop(self, ctx, batch, server_args)
            if batch.return_trajectory_latents and ctx.audio_latents is not None and not ctx.trajectory_audio_latents:
                ctx.trajectory_audio_latents.append(ctx.audio_latents)
            return result

        _before_denoising_loop._unirl_audio_traj_align = True  # type: ignore[attr-defined]
        LTX2DenoisingStage._before_denoising_loop = _before_denoising_loop

    orig_decode_av = LTX2AVDecodingStage.forward
    if not getattr(orig_decode_av, _SENTINEL, False):

        def forward_decode_av(self, batch, server_args):
            out = orig_decode_av(self, batch, server_args)
            rtd = out.rollout_trajectory_data
            if rtd is None:
                rtd = batch.rollout_trajectory_data
                if rtd is not None:
                    out.rollout_trajectory_data = rtd
            audio_traj = batch.trajectory_audio_latents
            dit_traj = rtd.dit_trajectory if rtd is not None else None
            if dit_traj is not None and audio_traj is not None and getattr(dit_traj, "audio_latents", None) is None:
                dit_traj.audio_latents = audio_traj.detach().cpu()
            _copy_conditions(batch, out)
            return out

        forward_decode_av._unirl_ltx2_rollout_sde = True  # type: ignore[attr-defined]
        LTX2AVDecodingStage.forward = forward_decode_av

    orig_stage_step = LTX2DenoisingStage._run_denoising_step
    if not getattr(orig_stage_step, _SENTINEL, False):

        def _run_denoising_step(self, ctx, step, batch, server_args):
            original_generator = batch.generator
            denoise_seeds = batch.denoise_seeds
            if batch.rollout and denoise_seeds is not None:
                base_seed = batch.seed
                if base_seed is not None:
                    batch.generator = make_denoise_step_generators(
                        base_seed=base_seed,
                        step_index=step.step_index,
                        sample_ids=denoise_seeds,
                    )
            try:
                return orig_stage_step(self, ctx, step, batch, server_args)
            finally:
                batch.generator = original_generator

        _run_denoising_step._unirl_ltx2_rollout_sde = True  # type: ignore[attr-defined]
        LTX2DenoisingStage._run_denoising_step = _run_denoising_step
