"""Per-sample SDE noise via ``denoise_seeds`` + per-request fallback (LIN-365).

Stock upstream builds one ``extra_step_kwargs["generator"] = batch.generator``
once per request (denoising.py) and reuses it across steps. The fork instead
rebuilds **per-sample, per-step** CPU generators with a blake2b key over
``(seed, step_index, denoise_seeds[i])`` -- the SAME derivation as UniRL's
``make_step_generators`` -- so the engine's SDE noise is reproducible /
driver-aligned while staying INDEPENDENT per sample. ``denoise_seeds`` is keyed
per-sample-unique (``sample_ids``) so each sample explores its own per-step SDE
noise; keying it on group ids (the fork's choice) would make same-group samples
share per-step noise. (Per-sample x_T already supplies within-group diversity, so
this is a secondary correctness/exploration knob, not the flat-reward root cause
-- that was the grouped-forward trajectory collapse fixed in
``patch_rollout_trajectory``.) Upstream has no consumer for ``denoise_seeds``
(``patch_sampling_io`` only copies the field onto the Req), so without this patch
every sample draws independent noise.

This AROUND-wraps ``DenoisingStage._run_denoising_step`` and, for rollout
requests that carry ``denoise_seeds`` + a seed, overrides
``ctx.extra_step_kwargs["generator"]`` with the per-step generator list right
before ``ctx.scheduler.step(**extra_step_kwargs)`` forwards it to
``flow_sde_sampling`` (whose ``_rollout_variance_noise`` indexes ``generator[i]``
per sample). ODE (non-SDE) steps ignore the generator, so injecting on every
step is safe and matches the fork's per-step re-seed for the SDE steps.

NOTE: model-specific stages may override ``_run_denoising_step``; this patches
the shared base ``DenoisingStage``, which SD3's image path uses. A subclass that
overrides the method would need its own wrap -- flagged for taiji verification.

**Per-request fallback** (``_patch_rollout_variance_noise_device``): when the
AROUND-wrap above is a no-op because a model-specific ``DenoisingStage``
subclass overrode ``_run_denoising_step`` (e.g. Qwen-Image-Edit-Plus), upstream's
default ``extra_step_kwargs["generator"] = batch.generator`` reaches
``_rollout_variance_noise``. That generator is seeded from the shared
``batch.seed`` — the SAME value for every GRPO-group sample (each is a separate
B=1 request reseeded to the same ``sampling_params.seed``) — so all samples draw
byte-identical per-step z_t → frozen exploration noise → breaks GRPO sample
independence → reward regresses after ~100 rollouts. This is the same root cause
as the vLLM-Omni BAGEL fix (PR #89, heguangxin's comment): ``pipeline_bagel``
reseeds the global RNG per request and the SDE scheduler drew z_t from it. Fix
(mirrors ``BagelFlowSDEScheduler.step``): when ``_rollout_variance_noise``
receives a single ``torch.Generator`` (not a denoise_seeds list), replace it
with a per-request generator seeded from ``os.urandom``, independent of the
reseeded batch/global RNG. Stashed on ``batch`` so the generator persists across
SDE steps within one request. The denoise_seeds path (list of generators) is
unaffected — it fires for models whose denoising stage inherits the base
``_run_denoising_step`` (e.g. SD3), preserving its deterministic driver-aligned
per-sample noise.
"""

from __future__ import annotations

import hashlib
import os

import torch

_MAX_TORCH_SEED = (1 << 63) - 1


def _make_step_generators(
    base_seed: int,
    step_index: int,
    device: torch.device,
    denoise_seeds: list[str],
) -> list[torch.Generator]:
    """Per-sample deterministic CPU generators for one SDE step.

    Copied from the fork's ``denoising.py``. CPU generators give cross-engine
    determinism (identical CPU random sequence regardless of GPU). ``device`` is
    accepted for signature parity but unused (generators are always CPU).
    """
    del device
    generators: list[torch.Generator] = []
    for seed_key in denoise_seeds:
        payload = (f"{int(base_seed)}::step::{int(step_index)}::sample::{str(seed_key)}").encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        seed = int.from_bytes(digest, byteorder="big", signed=False) % _MAX_TORCH_SEED
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        generators.append(g)
    return generators


def _resolve_base_seed(batch) -> int | None:
    seed = getattr(batch, "seed", None)
    if seed is None:
        seed = getattr(getattr(batch, "sampling_params", None), "seed", None)
    return int(seed) if seed is not None else None


def patch_denoising() -> None:
    from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
        DenoisingStage,
    )

    orig = DenoisingStage._run_denoising_step
    if getattr(orig, "_unirl_denoise_seeds", False):
        return

    def _run_denoising_step(self, ctx, step, batch, server_args):
        denoise_seeds = getattr(batch, "denoise_seeds", None)
        if getattr(batch, "rollout", False) and denoise_seeds is not None:
            base_seed = _resolve_base_seed(batch)
            if base_seed is not None:
                ctx.extra_step_kwargs["generator"] = _make_step_generators(
                    base_seed,
                    int(step.step_index),
                    ctx.latents.device,
                    list(denoise_seeds),
                )
        return orig(self, ctx, step, batch, server_args)

    _run_denoising_step._unirl_denoise_seeds = True  # type: ignore[attr-defined]
    DenoisingStage._run_denoising_step = _run_denoising_step

    _patch_rollout_variance_noise_device()


def _patch_rollout_variance_noise_device() -> None:
    """Make ``SchedulerRLMixin._rollout_variance_noise`` tolerate CPU generators.

    ``_make_step_generators`` builds CPU generators (cross-engine determinism --
    mirrors UniRL's ``make_step_generators``). Upstream draws via
    ``torch.randn(out=cuda_buffer, generator=gen)``, which requires
    ``gen.device == buffer.device`` -> ``Expected a 'cuda' device type for
    generator but found 'cpu'``. Draw on the generator's device then copy to the
    buffer (mirrors diffusers ``randn_tensor``). REPLACE-patched (the buggy draw is
    mid-method); the rest is byte-for-byte upstream.

    Also installs a per-request ``os.urandom``-seeded generator fallback for
    when the ``denoise_seeds`` AROUND-wrap on
    ``DenoisingStage._run_denoising_step`` is a no-op (model-specific subclass
    overrode the method). See the module docstring for the root-cause analysis.
    """
    from sglang.multimodal_gen.runtime.post_training.scheduler_rl_mixin import (
        SchedulerRLMixin,
    )

    if getattr(SchedulerRLMixin._rollout_variance_noise, "_unirl_dev", False):
        return

    def _rollout_variance_noise(self, batch, model_output, generator):
        assert generator is not None, "Generator must be provided"
        rsd = self._get_rollout_session_data(batch)
        device = model_output.device
        dtype = model_output.dtype
        local_shape = tuple(model_output.shape)
        B = local_shape[0]
        if isinstance(generator, torch.Generator):
            # Fallback: the denoise_seeds AROUND-wrap on
            # ``DenoisingStage._run_denoising_step`` did not fire (a
            # model-specific stage subclass overrode the method), so
            # upstream's default ``extra_step_kwargs["generator"] =
            # batch.generator`` reached us. That generator is seeded from
            # the shared ``batch.seed`` — the SAME value for every
            # GRPO-group sample (each is a separate B=1 request reseeded
            # to the same ``sampling_params.seed``) — so with this shared
            # generator all samples draw byte-identical per-step z_t ->
            # frozen exploration noise -> breaks GRPO sample independence
            # -> reward regresses after ~100 rollouts. Same root cause as
            # the vLLM-Omni BAGEL fix (PR #89, heguangxin's comment):
            # ``pipeline_bagel`` reseeds the global RNG per request and the
            # SDE scheduler drew z_t from it. Fix (mirrors
            # ``BagelFlowSDEScheduler.step`` in
            # ``vllm_omni/pipelines/bagel/bagel_flow_match_sde_scheduler.py``):
            # replace the shared generator with a per-request generator
            # seeded from ``os.urandom``, independent of the reseeded
            # batch/global RNG. Stash on ``batch`` so the generator
            # persists across SDE steps within one request (each
            # per-output forward has its own batch -> generator is
            # naturally per-sample). The denoise_seeds path (list of
            # generators) is unaffected: it fires for models whose
            # denoising stage inherits the base ``_run_denoising_step``
            # (e.g. SD3), preserving its deterministic driver-aligned
            # per-sample noise.
            assert B == 1, "Generator must be a list if batch size is not 1"
            gen = getattr(batch, "_unirl_noise_gen", None)
            if gen is None:
                gen = torch.Generator(device=device)
                gen.manual_seed(int.from_bytes(os.urandom(8), "big"))
                try:
                    batch._unirl_noise_gen = gen  # type: ignore[attr-defined]
                except AttributeError:
                    pass  # immutable batch — generator is still valid for this step
            generator = [gen]
        else:
            assert len(generator) == B, "Generator list must have the same length as batch size"
        buffer = self._get_or_create_rollout_noise_buffer(rsd, rsd.latents_shape, device, dtype)
        for i in range(B):
            g = generator[i]
            if g is not None and getattr(g, "device", None) is not None and g.device.type != buffer.device.type:
                tmp = torch.randn(rsd.latents_shape, generator=g, dtype=dtype, device=g.device)
                buffer[i : i + 1].copy_(tmp)
            else:
                torch.randn(rsd.latents_shape, out=buffer[i : i + 1], generator=g)
        sharded_noise, _ = rsd.pipeline_config.shard_latents_for_sp(batch=batch, latents=buffer)
        if tuple(sharded_noise.shape) != local_shape:
            raise ValueError(
                "Rollout SP noise shape mismatch after shard. "
                f"Expected local_shape={local_shape}, got {tuple(sharded_noise.shape)}."
            )
        return sharded_noise

    _rollout_variance_noise._unirl_dev = True  # type: ignore[attr-defined]
    SchedulerRLMixin._rollout_variance_noise = _rollout_variance_noise
