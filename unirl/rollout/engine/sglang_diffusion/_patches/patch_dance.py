"""REPLACE ``SchedulerRLMixin.flow_sde_sampling`` to add the DanceGRPO objective.

Stock upstream sglang supports only ``sde``/``cps``/``ode``; UniRL's
primary objective for FLUX.2-Klein is ``dance`` (DanceGRPO). ``dance`` is
FlowGRPO's SDE transition with a **constant** ``std_dev_t = eta`` (vs ``sde``'s
sigma-dependent ``sqrt(sigma/(1-sigma)) * eta``); ``prev_sample_mean`` and the
log-prob reduction are otherwise identical to the ``sde`` branch.

This exactly matches UniRL's train-side authority
``unirl/sde/kernels.py:DanceSDEStrategy`` (``.step`` / ``.compute_log_prob``)
that ``logprob_source='replay'`` recomputes against -- keeping the rollout
transition and the train-side log-prob consistent (iter-0 importance ratios ~1).
Parity with that authority is verified by hand for now (no automated parity test yet).

This is the ONLY REPLACE patch (all infra patches are additive). It re-vendors
upstream ``flow_sde_sampling`` with one extra ``elif``, so it must be re-synced by
hand against the pinned upstream source on any sglang bump.
"""

from __future__ import annotations

import math

import torch

_LOG_SQRT_2PI = math.log(math.sqrt(2 * math.pi))


def patch_dance() -> None:
    """Add ``dance`` to the rollout sde-type whitelist and to ``flow_sde_sampling``."""
    import sglang.multimodal_gen.configs.post_training.rl_rollout as rl_rollout
    import sglang.multimodal_gen.runtime.post_training.scheduler_rl_mixin as srm

    # (1) Allow "dance" through request-path validation + CLI choices. Both
    # `RLRolloutArgs.validate` and `add_cli_args` read this module global at
    # call time, so reassigning the attribute is sufficient. Idempotent.
    if "dance" not in rl_rollout._VALID_ROLLOUT_SDE_TYPES:
        rl_rollout._VALID_ROLLOUT_SDE_TYPES = tuple(rl_rollout._VALID_ROLLOUT_SDE_TYPES) + ("dance",)

    # (2) REPLACE flow_sde_sampling with the dance-aware version. Idempotent.
    if getattr(srm.SchedulerRLMixin.flow_sde_sampling, "_unirl_dance", False):
        return
    srm.SchedulerRLMixin.flow_sde_sampling = _flow_sde_sampling_with_dance


def _flow_sde_sampling_with_dance(
    self,
    batch,
    model_output: "torch.FloatTensor",
    sample: "torch.FloatTensor",
    current_sigma: "torch.FloatTensor",
    next_sigma: "torch.FloatTensor",
    generator: "torch.Generator",
) -> "torch.Tensor":
    """Re-vendor of upstream ``SchedulerRLMixin.flow_sde_sampling`` + ``dance``.

    Only the ``elif effective_sde_type == "dance"`` branch is new; everything
    else is byte-for-byte upstream so sde/cps/ode behaviour is unchanged.
    """
    rollout_session_data = self._get_rollout_session_data(batch)
    sde_type = batch.rollout_sde_type
    noise_level = float(batch.rollout_noise_level)
    log_prob_no_const = batch.rollout_log_prob_no_const
    debug_mode = bool(getattr(batch, "rollout_debug_mode", False))

    if not log_prob_no_const and sde_type != "ode":
        assert noise_level > 0, "True log-probability computation requires a non-zero noise level."

    dt = next_sigma - current_sigma

    sde_step_indices = getattr(batch, "rollout_sde_step_indices", None)
    loop_step_index = getattr(batch, "_rollout_loop_step_index", None)
    if (
        sde_type != "ode"
        and sde_step_indices is not None
        and loop_step_index is not None
        and loop_step_index not in sde_step_indices
    ):
        effective_sde_type = "ode"
    else:
        effective_sde_type = sde_type

    if effective_sde_type == "sde":
        model_output = model_output.float()
        sample = sample.float()
        variance_noise = self._rollout_variance_noise(batch, model_output, generator)
        full_variance_noise = rollout_session_data.noise_buffer
        std_dev_t = (
            torch.sqrt(
                current_sigma
                / (
                    1
                    - torch.where(
                        torch.isclose(current_sigma, current_sigma.new_tensor(1.0)),
                        rollout_session_data.sigma_max,
                        current_sigma,
                    )
                )
            )
            * noise_level
        )
        noise_std_dev = std_dev_t * torch.sqrt(-1 * dt)
        prev_sample_mean = (
            sample * (1 + std_dev_t**2 / (2 * current_sigma) * dt)
            + model_output * (1 + std_dev_t**2 * (1 - current_sigma) / (2 * current_sigma)) * dt
        )

        weighted_variance_noise = variance_noise * noise_std_dev
        prev_sample = prev_sample_mean + weighted_variance_noise
        log_prob_no_const_val = -((full_variance_noise * noise_std_dev) ** 2)

    elif effective_sde_type == "cps":
        model_output = model_output.float()
        sample = sample.float()
        variance_noise = self._rollout_variance_noise(batch, model_output, generator)
        full_variance_noise = rollout_session_data.noise_buffer
        std_dev_t = next_sigma * math.sin(noise_level * math.pi / 2)
        noise_std_dev = std_dev_t
        pred_original_sample = sample - current_sigma * model_output
        noise_estimate = sample + model_output * (1 - current_sigma)
        prev_sample_mean = pred_original_sample * (1 - next_sigma) + noise_estimate * torch.sqrt(
            next_sigma**2 - std_dev_t**2
        )

        weighted_variance_noise = variance_noise * noise_std_dev
        prev_sample = prev_sample_mean + weighted_variance_noise
        log_prob_no_const_val = -((full_variance_noise * noise_std_dev) ** 2)

    elif effective_sde_type == "dance":
        # DanceGRPO: identical to the "sde" branch except std_dev_t is the
        # CONSTANT eta (not sigma-dependent). Mirrors UniRL's train-side
        # DanceSDEStrategy.step (unirl/sde/kernels.py), which replay uses.
        model_output = model_output.float()
        sample = sample.float()
        variance_noise = self._rollout_variance_noise(batch, model_output, generator)
        full_variance_noise = rollout_session_data.noise_buffer
        std_dev_t = current_sigma.new_tensor(noise_level)
        noise_std_dev = std_dev_t * torch.sqrt(-1 * dt)
        prev_sample_mean = (
            sample * (1 + std_dev_t**2 / (2 * current_sigma) * dt)
            + model_output * (1 + std_dev_t**2 * (1 - current_sigma) / (2 * current_sigma)) * dt
        )

        weighted_variance_noise = variance_noise * noise_std_dev
        prev_sample = prev_sample_mean + weighted_variance_noise
        log_prob_no_const_val = -((full_variance_noise * noise_std_dev) ** 2)

    elif effective_sde_type == "ode":
        prev_sample = sample + dt * model_output
        prev_sample_mean = prev_sample
        variance_noise = torch.zeros_like(model_output)
        noise_std_dev = torch.zeros((), device=model_output.device, dtype=model_output.dtype)
        log_prob_no_const_val = torch.zeros(
            rollout_session_data.latents_shape,
            device=model_output.device,
            dtype=torch.float32,
        )
        if sde_type == "ode":
            assert log_prob_no_const, (
                "p_ode is always 0, true log_prob is meaningless, set rollout_log_prob_no_const to True to enable log_prob computation"
            )

    else:
        raise ValueError(f"Unsupported sde_type: {sde_type}")

    reduce_dims = list(range(1, len(log_prob_no_const_val.shape)))
    local_elem_count = log_prob_no_const_val.new_full(
        (log_prob_no_const_val.shape[0],),
        float(math.prod(log_prob_no_const_val.shape[1:])),
    )

    if log_prob_no_const or effective_sde_type == "ode":
        log_prob_local_sum = log_prob_no_const_val.sum(dim=reduce_dims)
    else:
        log_prob_local_sum = (
            log_prob_no_const_val / (2 * (noise_std_dev**2)) - torch.log(noise_std_dev) - _LOG_SQRT_2PI
        ).sum(dim=list(range(1, len(log_prob_no_const_val.shape))))

    if debug_mode:
        self.append_local_rollout_debug_tensors(
            batch,
            variance_noise=variance_noise,
            prev_sample_mean=prev_sample_mean,
            noise_std_dev=noise_std_dev,
            model_output=model_output,
        )

    self.append_local_rollout_log_probs(batch, log_prob_local_sum, local_elem_count)

    return prev_sample


_flow_sde_sampling_with_dance._unirl_dance = True  # type: ignore[attr-defined]
