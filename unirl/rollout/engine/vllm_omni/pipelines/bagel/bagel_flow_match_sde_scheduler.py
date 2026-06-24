"""Trajectory-capturing SDE flow-match scheduler for BAGEL's ``generate_image``.

BAGEL is the odd one out among the vLLM-Omni RL pipelines. SD3 / Qwen-Image run
diffusers ``FlowMatchEulerDiscreteScheduler`` loops and call
``scheduler.step(noise_pred, t, latents, return_dict=False)[0]`` — so their RL
subclass swaps in :class:`...flow_match_sde_scheduler.FlowMatchSDEDiscreteScheduler`
(a diffusers ``SchedulerMixin`` subclass with the ``set_timesteps(sigmas=...)`` /
``[0]`` conventions). BAGEL's vendored ``Bagel.generate_image``
(``vllm_omni/diffusion/models/bagel/bagel_transformer.py``) is a hand-rolled loop
with a DIFFERENT scheduler contract::

    out = scheduler.step(v_t, timesteps[i], x_t, dts[i], **scheduler_kwargs)
    x_t = out.prev_sample
    if out.log_prob is not None: trajectory_log_probs.append(out.log_prob)

i.e. the step takes ``(model_output, timestep, sample, dt)`` POSITIONALLY (no σ
schedule managed inside the scheduler, no diffusers ``step_index`` lifecycle, dt
handed in per step) and returns an object with ``.prev_sample`` + ``.log_prob``.
The diffusers ``FlowMatchSDEDiscreteScheduler`` cannot satisfy that signature, so
BAGEL gets its own scheduler — this file.

The SDE math is byte-identical to the trainside path
(:class:`unirl.sde.kernels.FlowSDEStrategy`, ``unirl/sde/kernels.py:253``) so the
GRPO on-policy invariant holds: under identical weights, replay's ``new_logp``
matches the rollout's emitted ``old_logp`` and the PPO ratio ``exp(new-old) ≈ 1``.
Concretely the per-step transition is::

    dt               = sigma_next - sigma                      (< 0, decreasing schedule)
    std_dev_t        = sqrt(sigma / (1 - clamp(sigma))) * eta   (clamp: σ==1 → sigma_max)
    prev_sample_mean = sample   * (1 + std_dev_t² / (2σ) · dt)
                     + v_t      * (1 + std_dev_t² (1-σ) / (2σ) · dt)
    prev_sample      = prev_sample_mean + std_dev_t · sqrt(-dt) · randn         (SDE step)
                     = prev_sample_mean                                         (ODE step, eta gate off)
    std_var          = std_dev_t · sqrt(-dt)
    log_prob         = mean[ -(prev_sample.detach() - prev_sample_mean)² / (2 std_var²)
                             - log(std_var) - ½ log(2π) ]                       (over all elems)

The dtype round-trip on ``prev_sample`` before the log-prob (cast to the stored
trajectory dtype, then back to fp32) mirrors ``FlowSDEStrategy._finalize_logp`` —
the rollout records ``logp(stored x_{t+1} | μ)`` so replay (reading the SAME stored
latent) lands on the same density.

Per-step SDE vs ODE is gated on ``_sde_indices`` (the sparse step set the driver
resolved, shared across the GRPO group). Latents/timesteps are captured DENSELY
(every step) so trainer-side replay has ``x_t`` at every storage slot; log-probs
are captured SPARSELY (only the SDE steps). The capturing mirrors
``BagelDiffusionStage.diffuse`` (``unirl/models/bagel/diffusion.py``), which stores
the step boundaries + final clean latent.

This scheduler is NOT a diffusers ``SchedulerMixin`` — it is a plain object whose
only consumer is BAGEL's ``generate_image``. ``set_for_request`` (re-armed every
``forward``) sets eta / sde_indices / the full σ schedule; ``drain_trajectory``
exports the captured triple after the loop.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from diffusers.utils.torch_utils import randn_tensor


@dataclass
class BagelSDEStepOutput:
    """Return payload for :meth:`BagelFlowSDEScheduler.step`.

    ``generate_image`` reads ``.prev_sample`` (always) and ``.log_prob``
    (appended to ``trajectory_log_probs`` only when not ``None``).
    """

    prev_sample: torch.Tensor
    log_prob: Optional[torch.Tensor]
    prev_sample_mean: torch.Tensor


class BagelFlowSDEScheduler:
    """SDE flow-match step for BAGEL ``generate_image`` with trajectory capture.

    One instance lives for the pipeline's lifetime (a worker singleton);
    :meth:`set_for_request` re-arms it per request and clears the capture
    buffers. ``step`` matches the positional ``(model_output, timestep, sample,
    dt)`` contract BAGEL's loop uses and records the dense latent trajectory +
    sparse SDE log-probs.
    """

    def __init__(self, *, eta: float = 1.0, sigma_max: float = 0.99) -> None:
        # ``eta == 0`` is allowed (every step degenerates to Euler ODE; the SDE
        # branch is gated on ``_sde_indices``, not eta). Negative eta is nonsense.
        if eta < 0.0:
            raise ValueError(f"BagelFlowSDEScheduler.eta must be >= 0; got {eta!r}.")
        self._eta = float(eta)
        self._sigma_max = float(sigma_max)
        # Per-request state (set by ``set_for_request``).
        self._sde_indices_set: Optional[frozenset] = None
        self._trajectory_dtype: torch.dtype = torch.float32
        self._step_index: int = 0
        # generator is required instead of the reseeded global RNG.
        self._noise_generator: Optional[torch.Generator] = None
        # Capture buffers (cleared each request).
        self._traj_latents: List[torch.Tensor] = []
        self._traj_timesteps: List[torch.Tensor] = []
        self._traj_log_probs: List[torch.Tensor] = []
        self._traj_sde_step_indices: List[int] = []
        self._initial_latent: Optional[torch.Tensor] = None
        self._initial_timestep: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ #
    # Per-request arming
    # ------------------------------------------------------------------ #

    def set_for_request(
        self,
        *,
        eta: float,
        sde_indices: Optional[List[int]],
        sigma_max: Optional[float] = None,
        trajectory_dtype: torch.dtype = torch.float32,
    ) -> None:
        """Arm this request: SDE strength, sparse step gate, σ_max, trajectory dtype.

        MUST fire on every pipeline ``forward`` — the scheduler is a long-lived
        worker singleton and stale state would silently mis-gate SDE steps.
        ``sde_indices=None`` disarms the SDE gate (pure Euler ODE; dense latent
        capture still runs — the trainer's clean-latents replay needs a
        non-empty trajectory regardless).

        ``sigma_max`` is the value that replaces σ==1 in the ``std_dev_t``
        denominator ``sqrt(σ/(1-σ))`` on the first step (σ_0 == 1.0 would divide
        by zero). It MUST equal the trainside ``BagelDiffusionStage``'s choice —
        ``schedule[1]`` (the second σ point) — or the first SDE step's std_dev_t /
        log-prob diverges and the GRPO ratio drifts off 1 (a hardcoded 0.99 gave
        ratio ≈ 0.8). The adapter passes ``req.sigmas[1]``; ``None`` keeps the
        prior instance value (smoke tests without an engine-pinned schedule).

        The σ schedule itself is NOT passed: BAGEL's ``generate_image`` builds it
        internally from ``(num_timesteps, timestep_shift)``, and this scheduler
        reconstructs the echo from the σ the loop visits (see
        :meth:`drain_trajectory`). The adapter is responsible for sending
        ``num_inference_steps = trainside_steps + 1`` and the matching shift so
        BAGEL's internal schedule equals the engine-pinned ``req.sigmas``.
        """
        if eta < 0.0:
            raise ValueError(f"BagelFlowSDEScheduler.set_for_request: eta must be >= 0; got {eta!r}.")
        self._eta = float(eta)
        self._sde_indices_set = frozenset(int(i) for i in sde_indices) if sde_indices is not None else None
        if sigma_max is not None:
            self._sigma_max = float(sigma_max)
        self._trajectory_dtype = trajectory_dtype
        self._step_index = 0
        self._noise_generator = None
        self._traj_latents = []
        self._traj_timesteps = []
        self._traj_log_probs = []
        self._traj_sde_step_indices = []
        self._initial_latent = None
        self._initial_timestep = None

    # ------------------------------------------------------------------ #
    # Per-step transition (BAGEL generate_image contract)
    # ------------------------------------------------------------------ #

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        dt: torch.Tensor,
        **_unused,
    ) -> BagelSDEStepOutput:
        """One SDE flow-match transition. Positional contract == BAGEL's loop.

        ``model_output`` is the CFG-combined velocity ``v_t``; ``timestep`` is the
        current σ (BAGEL's ``timesteps[i]``, a [0,1] flow-match σ, NOT a 1000-scale
        index); ``sample`` is ``x_t``; ``dt`` is BAGEL's ``dts[i] = timesteps[i] -
        timesteps[i+1]`` — a POSITIVE step size (σ decreases), the NEGATION of the
        trainside convention ``dt = sigma_next - sigma`` used by
        :class:`unirl.sde.kernels.FlowSDEStrategy`.

        We flip the sign once here so all downstream math matches the trainside
        kernel exactly (its ``dt`` is negative; ``sqrt(-dt)`` is the real SDE noise
        scale). ``sigma_next = sigma + dt`` then recovers ``timesteps[i+1]``.
        Sanity: the ODE branch's ``sample + v_t·dt`` equals BAGEL's
        no-scheduler ``x_t - v_t·dts[i]`` (since ``dt == -dts[i]``).
        """
        sigma = timestep if torch.is_tensor(timestep) else torch.as_tensor(float(timestep))
        sigma = sigma.to(device=sample.device, dtype=torch.float32).reshape(())
        dt_passed = dt if torch.is_tensor(dt) else torch.as_tensor(float(dt))
        dt_passed = dt_passed.to(device=sample.device, dtype=torch.float32).reshape(())
        # BAGEL hands a POSITIVE step (σ_i - σ_{i+1}); the trainside kernel wants
        # ``dt = σ_next - σ`` (negative). Flip once; everything below is trainside-form.
        dt_t = -dt_passed
        sigma_next = sigma + dt_t  # == timesteps[i+1]; captured as the post-step σ

        step_idx = self._step_index

        # Position-0 capture: stash the input x_T + its σ (== full[0]) on the
        # first step so ``drain_trajectory`` can return a dense [T+1] latent
        # trajectory and reconstruct the full [T+1] σ schedule. BAGEL builds its
        # σ schedule internally (no ``set_timesteps(sigmas=)`` hook), so the
        # genuine "did the worker use the trainer's schedule?" check is to
        # reconstruct it from the σ the loop actually visited: full[0] = the
        # first step's ``sigma``; full[1..T] = each step's ``sigma_next`` (the σ
        # AFTER the step, matching the post-step latent at that position).
        if self._initial_latent is None:
            self._initial_latent = sample.detach().to(self._trajectory_dtype).clone()
            self._initial_timestep = sigma.detach().clone()

        original_dtype = sample.dtype
        sample_f32 = sample.to(torch.float32)
        v_t_f32 = model_output.to(torch.float32)

        # SDE vs ODE: gated purely on ``_sde_indices_set`` (mirrors trainside
        # ``step_eta = eta if i in sde_set else 0.0``).
        if self._sde_indices_set is None or len(self._sde_indices_set) == 0:
            step_is_sde = False
        else:
            step_is_sde = int(step_idx) in self._sde_indices_set

        if step_is_sde:
            if float(self._eta) <= 0.0:
                raise RuntimeError(
                    f"BagelFlowSDEScheduler.step: step_index={int(step_idx)} is in the SDE "
                    f"gate but eta={self._eta!r}; eta must be > 0 for SDE steps. Check the "
                    f"adapter's eta / sde_indices wiring."
                )
            # std_dev_t clamps the σ==1 denominator to sigma_max (last high-σ
            # step on a flow-matching schedule); identical to FlowSDEStrategy.
            clamp_sigma = torch.where(sigma == 1, torch.as_tensor(self._sigma_max, device=sigma.device), sigma)
            std_dev_t = torch.sqrt(sigma / (1 - clamp_sigma)) * self._eta
            # EXACTLY FlowSDEStrategy.step (unirl/sde/kernels.py): note the
            # asymmetry — the ``sample`` term has ``· dt`` INSIDE the (1 + …);
            # the ``v_t`` term has ``· dt`` OUTSIDE, scaling the whole product.
            prev_sample_mean = (
                sample_f32 * (1 + std_dev_t**2 / (2 * sigma) * dt_t)
                + v_t_f32 * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt_t
            )
            # Draw z_t from a PER-REQUEST generator, NOT the global RNG
            # (``generator=None``). ``pipeline_bagel`` reseeds the global torch
            # RNG to ``sampling_params.seed`` at the start of EVERY request
            # (pipeline_bagel.py: torch.manual_seed(seed)), and each GRPO-group
            # sample is a separate bs=1 request reseeded to the SAME value. With
            # the global RNG that made the per-step z_t identical across the whole
            # group → frozen exploration noise → broke GRPO's sample independence
            # → grad-norm spikes → reward regressed after ~100 rollouts. A
            # dedicated generator seeded from ``os.urandom`` is independent of the
            # reseeded global RNG, so every group sample gets its own z_t (the
            # trainside path never reseeds per request, so this restores parity).
            #
            # Correct for cfg_parallel_size=1 (the current recipe): CFG is
            # combined on a single rank, so z_t is drawn once per step here.
            # cfg_parallel_size>1 would need care — with CFG branches on separate
            # ranks, os.urandom gives each rank a different z_t and the branches
            # diverge; there z_t must be drawn deterministically (seed by
            # (base_seed, group_id, step) on every rank, or draw on rank 0 and
            # broadcast across cfg_group like x_t).
            if self._noise_generator is None:
                self._noise_generator = torch.Generator(device=v_t_f32.device)
                self._noise_generator.manual_seed(int.from_bytes(os.urandom(8), "big"))
            noise = randn_tensor(
                v_t_f32.shape,
                generator=self._noise_generator,
                device=v_t_f32.device,
                dtype=torch.float32,
            )
            std_var = std_dev_t * torch.sqrt(-dt_t)
            prev_sample = prev_sample_mean + std_var * noise

            # Storage dtype round-trip BEFORE the log-prob (mirrors
            # FlowSDEStrategy._finalize_logp): the rollout records the density of
            # the stored (possibly bf16-rounded) sample so replay — reading the
            # same stored latent — matches.
            prev_sample = prev_sample.to(self._trajectory_dtype)
            prev_for_logp = prev_sample.to(torch.float32)
            log_prob_elem = (
                -((prev_for_logp.detach() - prev_sample_mean) ** 2) / (2 * std_var**2)
                - torch.log(std_var)
                - 0.5 * math.log(2 * math.pi)
            )
            log_prob: Optional[torch.Tensor] = log_prob_elem.mean()
        else:
            # Pure Euler ODE: x_{t+1} = x_t + v·dt. CRITICAL: no SDE drift
            # correction even when eta>0 — trainside replay drives non-SDE steps
            # with eta=0 (plain Euler), so an SDE-form mean here would push the
            # rollout off the replay manifold and bias every SDE-step log-prob.
            prev_sample_mean = sample_f32 + v_t_f32 * dt_t
            prev_sample = prev_sample_mean.to(self._trajectory_dtype)
            log_prob = None

        # Dense latent/timestep capture; sparse log-prob capture. The captured
        # timestep is ``sigma_next`` (the σ AFTER this step) so the stacked
        # post-step σ reconstruct full[1..T] — together with the position-0
        # ``sigma`` that is full[0], this rebuilds the exact [T+1] schedule the
        # loop ran on for the response-side σ verify.
        self._traj_latents.append(prev_sample.detach().clone())
        self._traj_timesteps.append(sigma_next.detach().clone())
        if log_prob is not None:
            self._traj_log_probs.append(log_prob.detach().clone())
            self._traj_sde_step_indices.append(int(step_idx))

        self._step_index += 1

        # Return prev_sample in the model's working dtype so the loop's next
        # forward sees a consistent dtype (matches BAGEL's no-scheduler branch
        # which keeps x_t in the autocast dtype).
        return BagelSDEStepOutput(
            prev_sample=prev_sample.to(original_dtype),
            log_prob=log_prob,
            prev_sample_mean=prev_sample_mean,
        )

    # ------------------------------------------------------------------ #
    # Trajectory drain (called by the pipeline after the loop)
    # ------------------------------------------------------------------ #

    @property
    def last_sde_step_indices(self) -> List[int]:
        """Step indices that ran the SDE branch on the most recent loop."""
        return list(self._traj_sde_step_indices)

    def drain_trajectory(
        self,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Return ``(latents [1,T+1,seq,C], sigmas [T+1], timesteps [1,T+1], log_probs [1,K])`` or ``None``.

        Latents/timesteps are dense (length ``T+1``): position-0 is the input
        x_T captured on the first ``step`` plus ``T`` post-step states. Log-probs
        are length ``K = len(last_sde_step_indices)`` (``K == 0`` for the
        no-SDE / forward-process path → ``[1, 0]``). Batch dim is 1 (BAGEL runs
        navit bs=1 per request); ``build_image_segment`` concatenates per-request.

        ``sigmas`` is reconstructed from the σ the loop actually visited
        (position-0 ``sigma`` = full[0]; each step's ``sigma_next`` = full[1..T]),
        so it is a GENUINE echo of the worker's schedule — the response layer's
        ``verify_engine_used_sigmas`` then asserts it matches the engine-pinned
        ``req.sigmas`` (BAGEL builds σ internally, so a divergent
        num_inference_steps / shift surfaces here rather than de-syncing replay).
        """
        if not self._traj_latents:
            return None
        post_latents = torch.stack(self._traj_latents, dim=0)  # [T, seq, C]
        post_timesteps = torch.stack(self._traj_timesteps, dim=0)  # [T] = full[1..T]

        if self._initial_latent is not None and self._initial_timestep is not None:
            init_lat = self._initial_latent.to(post_latents.dtype)
            latents = torch.cat([init_lat.unsqueeze(0), post_latents], dim=0)  # [T+1, seq, C]
            sigmas_full = torch.cat([self._initial_timestep.reshape(1), post_timesteps], dim=0)  # [T+1]
        else:
            latents = post_latents
            sigmas_full = post_timesteps

        latents = latents.unsqueeze(0)  # [1, T+1, seq, C]
        timesteps = sigmas_full.unsqueeze(0)  # [1, T+1] (σ echo, batch dim for cat)

        if self._traj_log_probs:
            log_probs = torch.stack(self._traj_log_probs, dim=0).reshape(1, -1)  # [1, K]
        else:
            log_probs = latents.new_zeros((1, 0), dtype=torch.float32)

        return latents, sigmas_full.to(latents.device), timesteps, log_probs


__all__ = ["BagelFlowSDEScheduler", "BagelSDEStepOutput"]
