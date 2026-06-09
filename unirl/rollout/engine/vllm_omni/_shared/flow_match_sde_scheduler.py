"""Trajectory-capturing SDE flow-match scheduler.

Port of ``vllm-omni/tests/e2e/offline_inference/custom_pipeline/flow_match_sde_scheduler.py``
keeping the ``sde`` branch only. Used by both the HunyuanImage-3 and
SD3.5 RL pipeline subclasses — their denoise loops both call
``self.scheduler.step(pred, t, latents, **_extra, return_dict=False)[0]``,
so we hijack ``step`` to do SDE math and stash a per-step
``(prev_sample, timestep, log_prob)`` triple on the instance for the
calling pipeline to drain after the loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor


@dataclass
class FlowMatchSDESchedulerOutput(BaseOutput):
    """``return_dict=True`` payload for :class:`FlowMatchSDEDiscreteScheduler`."""

    prev_sample: torch.Tensor
    log_prob: Optional[torch.Tensor]
    prev_sample_mean: torch.Tensor
    std_dev_t: torch.Tensor


class FlowMatchSDEDiscreteScheduler(FlowMatchEulerDiscreteScheduler):
    """SDE flow-match scheduler with on-instance trajectory capture.

    Math (closed-form Gaussian transition for the standard flow-matching SDE)::

        std_dev_t        = sqrt(σ / (1 - σ_max_clamp(σ))) * eta
        prev_sample_mean = sample * (1 + std_dev_t² / (2σ) * dt)
                         + model_output * (1 + std_dev_t² * (1-σ) / (2σ)) * dt
        prev_sample      = prev_sample_mean + std_dev_t * sqrt(-dt) * randn
        log_prob         = -((prev_sample.detach() - prev_sample_mean)²) /
                            (2 * (std_dev_t * sqrt(-dt))²)
                         - log(std_dev_t * sqrt(-dt))
                         - log(sqrt(2π))

    where ``dt = sigma_prev - sigma`` is negative on a decreasing schedule
    so ``sqrt(-dt)`` is real.

    The ``log_prob`` is mean-reduced across all non-batch dims so it ends up
    shape ``[B]`` per step. After the full denoise loop, calling code reads
    ``self._traj_latents``, ``self._traj_timesteps``, ``self._traj_log_probs``
    (each a list of per-step tensors) and stacks along ``dim=1`` to get
    ``[B, T, ...]`` trajectories.

    ``set_timesteps`` clears the trajectory buffers — that's the natural
    "new request" boundary in the calling pipeline.
    """

    # Stash trajectories on the instance so step() doesn't need to plumb
    # them through the diffusers ``[0] -> prev_sample`` return convention
    # the calling loop relies on.
    _traj_latents: List[torch.Tensor]
    _traj_timesteps: List[torch.Tensor]
    _traj_log_probs: List[torch.Tensor]
    _traj_sde_step_indices: List[int]

    def __init__(self, *args, eta: float = 1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # ``eta == 0`` is allowed: in that case every step takes the pure
        # Euler ODE branch (we gate the SDE math on ``_sde_indices_set``,
        # not on eta — see ``step``). Installing this scheduler with eta=0
        # is the right call when the algorithm has no SDE step but we
        # still need the dense latent trajectory captured (DiffusionNFT and other
        # forward-process flows that go through ``resp_to_samples`` which
        # requires ``segment.latents`` to be non-empty). Negative eta is
        # still nonsense.
        if eta < 0.0:
            raise ValueError(f"FlowMatchSDEDiscreteScheduler.eta must be >= 0; got eta={eta!r}.")
        # Stored on a shadow attr so the dataclass-style register_to_config of
        # the parent class doesn't try to serialize it via ``self.config``.
        self._eta = float(eta)
        self._traj_latents = []
        self._traj_timesteps = []
        self._traj_log_probs = []
        # Step indices that actually ran the SDE branch (and therefore wrote
        # an entry to ``_traj_log_probs``). Empty when ``_sde_indices_set``
        # is ``None`` / empty (no SDE step fires).
        self._traj_sde_step_indices = []
        # Position-0 capture — see ``step`` below. Stored separately so
        # ``drain_trajectory`` can prepend it without polluting the
        # per-step buffers (which must stay length-T to match log_probs).
        self._initial_latent: Optional[torch.Tensor] = None
        self._initial_timestep: Optional[torch.Tensor] = None
        # SDE-vs-ODE per-step gating, set by the pipeline subclass before
        # each forward (sourced from the resolved ``sde_indices`` on the request,
        # set at request construction in the trainer's ``_build_req``):
        #
        # - ``None`` / ``frozenset()`` — NO step runs SDE (forward-process
        #   / DiffusionNFT path). Every step takes the Euler ODE branch and writes
        #   no entry to ``_traj_log_probs``.
        # - ``frozenset({i,…})`` — those step indices run the SDE branch
        #   + capture log_prob; all other steps degenerate to Euler ODE.
        #
        # Latent / timestep capture stays dense across both kinds of steps
        # so trainer-side replay always has ``x_t`` at every storage slot.
        # This scheduler is installed unconditionally by the pipeline
        # subclass — ``resp_to_samples`` requires ``segment.latents`` to
        # be non-empty regardless of SDE choice, and only this scheduler
        # captures that trajectory.
        self._sde_indices_set: Optional[frozenset] = None

    # ------------------------------------------------------------------
    # Schedule lifecycle
    # ------------------------------------------------------------------

    def set_timesteps(
        self,
        num_inference_steps=None,
        device=None,
        sigmas=None,
        mu=None,
        timesteps=None,
    ):  # type: ignore[override]
        """Reset trajectory buffers; build σ schedule with the upstream
        static-shift double-application bug neutralized.

        Diffusers' ``FlowMatchEulerDiscreteScheduler.set_timesteps``
        applies the time shift in step 2 unconditionally — even when
        the caller passes ``sigmas`` externally (see issue #13243 / PR
        #13246 unmerged as of 2026-05). For UniRL we treat
        ``sigmas`` as **final values** computed main-side via
        :meth:`unirl.sde.runtime.FlowMatchSchedulePolicy.compute_sigma`,
        so any further shift on the worker would double-apply.

        Cross-repo precedent
        --------------------
        The same fix exists in the celve/sglang fork at commit
        ``2c5a2ecec`` "Support external sigma schedules for unirl
        alignment" (`github.com/celve/sglang@diffusionrl`,
        ``python/sglang/multimodal_gen/runtime/models/schedulers/
        scheduling_flow_match_euler_discrete.py:347-360``). That fork
        guards step 2 with ``if sigmas is None:`` at the source.

        **Upstream ``sgl-project/sglang`` does NOT have this fix** —
        we cannot rely on it. Our pyproject pins
        ``sglang @ git+https://github.com/celve/sglang.git@diffusionrl``
        so the sglang rollout path is fine; the vllm-omni rollout path
        falls under this scheduler subclass, hence this in-repo patch.

        Implementation
        --------------
        When ``sigmas`` is provided externally, transiently swap
        ``self._internal_dict`` to one with ``use_dynamic_shifting=False``
        AND set ``self._shift = 1.0`` (note: ``self.shift`` is a
        ``@property`` over ``_shift``, NOT a FrozenDict entry) so step
        2's two branches both collapse to identity, then restore via
        ``finally``. This is the smallest patch that's stable against
        future diffusers refactors (we don't reimplement steps 3-6).

        ``_sde_indices_set`` is intentionally NOT reset here — the
        calling pipeline installs it per-request right before driving
        the denoise loop; resetting would silently revert to "no SDE".
        """
        if sigmas is not None:
            # Neutralize step-2 shift for the duration of this call so
            # externally-provided sigmas (already shifted main-side)
            # don't get double-shifted by the parent's static or
            # dynamic branch. Two distinct sources to override:
            #
            # * ``self.config.use_dynamic_shifting`` — a ``FrozenDict``
            #   entry read at line ~347 of diffusers' set_timesteps to
            #   pick the static vs dynamic branch. Swap
            #   ``self._internal_dict`` to a copy with this flag False.
            #
            # * ``self.shift`` — a ``@property`` reading the instance
            #   attribute ``self._shift`` (NOT the FrozenDict), used at
            #   line ~350 inside the static branch. Set ``_shift`` to
            #   1.0 (identity) so the formula collapses to ``sigmas``.
            #
            # ``FrozenDict`` blocks ``__setitem__``, so the FrozenDict
            # itself can't be mutated in place — instance attributes
            # can. Both are restored in ``finally`` so the scheduler
            # config is unchanged across calls.
            from diffusers.configuration_utils import FrozenDict

            original_internal = self._internal_dict
            original_shift = self._shift
            overrides = dict(original_internal)
            overrides["use_dynamic_shifting"] = False
            self._internal_dict = FrozenDict(overrides)
            self._shift = 1.0
            try:
                out = super().set_timesteps(
                    num_inference_steps=num_inference_steps,
                    device=device,
                    sigmas=sigmas,
                    mu=mu,
                    timesteps=timesteps,
                )
            finally:
                self._internal_dict = original_internal
                self._shift = original_shift
        else:
            out = super().set_timesteps(
                num_inference_steps=num_inference_steps,
                device=device,
                sigmas=sigmas,
                mu=mu,
                timesteps=timesteps,
            )
        self._traj_latents = []
        self._traj_timesteps = []
        self._traj_log_probs = []
        self._traj_sde_step_indices = []
        self._initial_latent = None
        self._initial_timestep = None
        return out

    # ------------------------------------------------------------------
    # Per-step transition
    # ------------------------------------------------------------------

    def step(  # type: ignore[override]
        self,
        model_output: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = False,
        # Accept and ignore other kwargs that diffusers callers pass through
        # ``prepare_extra_func_kwargs`` (e.g. ``s_churn``, ``s_tmin``).
        **_unused,
    ) -> Union[FlowMatchSDESchedulerOutput, Tuple[torch.Tensor, ...]]:
        """SDE Flow-Match transition with trajectory capture.

        Calling denoise loops index ``[0]`` on the return value to extract
        ``prev_sample``; both the dataclass and tuple branches honor that.
        """
        if isinstance(timestep, (int, torch.IntTensor, torch.LongTensor)):
            raise ValueError(
                "FlowMatchSDEDiscreteScheduler.step expects a float-typed timestep "
                "from scheduler.timesteps (not an integer step index)."
            )
        if self.step_index is None:
            self._init_step_index(timestep)

        # Position-0 capture — stash the input ``sample`` (initial noise
        # for the first call after set_timesteps) so trajectory_latents
        # can be returned shape ``[B, T+1, ...]``. Only fires on the first
        # step() per request.
        if self._initial_latent is None:
            self._initial_latent = sample.detach().clone()
            if torch.is_tensor(timestep):
                init_t = timestep.detach().to(sample.device).clone()
            else:
                init_t = torch.as_tensor(float(timestep), device=sample.device)
            self._initial_timestep = init_t.expand(sample.shape[0]).clone()

        # The SDE math requires fp32 to keep variance noise well-conditioned.
        original_dtype = sample.dtype
        sample_f32 = sample.to(torch.float32)
        model_output_f32 = model_output.to(torch.float32)

        sigma_idx = self.step_index
        sigma = self.sigmas[sigma_idx]
        sigma_prev = self.sigmas[sigma_idx + 1]
        sigma_max = self.sigmas[1]
        dt = sigma_prev - sigma

        # SDE vs ODE per step: gated entirely on ``_sde_indices_set``.
        # ``None`` / empty → no step runs SDE (forward-process / DiffusionNFT).
        # Non-empty set → only those step indices run the SDE branch;
        # all others run pure Euler ODE.
        if self._sde_indices_set is None or len(self._sde_indices_set) == 0:
            step_is_sde = False
        else:
            step_is_sde = int(sigma_idx) in self._sde_indices_set

        if step_is_sde:
            # SDE branch: std_dev_t > 0 (eta-scaled). std_dev_t denominator
            # is clamped so sigma==1 doesn't divide by zero (last step on a
            # flow-matching schedule). The SDE-form drift correction
            # (``+ std_dev_t² / (2σ) * dt`` etc.) is needed here because
            # the noise we'll add below has variance matching that exact
            # form's expected magnitude.
            #
            # eta==0 would degenerate the log-prob density (log(0) = -inf);
            # if we land here that's an upstream wiring bug — sde_indices
            # said this step is SDE but the scheduler was constructed with
            # eta=0. Fail fast with a clear message rather than emit NaNs.
            if float(self._eta) <= 0.0:
                raise RuntimeError(
                    f"FlowMatchSDEDiscreteScheduler.step: step_index={int(sigma_idx)} "
                    f"is in _sde_indices_set but scheduler eta={self._eta!r}; "
                    f"eta must be > 0 for SDE steps. Check pipeline "
                    f"_ensure_scheduler_for_eta + driver sde_indices wiring."
                )
            std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))) * self._eta
            prev_sample_mean = (
                sample_f32 * (1 + std_dev_t**2 / (2 * sigma) * dt)
                + model_output_f32 * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
            )

            variance_noise = randn_tensor(
                model_output_f32.shape,
                generator=generator,
                device=model_output_f32.device,
                dtype=torch.float32,
            )
            prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-dt) * variance_noise

            # Cast prev_sample back to the model's dtype FIRST (this is the
            # value we store on the trajectory and what trainer-side replay
            # will read back via ``segment.latents_at(step_idx+1)``). Then
            # re-cast to fp32 for the log-prob density, so old_log_prob is
            # computed against the same bf16-roundtripped sample the replay
            # path sees. Without this round-trip the rollout records
            # ``logp(fp32 perfect | fp32 mu)`` while replay later computes
            # ``logp(bf16-stored | fp32 mu')``, producing a non-zero
            # ratio drift even on the on-policy first step.
            prev_sample = prev_sample.to(original_dtype)
            prev_sample_for_logp = prev_sample.to(torch.float32)

            # Closed-form Gaussian log-density of the actually-drawn (and
            # storage-roundtripped) prev_sample given the predicted mean.
            log_prob_per_elem = (
                -((prev_sample_for_logp.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-dt)) ** 2))
                - torch.log(std_dev_t * torch.sqrt(-dt))
                - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
            )
            # Mean over all non-batch dims → shape [B]
            log_prob: Optional[torch.Tensor] = log_prob_per_elem.mean(dim=tuple(range(1, log_prob_per_elem.ndim)))
        else:
            # Pure Euler ODE step. CRITICAL: do NOT use the SDE-form mean
            # (the ``+ std_dev_t² / (2σ) * dt`` corrections) here, even
            # though ``self._eta > 0`` — the trainer-side replay drives
            # non-SDE steps with ``eta=0`` (see
            # ``unirl/models/sd3/diffusion.py:375``:
            # ``step_eta = float(params.eta) if i in sde_set else 0.0``),
            # which collapses the mean to plain Euler ``sample + v·dt``.
            # Using SDE-form mean here would put the rollout trajectory
            # off the trainer's replay manifold on every non-SDE step
            # and bias every subsequent SDE-step log-prob.
            std_dev_t = sigma.new_zeros(())
            prev_sample_mean = sample_f32 + model_output_f32 * dt
            prev_sample = prev_sample_mean.to(original_dtype)
            log_prob = None

        # Trajectory stash — DENSE for latents/timesteps (every storage
        # slot recorded so replay has ``x_t`` at every step), SPARSE for
        # log_probs (only SDE-branch steps contribute).
        self._traj_latents.append(prev_sample.detach().clone())
        if torch.is_tensor(timestep):
            t_for_capture = timestep.detach().to(prev_sample.device).clone()
        else:
            t_for_capture = torch.as_tensor(float(timestep), device=prev_sample.device)
        # Broadcast to [B] so torch.stack(..., dim=1) yields [B, T] cleanly.
        self._traj_timesteps.append(t_for_capture.expand(prev_sample.shape[0]).clone())
        if log_prob is not None:
            self._traj_log_probs.append(log_prob.detach().clone())
            self._traj_sde_step_indices.append(int(sigma_idx))

        # Advance step index — required by parent's ``step_index`` lifecycle.
        self._step_index += 1

        if return_dict:
            return FlowMatchSDESchedulerOutput(
                prev_sample=prev_sample,
                log_prob=log_prob,
                prev_sample_mean=prev_sample_mean,
                std_dev_t=std_dev_t,
            )
        return (prev_sample, log_prob, prev_sample_mean, std_dev_t)

    # ------------------------------------------------------------------
    # Trajectory drain (called by the wrapping pipeline after the loop)
    # ------------------------------------------------------------------

    @property
    def last_sde_step_indices(self) -> List[int]:
        """Return the list of step indices that ran SDE on the most recent denoise loop.

        ``[]`` when no SDE step recorded (either ``_sde_indices_set`` was
        ``None`` / empty, or ``step()`` simply wasn't called). Otherwise
        it's the subset of ``range(T)`` the caller requested, in actual
        evaluation order (monotonically increasing under normal scheduler
        usage).
        """
        return list(self._traj_sde_step_indices)

    def drain_trajectory(
        self,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Return ``(latents [B,T+1,...], sigmas [T+1], timesteps [B,T+1], log_probs [B,K])`` or ``None``.

        Latents / timesteps are dense — length ``T+1`` — regardless of SDE
        vs ODE gating: position-0 is the input ``sample`` captured on the
        first ``step()`` call (the x_T), plus ``T`` post-step states. This
        is required by the trainer-side clean-latents replay path
        (``resp_to_samples`` raises when ``segment.latents`` is empty),
        which is why the pipeline subclasses install this scheduler
        unconditionally (see ``RLStableDiffusion3Pipeline._ensure_scheduler_for_eta``).

        Log-probs length ``K = len(last_sde_step_indices)``:
        - ``K == T`` when ``_sde_indices_set`` covers every step
        - ``K < T`` for any sparse subset
        - ``K == 0`` when ``_sde_indices_set`` is ``None`` or empty
          (forward-process / DiffusionNFT path; no SDE step fires, no log_prob
          captured). Returned as ``[B, 0]`` so downstream tensor ops
          have a real-but-empty tensor to consume — response.py then
          collapses this to ``segment.sde_logp = None``.

        Sigmas come straight from the parent's canonical
        ``self.sigmas`` schedule — 1D ``[T+1]`` in [0, 1], unbatched.
        Replay reads ``segment.sigmas[step_idx]`` directly so the segment
        is self-contained — no external schedule reconstruction needed.
        To recover which step IDs the K log_probs correspond to, read
        :attr:`last_sde_step_indices` AFTER calling this method.

        Returns ``None`` if no steps were recorded since the last
        ``set_timesteps`` call. Does not clear the buffers — the next
        ``set_timesteps`` call does that, so re-reads of the same
        trajectory are idempotent.
        """
        if not self._traj_latents:
            return None
        post_latents = torch.stack(self._traj_latents, dim=1)
        post_timesteps = torch.stack(self._traj_timesteps, dim=1)
        if self._traj_log_probs:
            log_probs = torch.stack(self._traj_log_probs, dim=1)
        else:
            # No SDE step fired (``_sde_indices_set`` was None or empty,
            # i.e. DiffusionNFT / forward-process). ``[B, 0]`` keeps tensor ops
            # alive; response.py collapses to ``sde_logp = None`` for the
            # clean-latents segment.
            B = post_latents.shape[0]
            log_probs = post_latents.new_zeros((B, 0), dtype=torch.float32)

        if self._initial_latent is not None and self._initial_timestep is not None:
            # Prepend position-0: cast initial latent to the post-step
            # dtype so torch.cat is well-typed (post-step is the model's
            # original dtype, possibly bf16; initial latent matches the
            # input ``sample`` dtype which is the same).
            init_lat = self._initial_latent.to(post_latents.dtype).unsqueeze(1)
            init_ts = self._initial_timestep.to(post_timesteps.dtype).unsqueeze(1)
            latents = torch.cat([init_lat, post_latents], dim=1)
            timesteps = torch.cat([init_ts, post_timesteps], dim=1)
        else:
            latents = post_latents
            timesteps = post_timesteps

        T_plus_1 = int(latents.shape[1])
        sigmas = self.sigmas[:T_plus_1].detach().clone().to(latents.device)

        return latents, sigmas, timesteps, log_probs


__all__ = ["FlowMatchSDEDiscreteScheduler", "FlowMatchSDESchedulerOutput"]
