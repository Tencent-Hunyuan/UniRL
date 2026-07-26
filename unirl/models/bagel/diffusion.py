"""Bagel diffusion: typed params + per-step kernel + rollout-level stage.

Bagel (BAGEL-7B-MoT) is a unified MoT flow-matching T2I model. Unlike SD3 /
HunyuanImage3 it does not expose a dense ``predict_noise(sample, sigma)``: its
forward (``_forward_flow``) consumes a *packed* (navit) sequence plus three KV-cache
contexts (gen / cfg_text / cfg_img) and returns the CFG-combined velocity ``v_t``.

This stage reads **exactly like** :class:`unirl.models.sd3.diffusion.SD3DiffusionStage`
— it rides UniRL's shared diffusion runtime and only swaps in Bagel's velocity call:

- **σ / timestep schedule** comes from ``req.sigmas`` (pinned by the engine via
  :func:`unirl.sde.runtime.ensure_req_sigmas` from the pipeline's
  ``build_schedule_policy``), passed in as ``schedule`` — NOT computed here.
- **which steps run SDE** comes from ``params.sde_indices`` (the driver resolved it
  via :meth:`DiffusionSamplingParams.resolve_sde_indices` → the recipe's indices
  scheduler, ``unirl.utils.scheduler_utils.AllSDEScheduler``) — NOT drawn here.
- **the SDE transition + log-prob** is :class:`unirl.sde.kernels.FlowSDEStrategy`
  (``strategy.denoise``) — NOT a flow_grpo port.
- **the initial noise x_T** is the driver-authored :class:`NoiseRecipe` value passed
  as ``initial_latents`` — NOT drawn here.

The only Bagel-specific machinery left is the navit adapter: building the three
packed KV contexts (``_build_generation_inputs`` / ``_forward_kwargs``), the per-step
CFG gate (``_gated_cfg_scales``), and the velocity call (``rl_ops.forward_flow`` over
the pristine ``_forward_flow``, grad-capable via ``__wrapped__``). Bagel runs navit
``bs=1`` so packed latents are ``[seq, C]`` (no batch dim); the kernel call adds a
unit batch dim so ``FlowSDEStrategy``'s per-sample log-prob reduction matches.

The on-policy invariant: under identical weights, replay's ``new_logp`` matches the
rollout's emitted ``old_logp`` so the PPO ratio ``exp(new-old) ≈ 1`` — exactly as for
SD3, because rollout and replay use the SAME ``FlowSDEStrategy`` over the same stored
fp32 trajectory.

This module deliberately avoids importing the vendored modeling (and its hard
``flash_attn`` dependency) at module load — it reaches the model methods through the
bundle instance at call time — so ``BagelDiffusionParams`` stays CPU-importable.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import torch

from unirl.config.require import require
from unirl.models.types.diffusion import DiffusionStage
from unirl.models.types.replay_result import ReplayResult
from unirl.sde.kernels import FlowSDEStrategy, StepStrategy
from unirl.types.sampling import DiffusionSamplingParams, compute_trajectory_positions
from unirl.types.segments.latent import LatentSegment
from unirl.utils.dtypes import parse_torch_dtype

from . import rl_ops
from .conditions import BagelDiffusionConditions

if TYPE_CHECKING:
    from .bundle import BagelBundle

CFG_RENORM_TYPES = ("global", "channel", "text_channel")


@dataclass
class BagelDiffusionParams(DiffusionSamplingParams):
    """Bagel diffusion knobs — one object for BOTH the trainer and the stage.

    Subclasses :class:`DiffusionSamplingParams` so a single ``sampling`` config
    object satisfies the two consumers that share it: the ``DiffusionTrainer``
    reads inherited fields (``samples_per_prompt`` for the GRPO group fan-out,
    ``height`` / ``width`` / ``seed`` / ``num_inference_steps`` / ``eta`` /
    ``init_same_noise`` / ``scheduler`` / ``sde_indices``), while
    :class:`BagelDiffusionStage` reads the Bagel-specific CFG knobs added here.

    Bagel runs **one prompt at a time** (``bs=1`` packed): ``samples_per_prompt``
    fan-out is materialized by the trainer (``RolloutInputs.expand``) into separate
    samples, not batched inside one ``_forward_flow``.

    The SDE machinery is now **all inherited / central**: ``eta`` is the SDE noise
    scale (flow_grpo's ``noise_level``); ``scheduler`` (the recipe's
    :class:`~unirl.utils.scheduler_utils.AllSDEScheduler`) picks the SDE steps via
    :meth:`resolve_sde_indices`; the σ schedule rides on ``req.sigmas``. No
    Bagel-specific window / schedule fields remain.
    """

    # Override base defaults for Bagel. ``num_inference_steps`` is the number of
    # STEPS (the σ schedule has steps+1 points); BAGEL's flow_grpo setup uses a
    # 15-point schedule → 14 steps.
    num_inference_steps: int = 14
    guidance_scale: float = 1.0
    height: int = 512
    width: int = 512
    # SDE noise scale (== flow_grpo's ``noise_level``); consumed by FlowSDEStrategy
    # as ``eta``. Inherited field, surfaced here for the BAGEL default.
    eta: float = 1.0

    # Bagel-specific CFG knobs (consumed by the navit ``_forward_flow``).
    cfg_text_scale: float = 1.0
    cfg_img_scale: float = 1.0
    cfg_interval: Tuple[float, float] = (0.0, 1.0)
    cfg_renorm_min: float = 0.0
    cfg_renorm_type: str = "global"
    cfg_type: str = "parallel"

    def __post_init__(self) -> None:
        super().__post_init__()
        require(
            int(self.num_inference_steps) >= 2,
            f"BagelDiffusionParams.num_inference_steps must be >= 2; got {self.num_inference_steps}",
        )
        require(
            self.cfg_renorm_type in CFG_RENORM_TYPES,
            f"BagelDiffusionParams.cfg_renorm_type must be one of {CFG_RENORM_TYPES}; got {self.cfg_renorm_type!r}",
        )


# The vendored ``prepare_*`` helpers build their packed index tensors on CPU;
# the MoT forward needs them on the model device. Shared with the AR adapters.
_to_device = rl_ops._to_device


class BagelDiffusionStep:
    """Per-step Bagel kernel — stateless navit adapter over the shared SDE strategy.

    Splits the step into the two Bagel-specific halves SD3 also has:
    :meth:`predict_velocity` (the navit ``_forward_flow`` call, with CFG done inside
    the model) and :meth:`denoise` (the shared :class:`StepStrategy.denoise`, with a
    unit batch dim added so the packed ``[seq, C]`` latent gets a per-sample log-prob).
    :meth:`step_with_logp` runs both, mirroring ``SD3DiffusionStep.step_with_logp``.
    """

    def predict_velocity(
        self,
        bagel: Any,
        *,
        x_t: torch.Tensor,
        t_cur: torch.Tensor,
        cfg_text_scale: float,
        cfg_img_scale: float,
        forward_kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        """CFG-combined velocity ``v_t`` for packed ``x_t`` ``[seq, C]`` at time ``t_cur``.

        ``_forward_flow`` takes a per-token ``timestep`` ``[seq]`` (all equal to the
        scalar ``t_cur``) and does the CFG combine internally (gen / cfg_text /
        cfg_img contexts in ``forward_kwargs`` + the gated scales).
        """
        # The pristine ``_forward_flow`` reads ``language_model.model.enable_taylorseer``
        # (the official ``generate_image`` sets it; the RL path calls ``_forward_flow``
        # directly). Set it here — the single chokepoint before every velocity call —
        # so the TaylorSeer cache is off (per-step determinism for replay). Idempotent.
        rl_ops.disable_inference_cache(bagel)
        seq = int(x_t.shape[0])
        timestep = torch.full((seq,), float(t_cur), device=x_t.device)
        return rl_ops.forward_flow(
            bagel,
            x_t=x_t,
            timestep=timestep,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            **forward_kwargs,
        )

    def denoise(
        self,
        strategy: StepStrategy,
        *,
        v_t: torch.Tensor,
        x_t: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        sigma_max: torch.Tensor,
        eta: float,
        prev_sample: Optional[torch.Tensor] = None,
        n_samples: int = 1,
        generators: Optional[List[torch.Generator]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """One SDE transition via the shared ``strategy.denoise`` over packed latents.

        Packed Bagel latents are ``[seq, C]`` (navit ``bs=1``, no batch dim), but
        ``FlowSDEStrategy`` reduces the per-element log-prob over dims ``>=1`` to get
        ONE scalar per sample (it assumes a leading batch dim). So we add a unit
        batch dim (``[1, seq, C]``) for the call and squeeze it back: the log-prob is
        then the mean over all ``seq*C`` elements — identical to flow_grpo's
        ``log_prob.mean()``. Returns ``(prev_sample, log_prob, prev_sample_mean)``;
        ``log_prob`` / ``prev_sample_mean`` are ``None`` for deterministic
        (``eta < 1e-7``) steps.
        """
        require(n_samples >= 1, f"BagelDiffusionStep.denoise: n_samples must be >= 1; got {n_samples}.")
        require(
            int(x_t.shape[0]) % int(n_samples) == 0,
            f"BagelDiffusionStep.denoise: packed token count {int(x_t.shape[0])} "
            f"is not divisible by n_samples={n_samples}.",
        )
        seq = int(x_t.shape[0]) // int(n_samples)
        channels = int(x_t.shape[-1])
        sample = x_t.reshape(int(n_samples), seq, channels)
        noise_pred = v_t.reshape(int(n_samples), seq, channels)
        replay_sample = None if prev_sample is None else prev_sample.reshape(int(n_samples), seq, channels)
        if generators is None or replay_sample is not None:
            prev, log_prob, prev_mean = strategy.denoise(
                noise_pred=noise_pred,
                sample=sample,
                sigma=sigma,
                sigma_next=sigma_next,
                eta=float(eta),
                prev_sample=replay_sample,
                sigma_max=float(sigma_max),
            )
        else:
            if len(generators) != int(n_samples):
                raise ValueError(
                    f"BagelDiffusionStep.denoise: got {len(generators)} generators for n_samples={n_samples}."
                )
            input_dtype = sample.dtype
            noise_pred_f32 = noise_pred.float()
            sample_f32 = sample.float()
            sigma_f32 = sigma.float().reshape(1)
            sigma_next_f32 = sigma_next.float().reshape(1)
            while sigma_f32.dim() < sample_f32.dim():
                sigma_f32 = sigma_f32.unsqueeze(-1)
                sigma_next_f32 = sigma_next_f32.unsqueeze(-1)
            prev, prev_mean, std_var = strategy.step(
                noise_pred=noise_pred_f32,
                sample=sample_f32,
                sigma=sigma_f32,
                sigma_next=sigma_next_f32,
                eta=float(eta),
                prev_sample=None,
                generator=generators,
                sigma_max=float(sigma_max),
            )
            prev, log_prob = strategy._finalize_logp(
                prev_sample=prev,
                prev_sample_mean=prev_mean,
                std_var=std_var,
                eta=float(eta),
                input_dtype=input_dtype,
            )
        if n_samples == 1:
            return (
                prev.reshape(seq, channels),
                None if log_prob is None else log_prob.reshape(()),
                None if prev_mean is None else prev_mean.reshape(seq, channels),
            )
        return (
            prev.reshape(int(n_samples) * seq, channels),
            None if log_prob is None else log_prob.reshape(int(n_samples)),
            None if prev_mean is None else prev_mean.reshape(int(n_samples) * seq, channels),
        )

    def step_with_logp(
        self,
        bagel: Any,
        strategy: StepStrategy,
        *,
        x_t: torch.Tensor,
        prev_sample: Optional[torch.Tensor],
        t_cur: torch.Tensor,
        t_next: torch.Tensor,
        sigma_max: torch.Tensor,
        eta: float,
        cfg_text_scale: float,
        cfg_img_scale: float,
        forward_kwargs: Dict[str, Any],
        n_samples: int = 1,
        generators: Optional[List[torch.Generator]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run ``predict_velocity`` then ``denoise`` for one step.

        ``prev_sample=None`` ⇒ sampling (draws a fresh next sample); ``prev_sample``
        set ⇒ replay (log-prob of the stored transition). Returns
        ``(prev_sample, log_prob, prev_sample_mean)``.
        """
        v_t = self.predict_velocity(
            bagel,
            x_t=x_t,
            t_cur=t_cur,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            forward_kwargs=forward_kwargs,
        )
        return self.denoise(
            strategy,
            v_t=v_t,
            x_t=x_t,
            sigma=t_cur,
            sigma_next=t_next,
            sigma_max=sigma_max,
            eta=eta,
            prev_sample=prev_sample,
            n_samples=n_samples,
            generators=generators,
        )


class BagelDiffusionStage(DiffusionStage[BagelDiffusionConditions]):
    """Bagel rollout-level diffusion stage (trainside A1) — central-runtime, SD3-shaped.

    Owns the bundle, the per-step navit kernel, the SDE ``strategy``, and the
    precision policy. ``diffuse`` runs the full sampling loop over the engine-pinned
    ``schedule`` (``req.sigmas``), recording SDE log-probs at ``params.sde_indices``;
    ``replay`` recomputes those log-probs for GRPO. Reuses
    ``unirl.algorithms.flowgrpo.FlowGRPO`` unchanged.

    Implements the ``DiffusionStage`` protocol (``diffuse`` / ``replay`` /
    ``predict_noise_at_step``) so the trainside engine's ``isinstance(stage,
    DiffusionStage)`` check passes → it builds the σ-schedule policy from
    :meth:`BagelPipeline.build_schedule_policy` and pins ``req.sigmas`` via
    ``ensure_req_sigmas`` before ``generate`` (same path as SD3 / Wan / Qwen-Image).
    """

    def __init__(
        self,
        *,
        model: "BagelBundle",
        step: Optional[BagelDiffusionStep] = None,
        strategy: Optional[StepStrategy] = None,
        autocast_precision: str = "bf16",
        trajectory_precision: str = "fp32",
        logprob_precision: str = "fp32",
    ) -> None:
        # ``model`` is the bundle (kept name-compatible with the other stages so
        # the pipeline / FSDPPolicy treat it uniformly). The Bagel nn.Module is
        # ``model.model``; the trainable MoT is ``model.transformer``.
        self.model = model
        self.step = step if step is not None else BagelDiffusionStep()
        self.strategy = strategy if strategy is not None else FlowSDEStrategy()
        self.autocast_dtype = parse_torch_dtype(autocast_precision, field_name="autocast_precision")
        self.trajectory_dtype = parse_torch_dtype(trajectory_precision, field_name="trajectory_precision")
        self.logprob_dtype = parse_torch_dtype(logprob_precision, field_name="logprob_precision")

    # ------------------------------------------------------------------
    # Helpers (navit adapter)
    # ------------------------------------------------------------------

    def _autocast_ctx(self, device: torch.device):
        if device.type == "cuda" and self.autocast_dtype in (torch.float16, torch.bfloat16):
            return torch.autocast("cuda", self.autocast_dtype)
        return nullcontext()

    @staticmethod
    def _sampling_generators(device: torch.device, count: int) -> List[torch.Generator]:
        """Fork stable per-sequence RNG streams from the current device RNG."""
        generators: List[torch.Generator] = []
        for _ in range(int(count)):
            seed = int(
                torch.randint(
                    0,
                    (1 << 63) - 1,
                    (),
                    dtype=torch.int64,
                    device=device,
                ).item()
            )
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
            generators.append(generator)
        return generators

    def _build_contexts_from_prompt(self, prompt: str) -> Tuple[Any, Any, Any]:
        """Rebuild the three KV contexts (gen / cfg_text / cfg_img) from a prompt.

        The vllm_omni rollout path ships only the prompt text (KV caches can't
        cross the worker→driver IPC boundary), so replay rebuilds the contexts
        here on the trainer's own bundle. Mirrors
        ``BagelPipeline._build_contexts`` (think=False, no image): ``cfg_text`` is
        the init snapshot taken BEFORE the prompt text (unconditional); ``gen``
        and ``cfg_img`` both ingest the prompt. The und/text path is frozen, so
        the rebuilt contexts equal those the rollout worker used.
        """
        from copy import deepcopy

        inf = self.model.inferencer
        device = torch.device(self.model.device)
        # Match the worker pipeline's prompt preprocessing EXACTLY
        # (vllm_omni/diffusion/models/bagel/pipeline_bagel.py): it strips any
        # ``<|im_start|>`` / ``<|im_end|>`` wrappers before ``prepare_prompts`` so
        # bos/eos aren't double-added. If replay tokenized a differently-wrapped
        # prompt, the KV context (and thus every step's v_t / log-prob) would
        # diverge from rollout — a silent rollout↔replay mismatch.
        clean_prompt = str(prompt).removeprefix("<|im_start|>").removesuffix("<|im_end|>")
        gen = inf.init_gen_context()
        cfg_img = deepcopy(gen)
        with torch.no_grad(), self._autocast_ctx(device):
            cfg_text = deepcopy(gen)  # snapshot before the prompt text → unconditional
            gen = inf.update_context_text(clean_prompt, gen)
            cfg_img = inf.update_context_text(clean_prompt, cfg_img)
        return gen, cfg_text, cfg_img

    def _resolve_single(self, conditions: BagelDiffusionConditions) -> Tuple[Any, Any, Any, Tuple[int, int]]:
        """Return ``(gen, cfg_text, cfg_img, image_shape)`` for a 1-sample batch.

        Two sources, transparently:

        - **opaque contexts present** (trainside / colocate): return them directly
          via :meth:`BagelDiffusionConditions.single`.
        - **deferred prompts only** (vllm_omni cross-process): rebuild the KV
          contexts from the shipped prompt on this bundle (the und path is frozen
          → identical contexts), then return them.
        """
        if conditions.has_contexts():
            return conditions.single()
        prompt, image_shape = conditions.single_prompt()
        gen, cfg_text, cfg_img = self._build_contexts_from_prompt(prompt)
        return gen, cfg_text, cfg_img, image_shape

    def _build_generation_inputs(
        self,
        gen: Any,
        cfg_text: Any,
        cfg_img: Any,
        image_shape: Tuple[int, int],
        *,
        device: torch.device,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Reconstruct the packed gen / cfg_text / cfg_img inputs from the contexts.

        Deterministic given (context kv_lens / ropes, image_shape) — the same packed
        index tensors ``_forward_flow`` consumes. ``gi["packed_init_noises"]`` is the
        vendored default x_T draw; ``diffuse`` overrides it with the driver-authored
        :class:`NoiseRecipe` value (``initial_latents``) when present and otherwise
        uses it as the fallback (tests / no driver recipe).
        """
        bagel = self.model.model
        gi = bagel.prepare_vae_latent(
            curr_kvlens=gen["kv_lens"],
            curr_rope=gen["ropes"],
            image_sizes=[image_shape],
            new_token_ids=self.model.new_token_ids,
        )
        gi_cfg_text = bagel.prepare_vae_latent_cfg(
            curr_kvlens=cfg_text["kv_lens"], curr_rope=cfg_text["ropes"], image_sizes=[image_shape]
        )
        gi_cfg_img = bagel.prepare_vae_latent_cfg(
            curr_kvlens=cfg_img["kv_lens"], curr_rope=cfg_img["ropes"], image_sizes=[image_shape]
        )
        return _to_device(gi, device), _to_device(gi_cfg_text, device), _to_device(gi_cfg_img, device)

    def _forward_kwargs(
        self,
        gen: Any,
        cfg_text: Any,
        cfg_img: Any,
        gi: Dict[str, Any],
        gi_cfg_text: Dict[str, Any],
        gi_cfg_img: Dict[str, Any],
        params: BagelDiffusionParams,
    ) -> Dict[str, Any]:
        """Static (step-invariant) kwargs for ``_forward_flow``.

        Everything except ``x_t`` / ``timestep`` / the per-step CFG scales. The three
        ``past_key_values`` come from the conditions' contexts; the packed index
        tensors come from the (device-pinned) generation inputs.
        """
        return dict(
            packed_vae_token_indexes=gi["packed_vae_token_indexes"],
            packed_vae_position_ids=gi["packed_vae_position_ids"],
            packed_text_ids=gi["packed_text_ids"],
            packed_text_indexes=gi["packed_text_indexes"],
            packed_position_ids=gi["packed_position_ids"],
            packed_indexes=gi["packed_indexes"],
            packed_seqlens=gi["packed_seqlens"],
            key_values_lens=gi["key_values_lens"],
            past_key_values=gen["past_key_values"],
            packed_key_value_indexes=gi["packed_key_value_indexes"],
            cfg_renorm_min=params.cfg_renorm_min,
            cfg_renorm_type=params.cfg_renorm_type,
            cfg_text_packed_position_ids=gi_cfg_text["cfg_packed_position_ids"],
            cfg_text_packed_query_indexes=gi_cfg_text["cfg_packed_query_indexes"],
            cfg_text_key_values_lens=gi_cfg_text["cfg_key_values_lens"],
            cfg_text_past_key_values=cfg_text["past_key_values"],
            cfg_text_packed_key_value_indexes=gi_cfg_text["cfg_packed_key_value_indexes"],
            cfg_img_packed_position_ids=gi_cfg_img["cfg_packed_position_ids"],
            cfg_img_packed_query_indexes=gi_cfg_img["cfg_packed_query_indexes"],
            cfg_img_key_values_lens=gi_cfg_img["cfg_key_values_lens"],
            cfg_img_past_key_values=cfg_img["past_key_values"],
            cfg_img_packed_key_value_indexes=gi_cfg_img["cfg_packed_key_value_indexes"],
            cfg_type=params.cfg_type,
        )

    @staticmethod
    def _gated_cfg_scales(t_value: float, params: BagelDiffusionParams) -> Tuple[float, float]:
        """CFG scales after the per-step ``cfg_interval`` gate (matches generate_image)."""
        lo, hi = float(params.cfg_interval[0]), float(params.cfg_interval[1])
        if lo < t_value <= hi:
            return float(params.cfg_text_scale), float(params.cfg_img_scale)
        return 1.0, 1.0

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def diffuse(
        self,
        conditions: BagelDiffusionConditions,
        *,
        schedule: torch.Tensor,
        params: BagelDiffusionParams,
        initial_latents: Optional[torch.Tensor] = None,
    ) -> LatentSegment:
        """Run Bagel sampling over the engine-pinned ``schedule``; return a segment.

        Mirrors ``SD3DiffusionStage.diffuse``: loops ``T = len(schedule) - 1`` steps,
        records an SDE log-prob (``eta`` noise) at every ``i in params.sde_indices``
        and runs deterministic Euler (``eta = 0``) elsewhere. ``initial_latents`` is
        the driver-authored x_T (:class:`NoiseRecipe`); when ``None`` the vendored
        ``prepare_vae_latent`` draw is used (tests / no driver recipe).

        The returned ``LatentSegment`` (``S = len(sde_indices)``, ``seq`` = packed
        image tokens, ``C`` = packed latent channels):

            latents      : [1, K, seq, C]   stored trajectory frames (window + final)
            sde_logp     : [1, S]           old per-step log-probs
            sde_indices  : [S]              SDE step indices
            indices      : [K]              stored frame step indices
            sigmas       : [T+1]            the full schedule
        """
        if conditions.batch_size > 1:
            if self._can_pack_conditions(conditions):
                return self._diffuse_batched(
                    conditions,
                    schedule=schedule,
                    params=params,
                    initial_latents=initial_latents,
                )
            return self._diffuse_serial_batch(
                conditions,
                schedule=schedule,
                params=params,
                initial_latents=initial_latents,
            )

        bagel = self.model.model
        device = torch.device(self.model.device)
        schedule = schedule.to(device)
        T = int(schedule.shape[0]) - 1
        require(
            T == int(params.num_inference_steps),
            f"BagelDiffusionStage.diffuse: schedule length {schedule.shape[0]} != "
            f"num_inference_steps+1 ({int(params.num_inference_steps) + 1})",
        )
        sigma_max = schedule[1] if int(schedule.shape[0]) > 1 else schedule[0]

        sde_set: Set[int] = set(int(i) for i in (params.sde_indices or []))
        sde_sorted: List[int] = sorted(sde_set)

        gen, cfg_text, cfg_img, image_shape = self._resolve_single(conditions)
        gi, gi_cfg_text, gi_cfg_img = self._build_generation_inputs(gen, cfg_text, cfg_img, image_shape, device=device)
        forward_kwargs = self._forward_kwargs(gen, cfg_text, cfg_img, gi, gi_cfg_text, gi_cfg_img, params)

        if initial_latents is not None:
            x_t = initial_latents.to(device=device, dtype=self.trajectory_dtype)
        else:
            x_t = gi["packed_init_noises"].to(device=device, dtype=self.trajectory_dtype)

        self.strategy.init_schedule(schedule)

        # Store SDE step boundaries (x_t before AND after each SDE step) so replay can
        # re-score them, plus the final clean latent (T) for VAE decode.
        needed: Set[int] = set(compute_trajectory_positions(sde_set, T))
        needed.add(T)
        stored_pairs: List[Tuple[int, torch.Tensor]] = []
        if 0 in needed:
            stored_pairs.append((0, x_t.detach().clone()))
        sde_logp_list: List[torch.Tensor] = []
        # Per-SDE-step mean μ_old (the deterministic transition mean), recorded on the
        # same SDE steps as sde_logp; consumed by BagelFlowUniGRPO(ratio_norm=True).
        sde_means_list: List[torch.Tensor] = []

        with torch.no_grad(), self._autocast_ctx(device):
            for i in range(T):
                t_cur = schedule[i]
                t_next = schedule[i + 1]
                cfg_text_scale, cfg_img_scale = self._gated_cfg_scales(float(t_cur.item()), params)
                step_eta = float(params.eta) if i in sde_set else 0.0
                x_t, log_prob, prev_mean = self.step.step_with_logp(
                    bagel,
                    self.strategy,
                    x_t=x_t,
                    prev_sample=None,
                    t_cur=t_cur,
                    t_next=t_next,
                    sigma_max=sigma_max,
                    eta=step_eta,
                    cfg_text_scale=cfg_text_scale,
                    cfg_img_scale=cfg_img_scale,
                    forward_kwargs=forward_kwargs,
                )
                x_t = x_t.to(dtype=self.trajectory_dtype)
                if (i + 1) in needed:
                    stored_pairs.append((i + 1, x_t.detach().clone()))
                if log_prob is not None:
                    sde_logp_list.append(log_prob.to(dtype=self.logprob_dtype))
                    # Nested on purpose: prev_mean is non-None on every step, so appending it
                    # outside this block would misalign sde_means with sde_logp / sde_indices.
                    if prev_mean is not None:
                        sde_means_list.append(prev_mean.detach().to(dtype=self.trajectory_dtype))

        positions_collected = [p for p, _ in stored_pairs]
        latents_stacked = torch.stack([t for _, t in stored_pairs], dim=0).unsqueeze(0)  # [1, K, seq, C]
        sde_logp = torch.stack(sde_logp_list, dim=0).unsqueeze(0) if sde_logp_list else None  # [1, S]
        sde_means = torch.stack(sde_means_list, dim=0).unsqueeze(0) if sde_means_list else None  # [1, S, seq, C]
        sde_indices = torch.tensor(sde_sorted, dtype=torch.long, device=device) if sde_sorted else None

        indices = torch.tensor(positions_collected, dtype=torch.long, device=device)

        return LatentSegment(
            latents=latents_stacked,
            sigmas=schedule,
            indices=indices,
            sde_logp=sde_logp,
            sde_means=sde_means,
            sde_indices=sde_indices,
        )

    # ------------------------------------------------------------------
    # Block-diagonal rollout batching
    # ------------------------------------------------------------------

    @staticmethod
    def _can_pack_conditions(conditions: BagelDiffusionConditions) -> bool:
        """Only opaque, same-shape contexts have an unambiguous packed latent geometry."""
        if not conditions.has_contexts() or conditions.batch_size < 2:
            return False
        if len(conditions.gen_contexts) != conditions.batch_size:
            return False
        shapes = [tuple(shape) for shape in conditions.image_shapes]
        return len(shapes) == conditions.batch_size and len(set(shapes)) == 1

    @staticmethod
    def _stack_segments(segments: List[LatentSegment]) -> LatentSegment:
        if len(segments) == 1:
            return segments[0]
        return LatentSegment(
            latents=torch.cat([segment.latents for segment in segments], dim=0),
            sigmas=segments[0].sigmas,
            indices=segments[0].indices,
            sde_logp=(
                torch.cat([segment.sde_logp for segment in segments], dim=0)
                if segments[0].sde_logp is not None
                else None
            ),
            sde_means=(
                torch.cat([segment.sde_means for segment in segments], dim=0)
                if segments[0].sde_means is not None
                else None
            ),
            sde_indices=segments[0].sde_indices,
        )

    def _diffuse_serial_batch(
        self,
        conditions: BagelDiffusionConditions,
        *,
        schedule: torch.Tensor,
        params: BagelDiffusionParams,
        initial_latents: Optional[torch.Tensor],
    ) -> LatentSegment:
        """Fallback for deferred contexts or mixed image shapes."""
        segments: List[LatentSegment] = []
        for index in range(conditions.batch_size):
            if conditions.has_contexts():
                gen = conditions.gen_contexts[index]
                cfg_text = (
                    conditions.cfg_text_contexts[index]
                    if conditions.cfg_text_contexts and conditions.cfg_text_contexts[index] is not None
                    else gen
                )
                cfg_img = (
                    conditions.cfg_img_contexts[index]
                    if conditions.cfg_img_contexts and conditions.cfg_img_contexts[index] is not None
                    else gen
                )
                condition = BagelDiffusionConditions.for_sample(
                    gen_context=gen,
                    cfg_text_context=cfg_text,
                    cfg_img_context=cfg_img,
                    prompt=conditions.prompts[index] if conditions.prompts else None,
                    image_shape=tuple(conditions.image_shapes[index]),
                )
            else:
                condition = BagelDiffusionConditions(
                    prompts=[conditions.prompts[index]],
                    image_shapes=[tuple(conditions.image_shapes[index])],
                )
            initial = initial_latents[index] if initial_latents is not None else None
            segments.append(self.diffuse(condition, schedule=schedule, params=params, initial_latents=initial))
        return self._stack_segments(segments)

    @staticmethod
    def _merge_contexts(contexts: List[Any]) -> Dict[str, Any]:
        """Merge per-sample NaiveCache objects while preserving empty CFG branches."""
        kv_lens = [int(context["kv_lens"][0]) for context in contexts]
        ropes = [int(context["ropes"][0]) for context in contexts]
        caches = [context["past_key_values"] for context in contexts]
        merged = type(caches[0])(int(caches[0].num_layers))
        for layer in range(int(caches[0].num_layers)):
            keys = [cache.key_cache[layer] for cache in caches if cache.key_cache[layer] is not None]
            values = [cache.value_cache[layer] for cache in caches if cache.value_cache[layer] is not None]
            merged.key_cache[layer] = torch.cat(keys, dim=0) if keys else None
            merged.value_cache[layer] = torch.cat(values, dim=0) if values else None
        return {"kv_lens": kv_lens, "ropes": ropes, "past_key_values": merged}

    def _build_generation_inputs_batched(
        self,
        gen: Any,
        cfg_text: Any,
        cfg_img: Any,
        image_shapes: List[Tuple[int, int]],
        *,
        device: torch.device,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Build the vendor's packed latent indexes for a multi-sequence context."""
        bagel = self.model.model
        gi = bagel.prepare_vae_latent(
            curr_kvlens=gen["kv_lens"],
            curr_rope=gen["ropes"],
            image_sizes=image_shapes,
            new_token_ids=self.model.new_token_ids,
        )
        gi_cfg_text = bagel.prepare_vae_latent_cfg(
            curr_kvlens=cfg_text["kv_lens"],
            curr_rope=cfg_text["ropes"],
            image_sizes=image_shapes,
        )
        gi_cfg_img = bagel.prepare_vae_latent_cfg(
            curr_kvlens=cfg_img["kv_lens"],
            curr_rope=cfg_img["ropes"],
            image_sizes=image_shapes,
        )
        return _to_device(gi, device), _to_device(gi_cfg_text, device), _to_device(gi_cfg_img, device)

    def _diffuse_batched(
        self,
        conditions: BagelDiffusionConditions,
        *,
        schedule: torch.Tensor,
        params: BagelDiffusionParams,
        initial_latents: Optional[torch.Tensor] = None,
    ) -> LatentSegment:
        """Run one block-diagonal navit forward per step for same-shape images."""
        bagel = self.model.model
        device = torch.device(self.model.device)
        schedule = schedule.to(device)
        num_steps = int(schedule.shape[0]) - 1
        require(
            num_steps == int(params.num_inference_steps),
            f"BagelDiffusionStage._diffuse_batched: schedule length {schedule.shape[0]} != "
            f"num_inference_steps+1 ({int(params.num_inference_steps) + 1})",
        )
        sigma_max = schedule[1] if int(schedule.shape[0]) > 1 else schedule[0]
        sde_set = {int(index) for index in (params.sde_indices or [])}
        sde_sorted = sorted(sde_set)
        batch_size = int(conditions.batch_size)

        gen_contexts = list(conditions.gen_contexts)
        cfg_text_contexts = [
            conditions.cfg_text_contexts[index]
            if conditions.cfg_text_contexts and conditions.cfg_text_contexts[index] is not None
            else gen_contexts[index]
            for index in range(batch_size)
        ]
        cfg_img_contexts = [
            conditions.cfg_img_contexts[index]
            if conditions.cfg_img_contexts and conditions.cfg_img_contexts[index] is not None
            else gen_contexts[index]
            for index in range(batch_size)
        ]
        image_shapes = [tuple(shape) for shape in conditions.image_shapes]
        gen = self._merge_contexts(gen_contexts)
        cfg_text = self._merge_contexts(cfg_text_contexts)
        cfg_img = self._merge_contexts(cfg_img_contexts)
        gi, gi_cfg_text, gi_cfg_img = self._build_generation_inputs_batched(
            gen,
            cfg_text,
            cfg_img,
            image_shapes,
            device=device,
        )
        forward_kwargs = self._forward_kwargs(gen, cfg_text, cfg_img, gi, gi_cfg_text, gi_cfg_img, params)

        if initial_latents is None:
            x_t = gi["packed_init_noises"].to(device=device, dtype=self.trajectory_dtype)
        else:
            x_t = initial_latents.to(device=device, dtype=self.trajectory_dtype).reshape(
                -1, int(initial_latents.shape[-1])
            )
        channels = int(x_t.shape[-1])
        require(
            int(x_t.shape[0]) % batch_size == 0,
            "BagelDiffusionStage._diffuse_batched: packed latent tokens must divide evenly by batch size.",
        )
        seq = int(x_t.shape[0]) // batch_size

        sampling_generators = self._sampling_generators(device, batch_size)
        self.strategy.init_schedule(schedule)
        needed = set(compute_trajectory_positions(sde_set, num_steps))
        needed.add(num_steps)
        stored_pairs: List[Tuple[int, torch.Tensor]] = []
        if 0 in needed:
            stored_pairs.append((0, x_t.detach().clone().reshape(batch_size, seq, channels)))
        sde_logps: List[torch.Tensor] = []
        sde_means: List[torch.Tensor] = []

        with torch.no_grad(), self._autocast_ctx(device):
            for index in range(num_steps):
                t_cur = schedule[index]
                t_next = schedule[index + 1]
                cfg_text_scale, cfg_img_scale = self._gated_cfg_scales(float(t_cur.item()), params)
                x_t, log_prob, prev_mean = self.step.step_with_logp(
                    bagel,
                    self.strategy,
                    x_t=x_t,
                    prev_sample=None,
                    t_cur=t_cur,
                    t_next=t_next,
                    sigma_max=sigma_max,
                    eta=float(params.eta) if index in sde_set else 0.0,
                    cfg_text_scale=cfg_text_scale,
                    cfg_img_scale=cfg_img_scale,
                    forward_kwargs=forward_kwargs,
                    n_samples=batch_size,
                    generators=sampling_generators,
                )
                x_t = x_t.to(dtype=self.trajectory_dtype)
                if (index + 1) in needed:
                    stored_pairs.append((index + 1, x_t.detach().clone().reshape(batch_size, seq, channels)))
                if log_prob is not None:
                    sde_logps.append(log_prob.to(dtype=self.logprob_dtype))
                    if prev_mean is not None:
                        sde_means.append(
                            prev_mean.detach().reshape(batch_size, seq, channels).to(dtype=self.trajectory_dtype)
                        )

        return LatentSegment(
            latents=torch.stack([tensor for _, tensor in stored_pairs], dim=1),
            sigmas=schedule,
            indices=torch.tensor([index for index, _ in stored_pairs], dtype=torch.long, device=device),
            sde_logp=torch.stack(sde_logps, dim=1) if sde_logps else None,
            sde_means=torch.stack(sde_means, dim=1) if sde_means else None,
            sde_indices=(torch.tensor(sde_sorted, dtype=torch.long, device=device) if sde_sorted else None),
        )

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------

    def replay(
        self,
        conditions: BagelDiffusionConditions,
        *,
        segment: LatentSegment,
        params: BagelDiffusionParams,
        step_indices: Optional[List[int]] = None,
    ) -> ReplayResult:
        """Recompute per-step log-probs over the SDE window (mirrors SD3 replay).

        Loops ``step.step_with_logp`` (``prev_sample`` = the stored next frame) over
        ``segment.sde_indices`` (or the ``step_indices`` subset). Returns a
        :class:`ReplayResult` with ``log_probs [1, S']`` aligned with
        ``segment.sde_logp`` plus ``prev_sample_means [1, S', seq, C]`` for KL.

        Caller owns ``.train()`` mode + grad scope; this method manages only the
        autocast scope (mirrors ``SD3DiffusionStage.replay``).
        """
        if segment.sde_indices is None or segment.latents is None or segment.sigmas is None:
            raise ValueError("BagelDiffusionStage.replay: segment.sde_indices / latents / sigmas missing")

        bagel = self.model.model
        device = torch.device(self.model.device)
        sde_set = set(int(i) for i in segment.sde_indices.tolist())
        target = [int(i) for i in step_indices] if step_indices is not None else sorted(sde_set)
        bad = [i for i in target if i not in sde_set]
        if bad:
            raise ValueError(
                f"BagelDiffusionStage.replay: step_indices {bad} not in segment.sde_indices={sorted(sde_set)}"
            )

        schedule = segment.sigmas.to(device)
        sigma_max = schedule[1] if int(schedule.shape[0]) > 1 else schedule[0]

        gen, cfg_text, cfg_img, image_shape = self._resolve_single(conditions)
        gi, gi_cfg_text, gi_cfg_img = self._build_generation_inputs(gen, cfg_text, cfg_img, image_shape, device=device)
        forward_kwargs = self._forward_kwargs(gen, cfg_text, cfg_img, gi, gi_cfg_text, gi_cfg_img, params)

        log_probs: List[torch.Tensor] = []
        prev_sample_means: List[torch.Tensor] = []
        with self._autocast_ctx(device):
            for step_idx in target:
                t_cur = schedule[step_idx]
                t_next = schedule[step_idx + 1]
                cfg_text_scale, cfg_img_scale = self._gated_cfg_scales(float(t_cur.item()), params)
                x_t = segment.latents_at(step_idx)[0].to(device)  # [seq, C]
                prev_sample = segment.latents_at(step_idx + 1)[0].to(device)
                _, log_prob, prev_mean = self.step.step_with_logp(
                    bagel,
                    self.strategy,
                    x_t=x_t,
                    prev_sample=prev_sample,
                    t_cur=t_cur,
                    t_next=t_next,
                    sigma_max=sigma_max,
                    eta=float(params.eta),
                    cfg_text_scale=cfg_text_scale,
                    cfg_img_scale=cfg_img_scale,
                    forward_kwargs=forward_kwargs,
                )
                if log_prob is None:
                    raise RuntimeError(
                        f"BagelDiffusionStage.replay: strategy returned None log-prob at step={step_idx} "
                        f"(deterministic mode); replay requires a stochastic SDE strategy (eta>0)."
                    )
                log_probs.append(log_prob)
                prev_sample_means.append(prev_mean)

        log_probs_t = torch.stack(log_probs, dim=0).unsqueeze(0).to(dtype=self.logprob_dtype)  # [1, S']
        means_t = torch.stack(prev_sample_means, dim=0).unsqueeze(0).to(dtype=self.trajectory_dtype)  # [1, S', seq, C]
        return ReplayResult(log_probs=log_probs_t, prev_sample_means=means_t)

    # ------------------------------------------------------------------
    # Single-step velocity (forward-process algorithms: DiffusionNFT et al.)
    # ------------------------------------------------------------------

    def build_forward_kwargs(
        self,
        conditions: BagelDiffusionConditions,
        *,
        params: BagelDiffusionParams,
        device: torch.device,
    ) -> Dict[str, Any]:
        """Rebuild the KV contexts → the step-invariant ``_forward_flow`` kwargs.

        The expensive part (resolving the per-sample gen / cfg contexts and packing
        the generation inputs) is done ONCE; a caller scoring several steps against
        the same conditioning then reuses the result via :meth:`predict_velocity_at`
        (no per-step rebuild). Used by
        :class:`~unirl.algorithms.bagel_flow_unigrpo.BagelFlowUniGRPO`'s velocity-MSE
        loop, which evaluates ``v_theta`` / ``v_ref`` across the SDE window.
        """
        gen, cfg_text, cfg_img, image_shape = self._resolve_single(conditions)
        gi, gi_cfg_text, gi_cfg_img = self._build_generation_inputs(gen, cfg_text, cfg_img, image_shape, device=device)
        return self._forward_kwargs(gen, cfg_text, cfg_img, gi, gi_cfg_text, gi_cfg_img, params)

    def predict_velocity_at(
        self,
        forward_kwargs: Dict[str, Any],
        *,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        params: BagelDiffusionParams,
    ) -> torch.Tensor:
        """Single ``(x_t, sigma)`` CFG velocity reusing prebuilt ``forward_kwargs``.

        Pairs with :meth:`build_forward_kwargs` so a caller scoring multiple steps
        against one conditioning pays the context prefill once. ``sample`` is packed
        ``[seq, C]`` (or ``[1, seq, C]``; the unit batch dim is squeezed). Grad flows
        through the velocity forward (gen experts) only — the contexts inside
        ``forward_kwargs`` are detached constants built under ``no_grad``.
        """
        bagel = self.model.model
        device = torch.device(self.model.device)
        t_val = float(sigma.item()) if isinstance(sigma, torch.Tensor) else float(sigma)
        cfg_text_scale, cfg_img_scale = self._gated_cfg_scales(t_val, params)
        sample = sample.to(device)
        if sample.dim() == 3:  # [1, seq, C] → [seq, C] (navit bs=1)
            sample = sample[0]
        with self._autocast_ctx(device):
            return self.step.predict_velocity(
                bagel,
                x_t=sample,
                t_cur=sigma,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                forward_kwargs=forward_kwargs,
            )

    def predict_noise_at_step(
        self,
        conditions: BagelDiffusionConditions,
        *,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        params: BagelDiffusionParams,
    ) -> torch.Tensor:
        """Single ``(x_t, sigma)`` velocity forward — no scheduler iteration.

        Completes the ``DiffusionStage`` protocol (used by forward-process algorithms
        like DiffusionNFT). Rebuilds the contexts (:meth:`build_forward_kwargs`) then
        runs the same navit ``_forward_flow`` call (:meth:`predict_velocity_at`) that
        ``diffuse`` / ``replay`` use, so CFG handling is identical. ``sample`` is packed
        ``[seq, C]`` (or ``[1, seq, C]``). Callers that score multiple steps (e.g. the
        velocity-MSE loop) should instead build the kwargs once via
        :meth:`build_forward_kwargs` and loop :meth:`predict_velocity_at`.
        """
        device = torch.device(self.model.device)
        forward_kwargs = self.build_forward_kwargs(conditions, params=params, device=device)
        return self.predict_velocity_at(forward_kwargs, sample=sample, sigma=sigma, params=params)

    # ------------------------------------------------------------------
    # Trainable surface for FSDPPolicy
    # ------------------------------------------------------------------

    def trainable_module(self) -> "torch.nn.Module":
        """The MoT transformer (``bundle.transformer`` == ``model.language_model``).

        This is the FSDP wrap target / LoRA injection root — the same module the
        vendored ``_forward_flow`` runs on, so sharding it shards the gen forward.
        """
        return self.model.transformer


__all__ = [
    "BagelDiffusionParams",
    "BagelDiffusionStage",
    "BagelDiffusionStep",
]
