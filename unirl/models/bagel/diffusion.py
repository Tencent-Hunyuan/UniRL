"""Bagel diffusion: typed params + per-step kernel + rollout-level stage."""

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
    """Bagel diffusion knobs — one object for BOTH the trainer and the stage."""

    num_inference_steps: int = 14
    guidance_scale: float = 1.0
    height: int = 512
    width: int = 512
    eta: float = 1.0

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


_to_device = rl_ops._to_device


class BagelDiffusionStep:
    """Per-step Bagel kernel — stateless navit adapter over the shared SDE strategy."""

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
        """CFG-combined velocity ``v_t`` for packed ``x_t`` ``[seq, C]`` at time ``t_cur``."""
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """One SDE transition over packed ``[seq, C]`` latents — a unit batch dim is added and squeezed back."""
        prev, log_prob, prev_mean = strategy.denoise(
            noise_pred=v_t.unsqueeze(0),
            sample=x_t.unsqueeze(0),
            sigma=sigma,
            sigma_next=sigma_next,
            eta=float(eta),
            prev_sample=None if prev_sample is None else prev_sample.unsqueeze(0),
            sigma_max=float(sigma_max),
        )
        return (
            prev.squeeze(0),
            None if log_prob is None else log_prob.reshape(()),
            None if prev_mean is None else prev_mean.squeeze(0),
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run ``predict_velocity`` then ``denoise`` for one step."""
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
        )


class BagelDiffusionStage(DiffusionStage[BagelDiffusionConditions]):
    """Bagel rollout-level diffusion stage (trainside A1) — central-runtime, SD3-shaped."""

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
        self.model = model
        self.step = step if step is not None else BagelDiffusionStep()
        self.strategy = strategy if strategy is not None else FlowSDEStrategy()
        self.autocast_dtype = parse_torch_dtype(autocast_precision, field_name="autocast_precision")
        self.trajectory_dtype = parse_torch_dtype(trajectory_precision, field_name="trajectory_precision")
        self.logprob_dtype = parse_torch_dtype(logprob_precision, field_name="logprob_precision")

    def _autocast_ctx(self, device: torch.device):
        if device.type == "cuda" and self.autocast_dtype in (torch.float16, torch.bfloat16):
            return torch.autocast("cuda", self.autocast_dtype)
        return nullcontext()

    def _build_contexts_from_prompt(self, prompt: str) -> Tuple[Any, Any, Any]:
        """Rebuild the three KV contexts (gen / cfg_text / cfg_img) from a prompt."""
        from copy import deepcopy

        inf = self.model.inferencer
        device = torch.device(self.model.device)
        clean_prompt = str(prompt).removeprefix("<|im_start|>").removesuffix("<|im_end|>")
        gen = inf.init_gen_context()
        cfg_img = deepcopy(gen)
        mot = self.model.transformer
        was_training = mot.training
        mot.eval()
        try:
            with torch.no_grad(), self._autocast_ctx(device):
                cfg_text = deepcopy(gen)
                gen = inf.update_context_text(clean_prompt, gen)
                cfg_img = inf.update_context_text(clean_prompt, cfg_img)
        finally:
            mot.train(was_training)
        return gen, cfg_text, cfg_img

    def _resolve_single(self, conditions: BagelDiffusionConditions) -> Tuple[Any, Any, Any, Tuple[int, int]]:
        """Return ``(gen, cfg_text, cfg_img, image_shape)`` for a 1-sample batch."""
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
        """Reconstruct the packed gen / cfg_text / cfg_img inputs from the contexts."""
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
        """Static (step-invariant) kwargs for ``_forward_flow``."""
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

    def diffuse(
        self,
        conditions: BagelDiffusionConditions,
        *,
        schedule: torch.Tensor,
        params: BagelDiffusionParams,
        initial_latents: Optional[torch.Tensor] = None,
    ) -> LatentSegment:
        """Run Bagel sampling over the pinned schedule; ``latents [1, K, seq, C]``, ``sde_logp [1, S]`` (navit bs=1)."""
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

        needed: Set[int] = set(compute_trajectory_positions(sde_set, T))
        needed.add(T)
        stored_pairs: List[Tuple[int, torch.Tensor]] = []
        if 0 in needed:
            stored_pairs.append((0, x_t.detach().clone()))
        sde_logp_list: List[torch.Tensor] = []
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
                    if prev_mean is not None:
                        sde_means_list.append(prev_mean.detach().to(dtype=self.trajectory_dtype))

        positions_collected = [p for p, _ in stored_pairs]
        latents_stacked = torch.stack([t for _, t in stored_pairs], dim=0).unsqueeze(0)
        sde_logp = torch.stack(sde_logp_list, dim=0).unsqueeze(0) if sde_logp_list else None
        sde_means = torch.stack(sde_means_list, dim=0).unsqueeze(0) if sde_means_list else None
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

    def replay(
        self,
        conditions: BagelDiffusionConditions,
        *,
        segment: LatentSegment,
        params: BagelDiffusionParams,
        step_indices: Optional[List[int]] = None,
    ) -> ReplayResult:
        """Recompute log-probs over the SDE window: ``log_probs [1, S']``, ``prev_sample_means [1, S', seq, C]``."""
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
                x_t = segment.latents_at(step_idx)[0].to(device)
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

        log_probs_t = torch.stack(log_probs, dim=0).unsqueeze(0).to(dtype=self.logprob_dtype)
        means_t = torch.stack(prev_sample_means, dim=0).unsqueeze(0).to(dtype=self.trajectory_dtype)
        return ReplayResult(log_probs=log_probs_t, prev_sample_means=means_t)

    def build_forward_kwargs(
        self,
        conditions: BagelDiffusionConditions,
        *,
        params: BagelDiffusionParams,
        device: torch.device,
    ) -> Dict[str, Any]:
        """Rebuild the KV contexts → the step-invariant ``_forward_flow`` kwargs."""
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
        """Single ``(x_t, sigma)`` CFG velocity reusing prebuilt ``forward_kwargs``."""
        bagel = self.model.model
        device = torch.device(self.model.device)
        t_val = float(sigma.item()) if isinstance(sigma, torch.Tensor) else float(sigma)
        cfg_text_scale, cfg_img_scale = self._gated_cfg_scales(t_val, params)
        sample = sample.to(device)
        if sample.dim() == 3:
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
        """Single ``(x_t, sigma)`` velocity forward — no scheduler iteration."""
        device = torch.device(self.model.device)
        forward_kwargs = self.build_forward_kwargs(conditions, params=params, device=device)
        return self.predict_velocity_at(forward_kwargs, sample=sample, sigma=sigma, params=params)

    def trainable_module(self) -> "torch.nn.Module":
        """The MoT transformer (``bundle.transformer`` == ``model.language_model``)."""
        return self.model.transformer


__all__ = [
    "BagelDiffusionParams",
    "BagelDiffusionStage",
    "BagelDiffusionStep",
]
