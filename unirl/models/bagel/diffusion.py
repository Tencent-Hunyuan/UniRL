"""Bagel diffusion: typed params + per-step kernel + rollout-level stage."""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

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


def stack_latent_segments(segments: List[LatentSegment]) -> LatentSegment:
    """Concatenate per-sample latent rows while preserving shared trajectory metadata."""
    if not segments:
        raise ValueError("stack_latent_segments: segments must be non-empty.")
    if len(segments) == 1:
        return segments[0]
    return LatentSegment(
        latents=torch.cat([segment.latents for segment in segments]),
        sigmas=segments[0].sigmas,
        indices=segments[0].indices,
        sde_logp=(torch.cat([segment.sde_logp for segment in segments]) if segments[0].sde_logp is not None else None),
        sde_means=(
            torch.cat([segment.sde_means for segment in segments]) if segments[0].sde_means is not None else None
        ),
        sde_indices=segments[0].sde_indices,
    )


def _cat_optional_kv(
    tensors: List[Optional[torch.Tensor]],
    kv_lens: List[int],
    *,
    name: str,
) -> Optional[torch.Tensor]:
    """Concatenate per-sample cache tensors, treating ``None`` as an empty cache only."""
    present = [tensor for tensor in tensors if tensor is not None]
    if not present:
        if any(length > 0 for length in kv_lens):
            raise ValueError(f"BagelDiffusionStage._merge_contexts: {name} is missing for non-empty KV.")
        return None

    prototype = present[0]
    filled: List[torch.Tensor] = []
    for tensor, length in zip(tensors, kv_lens):
        if tensor is None:
            if length > 0:
                raise ValueError(f"BagelDiffusionStage._merge_contexts: {name} is None for kv_len={length}.")
            filled.append(prototype.new_empty((0, *prototype.shape[1:])))
        else:
            if int(tensor.shape[0]) != length:
                raise ValueError(
                    f"BagelDiffusionStage._merge_contexts: {name} rows {tensor.shape[0]} != kv_len={length}."
                )
            filled.append(tensor)
    return torch.cat(filled)


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
        n_samples: int = 1,
        generators: Optional[List[torch.Generator]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run one transition after restoring the logical sample axis."""
        require(n_samples >= 1, f"BagelDiffusionStep.denoise: n_samples must be >= 1; got {n_samples}.")
        require(
            int(x_t.shape[0]) % n_samples == 0,
            f"BagelDiffusionStep.denoise: packed token count {x_t.shape[0]} is not divisible by n_samples={n_samples}.",
        )
        if generators is not None and len(generators) != n_samples:
            raise ValueError(f"BagelDiffusionStep.denoise: got {len(generators)} generators for n_samples={n_samples}.")
        seq = int(x_t.shape[0]) // n_samples
        channels = int(x_t.shape[-1])
        prev, log_prob, prev_mean = strategy.denoise(
            noise_pred=v_t.reshape(n_samples, seq, channels),
            sample=x_t.reshape(n_samples, seq, channels),
            sigma=sigma,
            sigma_next=sigma_next,
            eta=float(eta),
            prev_sample=None if prev_sample is None else prev_sample.reshape(n_samples, seq, channels),
            generator=None if prev_sample is not None else generators,
            sigma_max=float(sigma_max),
        )
        if n_samples > 1:
            return (
                prev.reshape(n_samples * seq, channels),
                None if log_prob is None else log_prob.reshape(n_samples),
                None if prev_mean is None else prev_mean.reshape(n_samples * seq, channels),
            )
        return (
            prev.reshape(seq, channels),
            None if log_prob is None else log_prob.reshape(()),
            None if prev_mean is None else prev_mean.reshape(seq, channels),
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
            n_samples=n_samples,
            generators=generators,
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

    def _build_contexts_from_prompt(
        self,
        prompt: str,
        image: Optional[Any] = None,
        *,
        differentiable: bool = False,
    ) -> Tuple[Any, Any, Any]:
        """Rebuild the three KV contexts from raw prompt and optional source image."""
        if image is not None and getattr(self.model.model, "vit_model", None) is None:
            raise ValueError(
                "BagelDiffusionStage: the conditions carry an it2i source image but the bundle "
                "was built without the und ViT; set BagelPipelineConfig.enable_vit=true."
            )

        inf = self.model.inferencer
        device = torch.device(self.model.device)
        clean_prompt = str(prompt).removeprefix("<|im_start|>").removesuffix("<|im_end|>")
        gen = inf.init_gen_context()
        cfg_img = deepcopy(gen)
        # Preserve inference dispatch while rebuilding frozen or differentiable it2i contexts.
        grad_context = torch.enable_grad if differentiable else torch.no_grad
        with (
            rl_ops.inference_dispatch_scope(self.model.model),
            grad_context(),
            self._autocast_ctx(device),
        ):
            if image is not None:
                resized = rl_ops.resize_input_image(self.model, image)
                gen = rl_ops.update_context_image(
                    self.model,
                    resized,
                    gen,
                    vae=True,
                    vit=True,
                    differentiable=differentiable,
                )
            cfg_text = rl_ops.clone_context(gen) if differentiable else deepcopy(gen)
            if differentiable:
                gen = rl_ops.update_context_text(self.model, clean_prompt, gen, differentiable=True)
                cfg_img = rl_ops.update_context_text(self.model, clean_prompt, cfg_img, differentiable=True)
            else:
                gen = inf.update_context_text(clean_prompt, gen)
                cfg_img = inf.update_context_text(clean_prompt, cfg_img)
        return gen, cfg_text, cfg_img

    def _resolve_single(
        self,
        conditions: BagelDiffusionConditions,
        *,
        differentiable: bool = False,
        force_rebuild: bool = False,
    ) -> Tuple[Any, Any, Any, Tuple[int, int]]:
        """Return ``(gen, cfg_text, cfg_img, image_shape)`` for a 1-sample batch."""
        if force_rebuild:
            require(
                bool(conditions.input_images),
                "_resolve_single(force_rebuild=True) requires image-conditioned "
                "raw material; prompt-only stored contexts (t2i/t2ti) cannot be "
                "reproduced from the bare prompt.",
            )
        if differentiable and conditions.input_images:
            prompt, input_image, image_shape = conditions.single_prompt()
            gen, cfg_text, cfg_img = self._build_contexts_from_prompt(
                prompt,
                input_image,
                differentiable=True,
            )
            return gen, cfg_text, cfg_img, image_shape
        if conditions.has_contexts() and not force_rebuild:
            return conditions.single()
        prompt, input_image, image_shape = conditions.single_prompt()
        gen, cfg_text, cfg_img = self._build_contexts_from_prompt(
            prompt,
            input_image,
            differentiable=differentiable,
        )
        return gen, cfg_text, cfg_img, image_shape

    def _build_generation_inputs(
        self,
        gen: Any,
        cfg_text: Any,
        cfg_img: Any,
        image_shapes: List[Tuple[int, int]],
        *,
        device: torch.device,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Reconstruct the packed gen / cfg_text / cfg_img inputs from the contexts."""
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
            return float(params.guidance_scale), float(params.cfg_img_scale)
        return 1.0, 1.0

    def diffuse(
        self,
        conditions: BagelDiffusionConditions,
        *,
        schedule: torch.Tensor,
        params: BagelDiffusionParams,
        initial_latents: Optional[torch.Tensor] = None,
    ) -> LatentSegment:
        """Run Bagel sampling over the pinned schedule."""
        batch_size = conditions.batch_size
        require(batch_size > 0, "BagelDiffusionStage.diffuse: conditions must be non-empty.")
        if batch_size > 1:
            self._require_stackable_image_shapes(conditions)
            if self._global_cfg_needs_serial(params) or not self._can_pack_conditions(conditions):
                return self._diffuse_serial_batch(
                    conditions,
                    schedule=schedule,
                    params=params,
                    initial_latents=initial_latents,
                )

        bagel = self.model.model
        device = torch.device(self.model.device)
        schedule = schedule.to(device)
        num_steps = int(schedule.shape[0]) - 1
        require(
            num_steps == int(params.num_inference_steps),
            f"BagelDiffusionStage.diffuse: schedule length {schedule.shape[0]} != "
            f"num_inference_steps+1 ({int(params.num_inference_steps) + 1})",
        )
        sigma_max = schedule[1] if int(schedule.shape[0]) > 1 else schedule[0]

        sde_set = {int(index) for index in (params.sde_indices or [])}
        sde_sorted = sorted(sde_set)
        gen, cfg_text, cfg_img, image_shapes = self._resolve_rollout_batch(conditions)
        gi, gi_cfg_text, gi_cfg_img = self._build_generation_inputs(
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
            initial_latents = initial_latents.to(device=device, dtype=self.trajectory_dtype)
            x_t = initial_latents.reshape(-1, int(initial_latents.shape[-1]))
        channels = int(x_t.shape[-1])
        require(
            int(x_t.shape[0]) % batch_size == 0,
            "BagelDiffusionStage.diffuse: packed latent tokens must divide evenly by batch size.",
        )
        seq = int(x_t.shape[0]) // batch_size

        generators = (
            rl_ops.fork_sampling_generators(device, batch_size)
            if batch_size > 1 and sde_set and float(params.eta) >= 1e-7
            else None
        )
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
                    generators=generators,
                )
                x_t = x_t.to(dtype=self.trajectory_dtype)
                if (index + 1) in needed:
                    stored_pairs.append((index + 1, x_t.detach().clone().reshape(batch_size, seq, channels)))
                if log_prob is not None:
                    sde_logps.append(log_prob.reshape(batch_size).to(dtype=self.logprob_dtype))
                    if prev_mean is not None:
                        sde_means.append(prev_mean.detach().reshape(batch_size, seq, channels))

        require(
            len(sde_logps) == len(sde_sorted) and len(sde_means) == len(sde_sorted),
            "BagelDiffusionStage.diffuse: SDE records are misaligned with sde_indices.",
        )
        return LatentSegment(
            latents=torch.stack([tensor for _, tensor in stored_pairs], dim=1),
            sigmas=schedule,
            indices=torch.tensor([index for index, _ in stored_pairs], dtype=torch.long, device=device),
            sde_logp=torch.stack(sde_logps, dim=1) if sde_logps else None,
            sde_means=torch.stack(sde_means, dim=1) if sde_means else None,
            sde_indices=torch.tensor(sde_sorted, dtype=torch.long, device=device) if sde_sorted else None,
        )

    @staticmethod
    def _can_pack_conditions(conditions: BagelDiffusionConditions) -> bool:
        """Accept only complete opaque contexts with one shared image geometry."""
        batch_size = conditions.batch_size
        if not conditions.has_contexts() or batch_size < 2:
            return False
        if len(conditions.gen_contexts) != batch_size or any(ctx is None for ctx in conditions.gen_contexts):
            return False
        shapes = [tuple(shape) for shape in conditions.image_shapes]
        return len(shapes) == batch_size and len(set(shapes)) == 1

    @staticmethod
    def _global_cfg_needs_serial(params: BagelDiffusionParams) -> bool:
        """Keep vendor global CFG renorm on its required one-sample path."""
        return params.cfg_renorm_type == "global" and float(params.guidance_scale) > 1.0

    def _require_stackable_image_shapes(self, conditions: BagelDiffusionConditions) -> None:
        """Reject image batches whose latent rows cannot share one tensor."""
        shapes = [tuple(int(value) for value in shape) for shape in conditions.image_shapes]
        require(
            len(shapes) == conditions.batch_size,
            f"BagelDiffusionStage.diffuse: image shape count {len(shapes)} != batch size {conditions.batch_size}.",
        )
        downsample = int(self.model.latent_downsample)
        lengths = [(height // downsample) * (width // downsample) for height, width in shapes]
        require(
            len(set(lengths)) == 1,
            "BagelDiffusionStage.diffuse: mixed image shapes produce variable latent lengths "
            f"{lengths}; split them into separate calls.",
        )

    def _resolve_rollout_batch(
        self,
        conditions: BagelDiffusionConditions,
    ) -> Tuple[Any, Any, Any, List[Tuple[int, int]]]:
        """Resolve one sample directly or merge a compatible packed batch."""
        batch_size = conditions.batch_size
        if batch_size == 1:
            gen, cfg_text, cfg_img, image_shape = self._resolve_single(conditions)
            return gen, cfg_text, cfg_img, [image_shape]

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
        return (
            self._merge_contexts(gen_contexts),
            self._merge_contexts(cfg_text_contexts),
            self._merge_contexts(cfg_img_contexts),
            [tuple(shape) for shape in conditions.image_shapes],
        )

    def _diffuse_serial_batch(
        self,
        conditions: BagelDiffusionConditions,
        *,
        schedule: torch.Tensor,
        params: BagelDiffusionParams,
        initial_latents: Optional[torch.Tensor],
    ) -> LatentSegment:
        """Fallback for deferred contexts, global CFG renorm, or unpackable inputs."""
        segments: List[LatentSegment] = []
        for index in range(conditions.batch_size):
            if conditions.has_contexts():
                gen = conditions.gen_contexts[index]
                condition = BagelDiffusionConditions.for_sample(
                    gen_context=gen,
                    cfg_text_context=(
                        conditions.cfg_text_contexts[index]
                        if conditions.cfg_text_contexts and conditions.cfg_text_contexts[index] is not None
                        else gen
                    ),
                    cfg_img_context=(
                        conditions.cfg_img_contexts[index]
                        if conditions.cfg_img_contexts and conditions.cfg_img_contexts[index] is not None
                        else gen
                    ),
                    prompt=conditions.prompts[index] if conditions.prompts else None,
                    image_shape=tuple(conditions.image_shapes[index]),
                )
            else:
                condition = BagelDiffusionConditions(
                    prompts=[conditions.prompts[index]],
                    input_images=[conditions.input_images[index]] if conditions.input_images else [],
                    image_shapes=[tuple(conditions.image_shapes[index])],
                )
            initial = initial_latents[index] if initial_latents is not None else None
            segments.append(self.diffuse(condition, schedule=schedule, params=params, initial_latents=initial))
        return stack_latent_segments(segments)

    @staticmethod
    def _merge_contexts(contexts: List[Any]) -> Dict[str, Any]:
        """Merge per-sample NaiveCache objects into one packed cache."""
        kv_lens = [int(context["kv_lens"][0]) for context in contexts]
        ropes = [int(context["ropes"][0]) for context in contexts]
        caches = [context["past_key_values"] for context in contexts]
        num_layers = int(caches[0].num_layers)
        if any(int(cache.num_layers) != num_layers for cache in caches):
            raise ValueError("BagelDiffusionStage._merge_contexts: cache layer counts differ.")
        merged = type(caches[0])(num_layers)
        for layer in range(num_layers):
            merged.key_cache[layer] = _cat_optional_kv(
                [cache.key_cache[layer] for cache in caches],
                kv_lens,
                name=f"key_cache[{layer}]",
            )
            merged.value_cache[layer] = _cat_optional_kv(
                [cache.value_cache[layer] for cache in caches],
                kv_lens,
                name=f"value_cache[{layer}]",
            )
        return {"kv_lens": kv_lens, "ropes": ropes, "past_key_values": merged}

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

        gen, cfg_text, cfg_img, image_shape = self._resolve_single(
            conditions,
            differentiable=torch.is_grad_enabled(),
        )
        gi, gi_cfg_text, gi_cfg_img = self._build_generation_inputs(
            gen,
            cfg_text,
            cfg_img,
            [image_shape],
            device=device,
        )
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
        means_t = torch.stack(prev_sample_means, dim=0).unsqueeze(0)
        return ReplayResult(log_probs=log_probs_t, prev_sample_means=means_t)

    def build_forward_kwargs(
        self,
        conditions: BagelDiffusionConditions,
        *,
        params: BagelDiffusionParams,
        device: torch.device,
        force_rebuild: bool = False,
    ) -> Dict[str, Any]:
        """Rebuild the KV contexts → the step-invariant ``_forward_flow`` kwargs."""
        gen, cfg_text, cfg_img, image_shape = self._resolve_single(
            conditions,
            differentiable=torch.is_grad_enabled(),
            force_rebuild=force_rebuild,
        )
        gi, gi_cfg_text, gi_cfg_img = self._build_generation_inputs(
            gen,
            cfg_text,
            cfg_img,
            [image_shape],
            device=device,
        )
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
    "stack_latent_segments",
]
