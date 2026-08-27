"""HunyuanImage3 diffusion: typed params + per-step kernel + rollout-level stage."""

from __future__ import annotations

import logging
import sys
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from unirl.models.types.diffusion import DiffusionStage, DiffusionStep
from unirl.models.types.replay_result import ReplayResult
from unirl.sde.kernels import GeneratorLike, StepStrategy
from unirl.types.conditions import ImageEmbedCondition
from unirl.types.noise_recipe import NoiseRecipe
from unirl.types.sampling import DiffusionSamplingParams, compute_trajectory_positions
from unirl.types.segments.latent import LatentSegment
from unirl.utils.dtypes import parse_torch_dtype

from .bundle import HunyuanImage3Bundle
from .conditions import HunyuanImage3DiffusionConditions, HunyuanImage3FusedMultimodalCondition
from .diffusion_state import HunyuanImage3DiffusionState
from .seed import make_sde_step_generators

logger = logging.getLogger(__name__)


class HunyuanImage3DiffusionStep(DiffusionStep[HunyuanImage3Bundle, HunyuanImage3DiffusionConditions]):
    """Per-step HunyuanImage3 denoising kernel — stateless."""

    def predict_noise(
        self,
        model: HunyuanImage3Bundle,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        conditions: HunyuanImage3DiffusionConditions,
        *,
        guidance_scale: float,
        state: Optional[HunyuanImage3DiffusionState] = None,
        step_index: int = 0,
    ) -> torch.Tensor:
        """Run the unified MM transformer in ``mode="gen_image"`` and return the CFG-combined noise prediction."""
        fused = conditions.fused
        if fused is None or fused.input_ids is None:
            raise ValueError(
                "HunyuanImage3DiffusionStep.predict_noise: "
                "conditions.fused.input_ids is None. The smoke t2i path expects the "
                "pipeline to populate conditions via "
                "HunyuanImage3TextEmbedStage.embed_for_gen_image(...)."
            )
        if fused.gen_image_mask is None or fused.gen_timestep_scatter_index is None:
            raise ValueError(
                "HunyuanImage3DiffusionStep.predict_noise: "
                "conditions.fused.gen_image_mask / gen_timestep_scatter_index missing — "
                "did you call HunyuanImage3TextEmbedStage.embed_for_gen_image(...)?"
            )

        transformer = model.transformer

        n_fused = int(fused.input_ids.shape[0])
        n_sample = int(sample.shape[0])
        if n_fused == n_sample:
            cfg = False
            sample_2 = sample
        elif n_fused == 2 * n_sample:
            cfg = True
            sample_2 = torch.cat([sample, sample], dim=0)
        else:
            raise ValueError(
                f"HunyuanImage3DiffusionStep.predict_noise: "
                f"fused.input_ids batch ({n_fused}) is neither equal to nor "
                f"2x of sample batch ({n_sample}). Unexpected capture shape."
            )
        t_scalar = sigma * 1000.0
        if t_scalar.numel() == 1:
            # Accept scalar-like [1] sigmas for backward compatibility.
            t_expand = t_scalar.reshape(()).expand(sample_2.shape[0])
        elif t_scalar.numel() == sample_2.shape[0]:
            t_expand = t_scalar.reshape(sample_2.shape[0])
        elif cfg and t_scalar.numel() == n_sample:
            # Preserve blockwise [cond; uncond] CFG ordering.
            t_expand = torch.cat([t_scalar.reshape(n_sample), t_scalar.reshape(n_sample)])
        else:
            raise ValueError(
                "HunyuanImage3DiffusionStep.predict_noise: sigma must be scalar, "
                f"[B], or [2B] under CFG; got shape={tuple(sigma.shape)}, "
                f"sample batch={n_sample}, model batch={int(sample_2.shape[0])}."
            )

        use_cache: bool = state is not None
        is_first: bool = state is None or step_index == 0

        if not use_cache or is_first:
            attention_mask_in = fused.attention_mask
            position_ids_in = fused.position_ids
            scatter_idx_in = fused.gen_timestep_scatter_index
            past_kv_in = None
        else:
            assert state is not None
            attention_mask_in = state.attention_mask
            position_ids_in = state.position_ids
            scatter_idx_in = state.gen_timestep_scatter_index
            past_kv_in = state.past_key_values

        if use_cache and is_first:
            past_kv_in = self._build_kv_cache(transformer, conditions)

        if is_first:
            cond_vae = conditions.cond_vae
            cond_vit = conditions.cond_vit
            cond_vae_images = cond_vae.latents if cond_vae is not None else None
            cond_vit_images = cond_vit.embeds if cond_vit is not None else None
            vit_kwargs: Optional[Dict[str, Any]] = None
            if cond_vit is not None and (cond_vit.spatial_shapes is not None or cond_vit.attn_mask is not None):
                vit_kwargs = {
                    "spatial_shapes": cond_vit.spatial_shapes,
                    "attention_mask": cond_vit.attn_mask,
                }
            cond_timestep = conditions.cond_timestep
            cond_vae_image_mask = fused.cond_vae_image_mask
            cond_vit_image_mask = fused.cond_vit_image_mask
            # Pass the cond-image timestep index or the continuous token is skipped.
            cond_timesteps_index = fused.cond_timestep_scatter_index
            # Move condition payloads to the latent device before projection.
            _dev = sample.device

            def _to_dev(x: Any) -> Any:
                if isinstance(x, torch.Tensor):
                    return x.to(_dev)
                if isinstance(x, list):
                    return [_to_dev(v) for v in x]
                if isinstance(x, dict):
                    return {k: _to_dev(v) for k, v in x.items()}
                return x

            cond_vae_images = _to_dev(cond_vae_images)
            cond_vit_images = _to_dev(cond_vit_images)
            cond_timestep = _to_dev(cond_timestep)
            cond_timesteps_index = _to_dev(cond_timesteps_index)
            cond_vae_image_mask = _to_dev(cond_vae_image_mask)
            cond_vit_image_mask = _to_dev(cond_vit_image_mask)
            vit_kwargs = _to_dev(vit_kwargs)
        else:
            cond_vae_images = None
            cond_timestep = None
            cond_timesteps_index = None  # Decode steps reuse the cached condition block.
            cond_vae_image_mask = None
            cond_vit_images = None
            cond_vit_image_mask = None
            vit_kwargs = None

        input_ids_in = fused.input_ids
        image_mask_in = fused.gen_image_mask
        if input_ids_in is not None and position_ids_in is not None:
            if input_ids_in.shape[1] != position_ids_in.shape[1]:
                input_ids_in = torch.gather(input_ids_in, dim=1, index=position_ids_in)
                if image_mask_in is not None:
                    image_mask_in = torch.gather(image_mask_in, dim=1, index=position_ids_in)
        # Seed CachedRoPE from rollout state to preserve sampling/replay parity.
        if fused.rope_cache is not None:
            # Set only the wrapper flag; recursive train() would affect frozen encoders.
            transformer.training = True
            _cr = transformer.cached_rope
            _rope = fused.rope_cache
            _cr.cos_cache = _rope[:, 0].to(device=input_ids_in.device)
            _cr.sin_cache = _rope[:, 1].to(device=input_ids_in.device)
            # Match input length so CachedRoPE bypasses rebuilding.
            _cr.seq_len = int(input_ids_in.shape[1])
            _cr.rope_image_info = None
            rope_image_info_val: Optional[List[List[Any]]] = None
        else:
            # Rebuild 2-D RoPE from masks when engine rollouts cannot share rope tables.
            _B = int(fused.input_ids.shape[0])
            rope_image_info_val = [[] for _ in range(_B)]
            if fused.gen_image_mask is not None:
                _h_lat = int(sample.shape[-2])
                _w_lat = int(sample.shape[-1])
                _gm = fused.gen_image_mask
                for _b in range(_B):
                    _idx = _gm[_b].nonzero(as_tuple=False).flatten()
                    if _idx.numel() == 0:
                        continue
                    _start = int(_idx[0].item())
                    _n = int(_idx.numel())
                    _contig = (int(_idx[-1].item()) - _start + 1) == _n
                    if not _contig or _w_lat <= 0 or _h_lat <= 0:
                        continue
                    _tw = int(round((_n * _w_lat / _h_lat) ** 0.5))
                    _th = _n // _tw if _tw > 0 else 0
                    if _tw > 0 and _th * _tw == _n:
                        rope_image_info_val[_b] = [(slice(_start, _start + _n), (_th, _tw))]
        # Decode uses a local timestep index; full-sequence indices can overflow.
        timesteps_index_in = scatter_idx_in if is_first else torch.zeros_like(scatter_idx_in)
        model_inputs = {
            "input_ids": input_ids_in,
            "attention_mask": attention_mask_in,
            "position_ids": position_ids_in,
            "past_key_values": past_kv_in,
            "rope_image_info": rope_image_info_val,
            "mode": "gen_image",
            "images": sample_2,
            "image_mask": image_mask_in,
            "timesteps": t_expand,
            "timesteps_index": timesteps_index_in,
            "gen_timestep_scatter_index": scatter_idx_in,
            "cond_vae_images": cond_vae_images,
            "cond_vae_image_mask": cond_vae_image_mask,
            "cond_timesteps": cond_timestep,
            "cond_timesteps_index": cond_timesteps_index,
            "cond_vit_images": cond_vit_images,
            "cond_vit_image_mask": cond_vit_image_mask,
            "cond_vit_image_kwargs": vit_kwargs,
        }
        _orig_check = getattr(transformer, "_check_inputs", None)
        transformer._check_inputs = lambda *a, **kw: None

        transformer.post_token_len = None
        n_img = int(fused.gen_image_mask.sum(dim=-1).max().item()) if fused.gen_image_mask is not None else 0
        transformer.num_image_tokens = n_img
        transformer.num_special_tokens = None if is_first else (int(input_ids_in.shape[1]) - n_img)

        output = transformer(**model_inputs, first_step=is_first)

        if _orig_check is not None:
            transformer._check_inputs = _orig_check

        if state is not None:
            self._update_state(transformer, output, conditions, state, is_first=is_first)

        if isinstance(output, dict):
            pred = output["diffusion_prediction"]
        else:
            pred = output.diffusion_prediction
        pred = pred.to(dtype=sample.dtype)

        if cfg:
            N_half = pred.shape[0] // 2
            pred_cond = pred[:N_half].contiguous()
            pred_uncond = pred[N_half:].contiguous()
            result = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
        else:
            result = pred

        return result

    @staticmethod
    def _build_kv_cache(transformer, conditions: HunyuanImage3DiffusionConditions):
        """Build a ``HunyuanStaticCache`` sized for the full sequence."""
        fused = conditions.fused
        if fused is None or fused.input_ids is None:
            return None
        upstream_mod = sys.modules[type(transformer).__module__]
        cache_cls = getattr(upstream_mod, "HunyuanStaticCache", None)
        if cache_cls is None:
            return None
        max_cache_len = int(fused.input_ids.shape[1])
        batch_size = int(fused.input_ids.shape[0])
        return cache_cls(
            config=transformer.config,
            batch_size=batch_size,
            max_cache_len=max_cache_len,
            dtype=torch.bfloat16,
            dynamic=False,
        )

    @staticmethod
    def _update_state(
        transformer,
        output,
        conditions: HunyuanImage3DiffusionConditions,
        state: HunyuanImage3DiffusionState,
        *,
        is_first: bool,
    ) -> None:
        """Carry past_key_values + the gathered position/attention tensors"""
        fused = conditions.fused
        assert fused is not None
        _rope_info = [[] for _ in range(int(fused.input_ids.shape[0]))]
        # Unbind stacked RoPE at the model boundary.
        _cpe = None if fused.rope_cache is None else (fused.rope_cache[:, 0], fused.rope_cache[:, 1])
        if is_first:
            mk: Dict[str, Any] = {
                "mode": "gen_image",
                "attention_mask": fused.attention_mask,
                "position_ids": fused.position_ids,
                "image_mask": fused.gen_image_mask,
                "gen_timestep_scatter_index": fused.gen_timestep_scatter_index,
                "timesteps_index": fused.gen_timestep_scatter_index,
                "custom_pos_emb": _cpe,
                "rope_image_info": _rope_info,
            }
            if conditions.tokenizer_output is not None:
                mk["tokenizer_output"] = conditions.tokenizer_output
        else:
            mk = {
                "mode": "gen_image",
                "attention_mask": state.attention_mask,
                "position_ids": state.position_ids,
                "image_mask": fused.gen_image_mask,
                "gen_timestep_scatter_index": state.gen_timestep_scatter_index,
                "timesteps_index": state.gen_timestep_scatter_index,
                "custom_pos_emb": _cpe,
                "rope_image_info": _rope_info,
            }
        updated = transformer._update_model_kwargs_for_generation(output, mk)
        state.past_key_values = updated.get("past_key_values")
        if updated.get("position_ids") is not None:
            state.position_ids = updated["position_ids"]
        if updated.get("attention_mask") is not None:
            state.attention_mask = updated["attention_mask"]
        if updated.get("gen_timestep_scatter_index") is not None:
            state.gen_timestep_scatter_index = updated["gen_timestep_scatter_index"]

    def forward(
        self,
        *,
        strategy: StepStrategy,
        noise_pred: torch.Tensor,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        prev_sample: Optional[torch.Tensor] = None,
        sigma_max: float = 0.99,
        eta: float = 1.0,
        step_index: int = 0,
        generator: GeneratorLike = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        return strategy.denoise(
            noise_pred=noise_pred,
            sample=sample,
            sigma=sigma,
            sigma_next=sigma_next,
            eta=eta,
            prev_sample=prev_sample,
            generator=generator,
            sigma_max=sigma_max,
            step_index=step_index,
        )

    def step(
        self,
        model: HunyuanImage3Bundle,
        conditions: HunyuanImage3DiffusionConditions,
        *,
        strategy: StepStrategy,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        guidance_scale: float,
        prev_sample: Optional[torch.Tensor] = None,
        sigma_max: float = 0.99,
        eta: float = 1.0,
        step_index: int = 0,
        state: Optional[HunyuanImage3DiffusionState] = None,
        generator: GeneratorLike = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        noise_pred = self.predict_noise(
            model,
            sample,
            sigma,
            conditions,
            guidance_scale=guidance_scale,
            state=state,
            step_index=step_index,
        )
        return self.forward(
            strategy=strategy,
            noise_pred=noise_pred,
            sample=sample,
            sigma=sigma,
            sigma_next=sigma_next,
            prev_sample=prev_sample,
            sigma_max=sigma_max,
            eta=eta,
            step_index=step_index,
            generator=generator,
        )

    def step_with_logp(
        self,
        model: HunyuanImage3Bundle,
        conditions: HunyuanImage3DiffusionConditions,
        *,
        strategy: StepStrategy,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        guidance_scale: float,
        prev_sample: Optional[torch.Tensor] = None,
        sigma_max: float = 0.99,
        eta: float = 1.0,
        step_index: int = 0,
        state: Optional[HunyuanImage3DiffusionState] = None,
        generator: GeneratorLike = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self.step(
            model,
            conditions,
            strategy=strategy,
            sample=sample,
            sigma=sigma,
            sigma_next=sigma_next,
            guidance_scale=guidance_scale,
            prev_sample=prev_sample,
            sigma_max=sigma_max,
            eta=eta,
            step_index=step_index,
            state=state,
            generator=generator,
        )


def _conditions_device_and_batch(
    conditions: HunyuanImage3DiffusionConditions,
    *,
    guidance_scale: float,
) -> Tuple[torch.device, int]:
    """Resolve ``(device, batch_size)`` from the conditions container."""
    fused = conditions.fused
    if fused is None or fused.input_ids is None:
        raise ValueError(
            "HunyuanImage3DiffusionStage: conditions.fused.input_ids is None; cannot infer device / batch."
        )
    n = int(fused.input_ids.shape[0])
    # Derive logical batch size before restacking CFG branches.
    if conditions.fused_uncond is not None:
        return fused.input_ids.device, n
    cfg = 2 if guidance_scale > 1.0 else 1
    if n % cfg != 0:
        raise ValueError(
            f"HunyuanImage3DiffusionStage: input_ids batch ({n}) is not a "
            f"multiple of CFG factor ({cfg}). Did the pipeline forget to "
            f"build CFG-batched inputs?"
        )
    return fused.input_ids.device, n // cfg


def _expand_cfg_for_forward(conditions: HunyuanImage3DiffusionConditions) -> HunyuanImage3DiffusionConditions:
    """Re-stack the B-batched cond/uncond fused into the guided ``[cond; uncond]``"""
    fused_uncond = conditions.fused_uncond
    if fused_uncond is None:
        return conditions

    def _dup2(x: Any) -> Any:
        # Repeat per-sample payloads as [cond; uncond].
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return torch.cat([x, x], dim=0)
        if isinstance(x, list):
            return list(x) + list(x)
        return x

    fused_cfg = HunyuanImage3FusedMultimodalCondition.concat([conditions.fused, fused_uncond])
    cond_vae = conditions.cond_vae
    if cond_vae is not None:
        cond_vae = type(cond_vae)(latents=_dup2(cond_vae.latents))
    cond_vit = conditions.cond_vit
    if cond_vit is not None:
        cond_vit = ImageEmbedCondition(
            embeds=_dup2(cond_vit.embeds),
            attn_mask=_dup2(cond_vit.attn_mask),
            spatial_shapes=_dup2(cond_vit.spatial_shapes),
        )
    return HunyuanImage3DiffusionConditions(
        fused=fused_cfg,
        fused_uncond=None,
        cond_vae=cond_vae,
        cond_vit=cond_vit,
        cond_timestep=_dup2(conditions.cond_timestep),
        tokenizer_output=conditions.tokenizer_output,
    )


class HunyuanImage3DiffusionStage(DiffusionStage[HunyuanImage3DiffusionConditions]):
    """HunyuanImage3 rollout-level diffusion stage."""

    def __init__(
        self,
        *,
        model: HunyuanImage3Bundle,
        step: HunyuanImage3DiffusionStep,
        strategy: StepStrategy,
        autocast_precision: str = "bf16",
        trajectory_precision: str = "fp16",
        logprob_precision: str = "fp32",
        vae_scale_factor: int = 16,
        latent_channels: int = 32,
        diffuse_kv_cache: bool = True,
    ) -> None:
        self.model = model
        self.step = step
        self.strategy = strategy
        self.autocast_dtype = parse_torch_dtype(autocast_precision, field_name="autocast_precision")
        self.trajectory_dtype = parse_torch_dtype(trajectory_precision, field_name="trajectory_precision")
        self.logprob_dtype = parse_torch_dtype(logprob_precision, field_name="logprob_precision")
        self.vae_scale_factor = vae_scale_factor
        self.latent_channels = latent_channels
        self.diffuse_kv_cache = bool(diffuse_kv_cache)

    def diffuse(
        self,
        conditions: HunyuanImage3DiffusionConditions,
        *,
        schedule: torch.Tensor,
        params: DiffusionSamplingParams,
        sde_sample_keys: Optional[List[str]] = None,
    ) -> LatentSegment:
        """Run full HunyuanImage3 DiT sampling. Returns a ``LatentSegment``."""
        from unirl.sde.noise import generate_latents

        device, batch_size = _conditions_device_and_batch(conditions, guidance_scale=float(params.guidance_scale))
        # Restack CFG branches after sizing initial noise.
        conditions = _expand_cfg_for_forward(conditions)
        T = int(params.num_inference_steps)
        if int(schedule.shape[0]) != T + 1:
            raise ValueError(f"HunyuanImage3DiffusionStage.diffuse: schedule length {schedule.shape[0]} != T+1={T + 1}")
        schedule = schedule.to(device)
        self.strategy.init_schedule(schedule)

        ip = getattr(self.model.transformer, "image_processor", None)
        info = None
        if ip is not None:
            if hasattr(ip, "build_image_info"):
                info = ip.build_image_info(f"{int(params.height)}x{int(params.width)}")
            elif hasattr(ip, "build_gen_image_info"):
                info = ip.build_gen_image_info(f"{int(params.height)}x{int(params.width)}")
        if info is not None:
            latent_h, latent_w = int(info.token_height), int(info.token_width)
            snapped_hw = (latent_h * int(self.vae_scale_factor), latent_w * int(self.vae_scale_factor))
            if snapped_hw != (int(params.height), int(params.width)):
                logger.warning(
                    "HunyuanImage3DiffusionStage.diffuse: requested %dx%d is not a supported HI3 preset; "
                    "snapped to %dx%d (nearest preset at ~%dpx base area).",
                    int(params.height),
                    int(params.width),
                    snapped_hw[0],
                    snapped_hw[1],
                    int(self.vae_scale_factor) * latent_h,
                )
        else:
            latent_h = int(params.height) // int(self.vae_scale_factor)
            latent_w = int(params.width) // int(self.vae_scale_factor)
        per_sample_shape = (int(self.latent_channels), latent_h, latent_w)
        latents = None
        if params.noise_group_ids:
            latents = (
                NoiseRecipe(
                    noise_group_ids=[str(g) for g in params.noise_group_ids],
                    base_seed=int(params.seed),
                )
                .for_batch(batch_size, latent_shape=per_sample_shape)
                .resolve(device=device, dtype=self.trajectory_dtype)
            )
        if latents is None:
            latents = generate_latents(
                batch_size=batch_size,
                latent_shape=per_sample_shape,
                device=device,
                dtype=self.trajectory_dtype,
                init_same_noise=bool(params.init_same_noise),
                samples_per_prompt=int(params.samples_per_prompt),
                noise_group_ids=params.noise_group_ids,
                base_seed=params.seed,
            )

        sde_set: Set[int] = set(int(i) for i in (params.sde_indices or []))
        sde_sorted: List[int] = sorted(sde_set)

        needed: Set[int] = set(compute_trajectory_positions(sde_set, T))
        needed.add(T)

        stored_pairs: List[Tuple[int, torch.Tensor]] = []
        if 0 in needed:
            stored_pairs.append((0, latents.detach().clone()))
        sde_logp_list: List[torch.Tensor] = []

        autocast_ctx = (
            torch.autocast("cuda", self.autocast_dtype)
            if device.type == "cuda" and self.autocast_dtype in (torch.float16, torch.bfloat16)
            else nullcontext()
        )
        sigma_max = float(schedule[1].item()) if int(schedule.shape[0]) > 1 else 0.99

        # Disable KV caching when rollout log-probs must match full-prefill replay.
        state = HunyuanImage3DiffusionState() if self.diffuse_kv_cache else None

        for i in range(T):
            sigma = schedule[i].to(device)
            sigma_next = schedule[i + 1].to(device)
            step_eta = float(params.eta) if i in sde_set else 0.0
            noise_generators = None
            if step_eta > 0.0 and params.seed is not None and sde_sample_keys is not None:
                if len(sde_sample_keys) != batch_size:
                    raise ValueError(
                        "HunyuanImage3DiffusionStage.diffuse: sde_sample_keys must align "
                        f"with batch_size={batch_size}, got {len(sde_sample_keys)}."
                    )
                noise_generators = make_sde_step_generators(
                    int(params.seed),
                    sde_sample_keys,
                    i,
                )

            with torch.no_grad(), autocast_ctx:
                new_latents, log_prob, _ = self.step.step_with_logp(
                    self.model,
                    conditions,
                    strategy=self.strategy,
                    sample=latents,
                    sigma=sigma,
                    sigma_next=sigma_next,
                    guidance_scale=float(params.guidance_scale),
                    eta=step_eta,
                    sigma_max=sigma_max,
                    step_index=i,
                    state=state,
                    generator=noise_generators,
                )
            latents = new_latents.to(dtype=self.trajectory_dtype)

            if (i + 1) in needed:
                stored_pairs.append((i + 1, latents.detach().clone()))

            if log_prob is not None:
                sde_logp_list.append(log_prob.to(dtype=self.logprob_dtype))

        positions_collected = [p for p, _ in stored_pairs]
        latents_stacked = torch.stack([t for _, t in stored_pairs], dim=1)

        sde_logp = torch.stack(sde_logp_list, dim=1) if sde_logp_list else None
        sde_indices_tensor = torch.tensor(sde_sorted, dtype=torch.long, device=device) if sde_sorted else None

        indices_tensor = torch.tensor(positions_collected, dtype=torch.long, device=device)

        return LatentSegment(
            latents=latents_stacked,
            sigmas=schedule,
            indices=indices_tensor,
            sde_logp=sde_logp,
            sde_indices=sde_indices_tensor,
        )

    def replay(
        self,
        conditions: HunyuanImage3DiffusionConditions,
        *,
        segment: LatentSegment,
        params: DiffusionSamplingParams,
        step_indices: Optional[List[int]] = None,
    ) -> ReplayResult:
        """Segment-based log-prob replay over the rollout's SDE transitions."""
        if segment.sde_indices is None or segment.latents is None:
            raise ValueError("HunyuanImage3DiffusionStage.replay: segment.sde_indices / latents missing")
        if segment.sigmas is None:
            raise ValueError("HunyuanImage3DiffusionStage.replay: segment.sigmas missing")

        # Restack CFG branches so replay matches guided rollout.
        conditions = _expand_cfg_for_forward(conditions)

        sde_set = set(int(i) for i in segment.sde_indices.tolist())
        target = (
            [int(i) for i in step_indices]
            if step_indices is not None
            else [int(i) for i in segment.sde_indices.tolist()]
        )
        bad = [i for i in target if i not in sde_set]
        if bad:
            raise ValueError(
                f"HunyuanImage3DiffusionStage.replay: step_indices {bad} not in segment.sde_indices={sorted(sde_set)}"
            )

        device = segment.latents.device
        sigmas = segment.sigmas.to(device)
        sigma_max = float(sigmas[1].item()) if int(sigmas.shape[0]) > 1 else 0.99

        autocast_ctx = (
            torch.autocast("cuda", self.autocast_dtype)
            if device.type == "cuda" and self.autocast_dtype in (torch.float16, torch.bfloat16)
            else nullcontext()
        )
        log_probs: List[torch.Tensor] = []
        prev_sample_means: List[torch.Tensor] = []
        with autocast_ctx:
            for step_idx in target:
                sigma = sigmas[step_idx].to(dtype=torch.float32)
                sigma_next = sigmas[step_idx + 1].to(dtype=torch.float32)
                sample = segment.latents_at(step_idx)
                prev_sample = segment.latents_at(step_idx + 1)
                _, log_prob, prev_mean = self.step.step_with_logp(
                    self.model,
                    conditions,
                    strategy=self.strategy,
                    sample=sample,
                    prev_sample=prev_sample,
                    sigma=sigma,
                    sigma_next=sigma_next,
                    guidance_scale=float(params.guidance_scale),
                    eta=float(params.eta),
                    sigma_max=sigma_max,
                    step_index=step_idx,
                )
                if log_prob is None:
                    raise RuntimeError(
                        f"HunyuanImage3DiffusionStage.replay: strategy "
                        f"returned None log-prob at step_index={step_idx} "
                        f"(deterministic mode); replay requires a stochastic "
                        f"SDE strategy."
                    )
                log_probs.append(log_prob)
                if prev_mean is not None:
                    prev_sample_means.append(prev_mean)

        log_probs_t = torch.stack(log_probs, dim=1).to(dtype=self.logprob_dtype)
        means_t = torch.stack(prev_sample_means, dim=1).to(dtype=self.trajectory_dtype) if prev_sample_means else None
        return ReplayResult(log_probs=log_probs_t, prev_sample_means=means_t)

    def predict_noise_at_step(
        self,
        conditions: HunyuanImage3DiffusionConditions,
        *,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        params: DiffusionSamplingParams,
    ) -> torch.Tensor:
        """Single ``(xt, sigma)`` model forward — no scheduler iteration."""
        # Restack CFG branches to match diffuse() and replay().
        conditions = _expand_cfg_for_forward(conditions)
        return self.step.predict_noise(
            self.model,
            sample,
            sigma,
            conditions,
            guidance_scale=float(params.guidance_scale),
        )

    def trainable_module(self) -> "torch.nn.Module":
        """Return the module the diffusion forward operates on."""
        return self.model.transformer.model


__all__ = [
    "HunyuanImage3DiffusionStage",
    "HunyuanImage3DiffusionStep",
]
