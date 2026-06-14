"""HunyuanImage3 diffusion: typed params + per-step kernel + rollout-level stage.

Three classes:

- ``HunyuanImage3DiffusionParams`` — typed request-shape knobs (steps /
  guidance / size / seed / sde_indices / eta / init_same_noise /
  samples_per_prompt / noise_group_ids / taylor_cache_*).
- ``HunyuanImage3DiffusionStep`` — stateless per-step kernel. ``step`` /
  ``step_with_logp`` take the model + conditions + strategy and run both
  CFG noise prediction and the SDE transition (via
  ``StepStrategy.denoise``). ``forward`` is a lower-level helper that
  takes a precomputed ``noise_pred``.
- ``HunyuanImage3DiffusionStage`` — implements
  ``DiffusionStage[HunyuanImage3DiffusionConditions]``. Owns the SDE
  ``strategy`` and the loop bookkeeping; delegates per-step model+SDE
  work to the kernel. Also exposes ``replay`` for single-step log-prob
  replay during training.

``predict_noise`` drives the real upstream
``HunyuanImage3ForCausalMM.forward(mode="gen_image")`` — the unified
multimodal transformer where text + image tokens share one sequence.
It reads the prepared multimodal tensors from
``HunyuanImage3DiffusionConditions.fused`` (a
``HunyuanImage3FusedMultimodalCondition`` carrying ``input_ids``,
``attention_mask``, ``position_ids``, ``rope_cache``, plus the 5
scatter-layout masks/indices), all built by
:meth:`HunyuanImage3TextEmbedStage.embed_for_gen_image`. It calls
``transformer.prepare_inputs_for_generation(...)`` followed by the
forward with ``first_step=True, use_cache=False`` — KV-cache reuse
across diffusion steps is intentionally out of scope and tracked as a
follow-up.
"""

from __future__ import annotations

import logging
import sys
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from unirl.models.types.diffusion import DiffusionStage, DiffusionStep
from unirl.models.types.replay_result import ReplayResult
from unirl.sde.kernels import StepStrategy
from unirl.types.noise_recipe import NoiseRecipe
from unirl.types.sampling import DiffusionSamplingParams
from unirl.types.segments.latent import LatentSegment
from unirl.types.trajectory_store import compute_trajectory_positions
from unirl.utils.dtypes import parse_torch_dtype

from .bundle import HunyuanImage3Bundle
from .conditions import HunyuanImage3DiffusionConditions
from .diffusion_state import HunyuanImage3DiffusionState

logger = logging.getLogger(__name__)


class HunyuanImage3DiffusionStep(DiffusionStep[HunyuanImage3Bundle, HunyuanImage3DiffusionConditions]):
    """Per-step HunyuanImage3 denoising kernel — stateless.

    ``step`` / ``step_with_logp`` take the model + conditions + an SDE
    ``strategy`` per call, run CFG noise prediction internally, then
    apply the transition via ``strategy.denoise``. ``forward`` is the
    lower-level escape hatch that takes a precomputed ``noise_pred``.
    """

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
        """
        Run the unified MM transformer in ``mode="gen_image"`` and return
        the CFG-combined noise prediction.
        """
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

        # CFG batching is driven by the captured ``fused`` shape, NOT by
        # ``guidance_scale``. vllm-omni's HI3 pipeline captures
        # ``prepare_inputs_for_generation``'s ``input_ids`` at a single-prefill
        # boundary — the capture is always shape ``[B, L]`` (cond-only),
        # regardless of whether the engine internally implements CFG via two
        # separate forwards or one cfg-batched forward.
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
        # timestep: scalar or [B] -> [N] float, matching sample_2's batch axis.
        t_scalar = sigma * 1000.0
        if t_scalar.dim() == 0:
            t_expand = t_scalar.expand(sample_2.shape[0])
        else:
            t_expand = t_scalar.expand(sample_2.shape[0])

        # Decide which path we're on — stateless vs KV-cached.
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
        else:
            cond_vae_images = None
            cond_timestep = None
            cond_vae_image_mask = None
            cond_vit_images = None
            cond_vit_image_mask = None
            vit_kwargs = None

        # Build model_inputs directly instead of calling
        # transformer.prepare_inputs_for_generation(). Under FSDP2 the
        # method dispatch can strip **kwargs. Building the dict here is
        # equivalent and robust to FSDP wrapping.
        input_ids_in = fused.input_ids
        # On decode steps (is_first=False) the checkpoint's _update shrinks
        # position_ids to the changed slice (timestep + image tokens); gather
        # input_ids AND gen_image_mask to that slice so they match the forward's
        # hidden length (else masked_select sees full-L mask vs slice-L hidden).
        image_mask_in = fused.gen_image_mask
        if input_ids_in is not None and position_ids_in is not None:
            if input_ids_in.shape[1] != position_ids_in.shape[1]:
                input_ids_in = torch.gather(input_ids_in, dim=1, index=position_ids_in)
                if image_mask_in is not None:
                    image_mask_in = torch.gather(image_mask_in, dim=1, index=position_ids_in)
        # [ROPE-FIX] config.rope_type=="2d" needs a 2-D
        # RoPE for image tokens, built from per-image (slice,(token_h,token_w)).
        # The original replay passed EMPTY rope_image_info, so build_2d_rope gives
        # image tokens plain 1-D sequential positions — wrong for a 2-D model and
        # the confirmed cause of the ~40% vllm-vs-HF noise_pred divergence (image
        # ratio 0.95 → 0.996 once fixed). We reconstruct per sample:
        #   - slice: the contiguous image-token run from gen_image_mask.
        #   - (token_h, token_w): the patchified token grid. Patchify uses uniform
        #     square patches, so the token grid preserves the LATENT aspect ratio:
        #     token_h/token_w == H_lat/W_lat and token_h*token_w == n. Solving:
        #       token_w = round(sqrt(n * W_lat / H_lat)), token_h = n // token_w.
        #     This is general (handles non-square aspect ratios) and self-contained
        #     — no need to plumb (h,w) from the rollout capture. Square images
        #     degenerate to token_h==token_w==isqrt(n). ``sample`` is
        #     [n, C, H_lat, W_lat]; H_lat/W_lat are uniform across the batch.
        # Then the transformer builds its native 128-dim 2-D rope (no tensor
        # injection → no head-dim mismatch).
        _B = int(fused.input_ids.shape[0])
        rope_image_info_val: List[List[Any]] = [[] for _ in range(_B)]
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
        # Forward-scatter index for the timestep continuous embedding: on is_first
        # the model scatters into the full sequence; on decode steps it scatters
        # into the [timestep, image] slice where the timestep is the first token
        # (index 0). The full-sequence gen_timestep_scatter_index overflows the
        # slice for long cond-image sequences (it2i) -> scatter index OOB.
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
            # native sets timesteps_index = gen_timestep_scatter_index
            # (modeling:2836) so instantiate_continuous_tokens injects the
            # timestep token's continuous embedding (required for gen_image;
            # passing None silently skips it and corrupts the noise_pred).
            "timesteps_index": timesteps_index_in,
            "gen_timestep_scatter_index": scatter_idx_in,
            "cond_vae_images": cond_vae_images,
            "cond_vae_image_mask": cond_vae_image_mask,
            "cond_timesteps": cond_timestep,
            "cond_timesteps_index": None,
            "cond_vit_images": cond_vit_images,
            "cond_vit_image_mask": cond_vit_image_mask,
            "cond_vit_image_kwargs": vit_kwargs,
        }
        # Bypass _check_inputs: we build model_inputs by hand (not via
        # prepare_inputs_for_generation), so the upstream first_step+gen_image
        # assertions don't all line up with this hand-built dict. timesteps_index
        # IS provided (scatter_idx_in above) so instantiate_continuous_tokens runs.
        _orig_check = getattr(transformer, "_check_inputs", None)
        transformer._check_inputs = lambda *a, **kw: None

        # Ensure runtime attributes that forward() reads off ``self`` are set.
        # UNCONDITIONAL reset — GRPO sets num_image_tokens=0 on the same
        # transformer instance. If DiffGRPO runs after GRPO, the 0 persists
        # → rope OOB / NaN.
        transformer.post_token_len = None
        n_img = int(fused.gen_image_mask.sum(dim=-1).max().item()) if fused.gen_image_mask is not None else 0
        transformer.num_image_tokens = n_img
        # ragged_final_layer at decode steps slices hidden_states[:, num_special_tokens:]
        # to drop the leading timestep/special tokens before the image region; on
        # is_first it uses image_mask instead (None is correct there).
        transformer.num_special_tokens = None if is_first else (int(input_ids_in.shape[1]) - n_img)

        output = transformer(**model_inputs, first_step=is_first)

        # Restore _check_inputs
        if _orig_check is not None:
            transformer._check_inputs = _orig_check

        # Update state for the next step.
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
        """Build a ``HunyuanStaticCache`` sized for the full sequence.

        Mirrors the upstream pattern at ``hunyuan.py:~2330``: for
        ``mode="gen_image"``, ``max_cache_len = output.tokens.shape[1]``
        (the full L), ``dynamic=False`` (no growth across diffusion
        steps), batch_size = N (CFG-batched).

        Returns ``None`` if the upstream module doesn't expose a
        ``HunyuanStaticCache`` symbol — caller falls back to whatever
        default the model uses (typically a ``DynamicCache``, which
        also works but allocates more aggressively).
        """
        fused = conditions.fused
        if fused is None or fused.input_ids is None:
            return None
        upstream_mod = sys.modules[type(transformer).__module__]
        cache_cls = getattr(upstream_mod, "HunyuanStaticCache", None)
        if cache_cls is None:
            return None
        max_cache_len = int(fused.input_ids.shape[1])
        batch_size = int(fused.input_ids.shape[0])
        # bf16 default matches upstream pipeline; safe regardless of
        # autocast since the cache stores K/V at the model's compute dtype.
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
        """Carry past_key_values + the gathered position/attention tensors
        into ``state`` for the next diffusion step.

        Mirrors upstream ``HunyuanImage3ForCausalMM._update_model_kwargs_for_generation``
        (hunyuan.py:2438):

        - On the **first** call (``is_first=True``), passes
          ``tokenizer_output`` from the conditions so the upstream branches
          into the gather-down path: it builds new ``position_ids`` of
          shape ``[N, L']`` (just the timestep + image positions), and
          ``index_select``-s the original ``[N, 1, L', L]``-shaped attention
          mask down to those L' rows.

        - On **subsequent** calls (``is_first=False``), omits
          ``tokenizer_output`` so the upstream falls into the trivial
          else-branch and just propagates ``position_ids`` / ``attention_mask``
          / ``gen_timestep_scatter_index`` unchanged from ``state``.

        Without ``tokenizer_output`` on the first call, the model would try
        to mix step-1+'s ``inputs_embeds`` (length L') against full-length
        rope tables (length L) and crash with a tensor-size mismatch in
        ``apply_rotary_pos_emb``.
        """
        # Build the input model_kwargs that the helper expects. On step 0
        # it sees the conditions (full L) plus ``tokenizer_output`` to
        # trigger the gather. On subsequent steps it sees the state
        # (slice L') and omits tokenizer_output.
        fused = conditions.fused
        assert fused is not None  # asserted by predict_noise before reaching here
        # upstream _update_model_kwargs_for_generation reads model_kwargs[
        # "rope_image_info"] unconditionally (modeling_hunyuan_image_3.py:2944)
        # and just propagates it forward; predict_noise rebuilds it fresh each
        # step (line ~197) so the value carried here is never consumed — it only
        # needs to be PRESENT to avoid a KeyError. 1D rope => empty per-sample.
        _rope_info = [[] for _ in range(int(fused.input_ids.shape[0]))]
        if is_first:
            mk: Dict[str, Any] = {
                "mode": "gen_image",
                "attention_mask": fused.attention_mask,
                "position_ids": fused.position_ids,
                "image_mask": fused.gen_image_mask,
                "gen_timestep_scatter_index": fused.gen_timestep_scatter_index,
                "timesteps_index": fused.gen_timestep_scatter_index,
                "custom_pos_emb": fused.rope_cache,
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
                "custom_pos_emb": fused.rope_cache,
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

    # ---- Protocol surface ---------------------------------------------------

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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        return strategy.denoise(
            noise_pred=noise_pred,
            sample=sample,
            sigma=sigma,
            sigma_next=sigma_next,
            eta=eta,
            prev_sample=prev_sample,
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
        )


def _conditions_device_and_batch(
    conditions: HunyuanImage3DiffusionConditions,
    *,
    guidance_scale: float,
) -> Tuple[torch.device, int]:
    """Resolve ``(device, batch_size)`` from the conditions container.

    Reads ``conditions.fused.input_ids`` (shape ``[N, L]`` with
    ``N = B * cfg``).
    """
    cfg = 2 if guidance_scale > 1.0 else 1
    fused = conditions.fused
    if fused is None or fused.input_ids is None:
        raise ValueError(
            "HunyuanImage3DiffusionStage: conditions.fused.input_ids is None; cannot infer device / batch."
        )
    n = int(fused.input_ids.shape[0])
    if n % cfg != 0:
        raise ValueError(
            f"HunyuanImage3DiffusionStage: input_ids batch ({n}) is not a "
            f"multiple of CFG factor ({cfg}). Did the pipeline forget to "
            f"build CFG-batched inputs?"
        )
    return fused.input_ids.device, n // cfg


class HunyuanImage3DiffusionStage(DiffusionStage[HunyuanImage3DiffusionConditions]):
    """HunyuanImage3 rollout-level diffusion stage.

    Owns the SDE ``strategy``, bundle, kernel, and precision policy. The
    kernel is stateless and is invoked per-step with the strategy passed
    in.

    ``diffuse(conditions, *, schedule, params)`` runs the full sampling
    loop and returns a ``LatentSegment`` carrying the trajectory plus
    per-SDE log probs (``sde_logp [N, S]`` + ``sde_indices [S]``).

    ``replay(conditions, *, segment, params, step_indices=None)``
    recomputes log-probs for the SDE transitions in a stored
    ``LatentSegment``. Returns ``[B, S']`` aligned with
    ``segment.sde_logp`` (or a slice when ``step_indices`` selects a
    subset). Used by GRPO-style training.
    """

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
    ) -> None:
        self.model = model
        self.step = step
        self.strategy = strategy
        self.autocast_dtype = parse_torch_dtype(autocast_precision, field_name="autocast_precision")
        self.trajectory_dtype = parse_torch_dtype(trajectory_precision, field_name="trajectory_precision")
        self.logprob_dtype = parse_torch_dtype(logprob_precision, field_name="logprob_precision")
        self.vae_scale_factor = vae_scale_factor
        self.latent_channels = latent_channels

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def diffuse(
        self,
        conditions: HunyuanImage3DiffusionConditions,
        *,
        schedule: torch.Tensor,
        params: DiffusionSamplingParams,
    ) -> LatentSegment:
        """Run full HunyuanImage3 DiT sampling. Returns a ``LatentSegment``.

        Shape contract for the returned segment (with ``B`` = number of
        prompts, ``T`` = ``params.num_inference_steps``,
        ``C = self.latent_channels``,
        ``H = params.height // self.vae_scale_factor``,
        ``W = params.width  // self.vae_scale_factor``,
        ``K`` = number of stored trajectory positions including the clean
        latent at position ``T``):

            latents      : [B, K, C, H, W]
            sde_logp     : [B, S]    where S = len(params.sde_indices)
            sde_indices  : [S]       long
            indices      : [K]       long (positions of the stored snapshots)
            sigmas       : [T+1]     float (the schedule)
        """
        from unirl.sde.noise import generate_latents

        device, batch_size = _conditions_device_and_batch(conditions, guidance_scale=float(params.guidance_scale))
        T = int(params.num_inference_steps)
        if int(schedule.shape[0]) != T + 1:
            raise ValueError(f"HunyuanImage3DiffusionStage.diffuse: schedule length {schedule.shape[0]} != T+1={T + 1}")
        schedule = schedule.to(device)
        self.strategy.init_schedule(schedule)

        # HI3 snaps any requested H×W to the nearest preset at base_size² area
        # (image_base_size=1024 → ~1MP) — the text-embed stage's
        # build_gen_image_info does this, so the <img> placeholder span it
        # splices is sized from the SNAPPED token grid. Size the latent from that
        # SAME snapped grid, else a non-preset request (e.g. 512²) yields a
        # latent (raw H//vae_scale) shorter than the placeholder span → the
        # image-token scatter mismatches (index N vs src M). At a preset size
        # (e.g. 1024²) get_target_size is a no-op, so this is identical to before.
        ip = getattr(self.model.transformer, "image_processor", None)
        info = None
        if ip is not None:
            if hasattr(ip, "build_image_info"):
                info = ip.build_image_info(f"{int(params.height)}x{int(params.width)}")
            elif hasattr(ip, "build_gen_image_info"):
                info = ip.build_gen_image_info(f"{int(params.height)}x{int(params.width)}")
        if info is not None:
            # token_{height,width} are computed from the snapped image dims; the
            # DiT treats each latent spatial position as one image token, so the
            # latent grid == the placeholder grid.
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
        # latents: [B, C, H, W]. Driver-authoritative x_T via the shared
        # ``NoiseRecipe`` — the SAME ``for_batch(...).resolve(...)`` path the
        # vLLM worker (RLHunyuanImage3Pipeline) takes, so trainside and rollout
        # regenerate a BYTE-IDENTICAL x_T from the recipe (gids + seed) on
        # CPU-fp32. The shape is AR-known only here, so it's filled via
        # ``for_batch``. Falls back to engine-drawn noise when no recipe gids
        # were shipped (e.g. DISABLE_DRIVER_XT).
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

        # Per-rollout KV-cache state. Step 0 fills it; steps 1..T-1
        # consume + update. Replay() does NOT use this — each replay step
        # starts from a stored intermediate latent so prior-step cache is
        # meaningless.
        state = HunyuanImage3DiffusionState()

        for i in range(T):
            sigma = schedule[i].to(device)
            sigma_next = schedule[i + 1].to(device)
            step_eta = float(params.eta) if i in sde_set else 0.0

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
                )
            latents = new_latents.to(dtype=self.trajectory_dtype)

            if (i + 1) in needed:
                stored_pairs.append((i + 1, latents.detach().clone()))

            if log_prob is not None:
                sde_logp_list.append(log_prob.to(dtype=self.logprob_dtype))

        positions_collected = [p for p, _ in stored_pairs]
        # latents_stacked: [B, K, C, H, W]
        latents_stacked = torch.stack([t for _, t in stored_pairs], dim=1)

        # sde_logp: [B, S]
        sde_logp = torch.stack(sde_logp_list, dim=1) if sde_logp_list else None
        # sde_indices_tensor: [S] long
        sde_indices_tensor = torch.tensor(sde_sorted, dtype=torch.long, device=device) if sde_sorted else None

        indices_tensor = torch.tensor(positions_collected, dtype=torch.long, device=device)

        return LatentSegment(
            latents=latents_stacked,
            sigmas=schedule,
            indices=indices_tensor,
            sde_logp=sde_logp,
            sde_indices=sde_indices_tensor,
        )

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------

    def replay(
        self,
        conditions: HunyuanImage3DiffusionConditions,
        *,
        segment: LatentSegment,
        params: DiffusionSamplingParams,
        step_indices: Optional[List[int]] = None,
    ) -> ReplayResult:
        """Segment-based log-prob replay over the rollout's SDE transitions.

        Mirrors ``SD3DiffusionStage.replay``: loop the per-step replay
        primitive (``step.step_with_logp`` with ``prev_sample`` set) over
        the segment's SDE indices (or the ``step_indices`` subset, which
        must be a subset of ``segment.sde_indices``). Returns a
        :class:`ReplayResult` with ``log_probs`` shape ``[B, len(target)]``
        aligned with the corresponding slice of ``segment.sde_logp``
        (cast to ``logprob_precision``) and ``prev_sample_means`` shape
        ``[B, len(target), *latent_shape]`` carrying the SDE Gaussian
        means μ_θ for KL-penalty consumption.
        """
        if segment.sde_indices is None or segment.latents is None:
            raise ValueError("HunyuanImage3DiffusionStage.replay: segment.sde_indices / latents missing")
        if segment.sigmas is None:
            raise ValueError("HunyuanImage3DiffusionStage.replay: segment.sigmas missing")

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
                # sample, prev_sample: [B, C, H, W]
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

        # log_probs: [B, len(target)] float, in logprob_precision
        log_probs_t = torch.stack(log_probs, dim=1).to(dtype=self.logprob_dtype)
        # prev_sample_means: [B, len(target), *latent_shape] in trajectory dtype.
        # None if the strategy didn't produce them at any step.
        means_t = torch.stack(prev_sample_means, dim=1).to(dtype=self.trajectory_dtype) if prev_sample_means else None
        return ReplayResult(log_probs=log_probs_t, prev_sample_means=means_t)

    # ------------------------------------------------------------------
    # Single-step noise prediction (forward-process algorithms: DiffusionNFT et al.)
    # ------------------------------------------------------------------

    def predict_noise_at_step(
        self,
        conditions: HunyuanImage3DiffusionConditions,
        *,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        params: DiffusionSamplingParams,
    ) -> torch.Tensor:
        """Single ``(xt, sigma)`` model forward — no scheduler iteration.

        Stateless mode (``state=None``, ``step_index=0``); HI3's stateful
        cache is only meaningful inside an SDE trajectory, which DiffusionNFT-style
        forward-process algorithms don't traverse.
        """
        return self.step.predict_noise(
            self.model,
            sample,
            sigma,
            conditions,
            guidance_scale=float(params.guidance_scale),
        )

    # ------------------------------------------------------------------
    # Trainable surface for FSDPPolicy
    # ------------------------------------------------------------------

    def trainable_module(self) -> "torch.nn.Module":
        """Return the module the diffusion forward operates on.

        For HI3, that's the bare decoder (``HunyuanImage3Model``) — the
        FSDP wrap target. The HF wrapper (``HunyuanImage3ForCausalMM``)
        owns frozen VAE + ViT siblings that must NOT be FSDP-wrapped
        (mixed dtypes; not in the diffusion forward path).
        """
        return self.model.transformer.model


__all__ = [
    "HunyuanImage3DiffusionStage",
    "HunyuanImage3DiffusionStep",
]
