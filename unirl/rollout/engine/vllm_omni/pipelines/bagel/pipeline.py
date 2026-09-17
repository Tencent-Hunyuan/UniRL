"""RL-aware BAGEL-7B-MoT pipeline subclass."""

from __future__ import annotations

import copy
import logging
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import PIL.Image
import torch
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.bagel.bagel_transformer import NaiveCache
from vllm_omni.diffusion.models.bagel.pipeline_bagel import BagelPipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from unirl.models.bagel.rl_ops import (
    build_image_transforms,
    resize_input_image,
    update_context_image,
)
from unirl.rollout.engine.vllm_omni.pipelines._shared.interception import (
    drain_trajectory_into,
    finalize_output,
    resolve_request_noise,
    set_payload,
    single_request,
)
from unirl.rollout.engine.vllm_omni.pipelines.bagel.bagel_flow_match_sde_scheduler import (
    BagelFlowSDEScheduler,
)
from unirl.utils.dtypes import parse_torch_dtype

logger = logging.getLogger(__name__)


class RLBagelPipeline(BagelPipeline):
    """BAGEL pipeline with the RL interception protocol installed."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__(od_config=od_config, prefix=prefix)
        self._sde_scheduler = BagelFlowSDEScheduler()
        self._sde_scheduler_installed = False
        self._noise_tap_installed = False
        self._generate_image_tap_installed = False
        self._rope_fp32_patched = False
        self._rmsnorm_fp32_patched = False
        self._pending_initial_noise: Optional[torch.Tensor] = None
        self._pending_spp: int = 1
        self._pending_batched_latents: Optional[list] = None
        self._trajectory_dtype: torch.dtype = torch.float32
        self._vae_transform: Optional[Any] = None
        self._vit_transform: Optional[Any] = None

    def _install_sde_scheduler(self) -> None:
        """Point ``self.scheduler`` at the trajectory-capturing SDE scheduler — always installed, even at eta=0."""
        if self._sde_scheduler_installed:
            return
        self.scheduler = self._sde_scheduler
        self.scheduler_kwargs = {}
        self._sde_scheduler_installed = True

    def _install_rope_fp32(self) -> None:
        """Force the rotary cos/sin to fp32 to bit-match trainside — the worker rotary runs under unguarded autocast."""
        if self._rope_fp32_patched:
            return
        try:
            rotary = self.bagel.language_model.model.rotary_emb
        except AttributeError:
            # Topology changed (e.g. und-only build); skip rather than crash.
            self._rope_fp32_patched = True
            return

        if getattr(rotary, "_unirl_fp32_forward", False):
            self._rope_fp32_patched = True
            return

        orig_forward = rotary.forward

        @torch.no_grad()
        def fp32_forward(x: torch.Tensor, position_ids: torch.Tensor):
            inv_freq_expanded = rotary.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
            position_ids_expanded = position_ids[:, None, :].float()
            device_type = x.device.type
            device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
            with torch.autocast(device_type=device_type, enabled=False):
                freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos()
                sin = emb.sin()
            cos = cos * rotary.attention_scaling
            sin = sin * rotary.attention_scaling
            return cos.to(dtype=torch.float32), sin.to(dtype=torch.float32)

        rotary.forward = fp32_forward  # type: ignore[assignment]
        rotary._unirl_fp32_forward = True  # type: ignore[attr-defined]
        # Keep a handle for debugging / potential revert; never restored in-run.
        rotary._unirl_orig_forward = orig_forward  # type: ignore[attr-defined]
        logger.warning("[PATCH-INSTALLED] rope_fp32 modules=1 (rotary_emb)")
        self._rope_fp32_patched = True

    def _install_rmsnorm_fp32(self) -> None:
        """Make every worker RMSNorm bit-match trainside ``Qwen2RMSNorm`` — vLLM rounds the fp32 q/k-norm to bf16."""
        if self._rmsnorm_fp32_patched:
            return
        try:
            from vllm.model_executor.layers.layernorm import RMSNorm as _VllmRMSNorm
        except Exception:
            self._rmsnorm_fp32_patched = True
            return

        def _make_fp32_forward(module: Any):
            eps = float(getattr(module, "variance_epsilon", getattr(module, "eps", 1e-6)))
            orig = module.forward

            def fp32_forward(x: torch.Tensor, residual: Optional[torch.Tensor] = None):
                if residual is not None:
                    return orig(x, residual)
                input_dtype = x.dtype
                h = x.to(torch.float32)
                variance = h.pow(2).mean(-1, keepdim=True)
                h = h * torch.rsqrt(variance + eps)
                return module.weight * h.to(input_dtype)

            return fp32_forward

        patched = 0
        for module in self.bagel.modules():
            if isinstance(module, _VllmRMSNorm) and not getattr(module, "_unirl_fp32_rmsnorm", False):
                module._unirl_orig_forward = module.forward  # type: ignore[attr-defined]
                module.forward = _make_fp32_forward(module)  # type: ignore[assignment]
                module._unirl_fp32_rmsnorm = True  # type: ignore[attr-defined]
                patched += 1
        logger.warning("[PATCH-INSTALLED] rmsnorm_fp32 modules=%d", patched)
        self._rmsnorm_fp32_patched = True

    def _install_noise_tap(self) -> None:
        """Wrap ``bagel.prepare_vae_latent`` to swap the driver-authored x_T in for upstream's RNG-drawn noises."""
        if self._noise_tap_installed:
            return

        orig = self.bagel.prepare_vae_latent
        pipeline_self = self

        def tapped(*args: Any, **kw: Any) -> Any:
            spp = pipeline_self._pending_spp
            if spp > 1 and "image_sizes" in kw and len(kw["image_sizes"]) == 1:
                kw = dict(kw)
                kw["image_sizes"] = list(kw["image_sizes"]) * spp
                kw["curr_kvlens"] = list(kw["curr_kvlens"]) * spp
                kw["curr_rope"] = list(kw["curr_rope"]) * spp
            out = orig(*args, **kw)
            noise = pipeline_self._pending_initial_noise
            if noise is not None:
                pipeline_self._pending_initial_noise = None
                ref = out.get("packed_init_noises")
                if ref is None:
                    raise RuntimeError(
                        "RLBagelPipeline noise tap: prepare_vae_latent returned no 'packed_init_noises' to override."
                    )
                if noise.dim() == ref.dim() + 1:
                    noise = noise.reshape(-1, noise.shape[-1]) if noise.shape[0] > 1 else noise.squeeze(0)
                if tuple(noise.shape) != tuple(ref.shape):
                    raise RuntimeError(
                        "RLBagelPipeline noise tap: driver x_T shape "
                        f"{tuple(noise.shape)} != worker packed_init_noises shape "
                        f"{tuple(ref.shape)} — check the recipe's "
                        "init_noise_latent_shape (bagel_latent_shape) vs the "
                        "request's height/width."
                    )
                out["packed_init_noises"] = noise.to(dtype=ref.dtype, device=ref.device)
            return out

        self.bagel.prepare_vae_latent = tapped  # type: ignore[assignment]
        self._noise_tap_installed = True

    @staticmethod
    def _replicate_prompt_kv(kwargs: Dict[str, Any], spp: int, merge_kv_caches: Any) -> Dict[str, Any]:
        """Clone the single prompt KV cache into ``spp`` views (one per packed image)."""
        past = kwargs.get("past_key_values")
        if past is None:
            return kwargs
        out = dict(kwargs)
        out["past_key_values"] = merge_kv_caches([past] * spp)
        return out

    def _install_generate_image_tap(self) -> None:
        """Wrap ``bagel.generate_image`` once for the grouped (spp>1) path."""
        if self._generate_image_tap_installed:
            return

        original_generate_image = self.bagel.generate_image
        # vllm-omni 0.27 moved _merge_naive_caches onto the cache as NaiveCache.merge.
        merge_kv_caches = NaiveCache.merge
        pipeline = self

        def generate_image_grouped(*args: Any, **kwargs: Any) -> Any:
            spp = pipeline._pending_spp
            if spp > 1:
                kwargs = pipeline._replicate_prompt_kv(kwargs, spp, merge_kv_caches)
            result = original_generate_image(*args, **kwargs)
            if spp > 1:
                pipeline._pending_batched_latents = list(result[0])
            return result

        self.bagel.generate_image = generate_image_grouped  # type: ignore[assignment]
        self._generate_image_tap_installed = True

    def _bundle_view(self) -> SimpleNamespace:
        """Return a ``BagelBundle``-shaped worker view for shared image prefill."""
        if self._vae_transform is None:
            self._vae_transform, self._vit_transform = build_image_transforms()
        return SimpleNamespace(
            model=self.bagel,
            vae=self.vae,
            vae_transform=self._vae_transform,
            vit_transform=self._vit_transform,
            new_token_ids=self.new_token_ids,
            device=self.device,
        )

    @staticmethod
    def _prompt_text(req: OmniDiffusionRequest) -> str:
        """The request's prompt string (upstream's own extraction, pipeline_bagel:327)."""
        prompt = req.prompts[0]
        return prompt if isinstance(prompt, str) else (prompt.get("prompt") or "")

    @staticmethod
    def _source_image(req: OmniDiffusionRequest) -> Optional[PIL.Image.Image]:
        """The it2i source PIL off the prompt dict; ``None`` on the t2i path."""
        prompt = req.prompts[0] if getattr(req, "prompts", None) else None
        if not isinstance(prompt, dict):
            return None
        image = (prompt.get("multi_modal_data") or {}).get("image")
        if image is None:
            return None
        if not isinstance(image, PIL.Image.Image):
            raise TypeError(
                "RLBagelPipeline: multi_modal_data['image'] must be ONE PIL image "
                f"(BagelInputAdapter ships one per sample); got {type(image).__name__}."
            )
        return image

    def _prefill_text(self, ctx: Dict[str, Any], prompt: str) -> Dict[str, Any]:
        """Advance a KV context with the upstream text-prefill path."""
        clean = str(prompt).removeprefix("<|im_start|>").removesuffix("<|im_end|>")
        gi, kv_lens, ropes = self.bagel.prepare_prompts(
            curr_kvlens=ctx["kv_lens"],
            curr_rope=ctx["ropes"],
            prompts=[clean],
            tokenizer=self.tokenizer,
            new_token_ids=self.new_token_ids,
        )
        gi = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in gi.items()}
        past = self.bagel.forward_cache_update_text(ctx["past_key_values"], **gi)
        return {"kv_lens": kv_lens, "ropes": ropes, "past_key_values": past}

    def _build_it2i_contexts(
        self, image: PIL.Image.Image, prompt: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Build the gen, drop-text, and drop-image KV contexts for editing."""
        bundle = self._bundle_view()
        gen = {
            "kv_lens": [0],
            "ropes": [0],
            "past_key_values": NaiveCache(self.bagel.config.llm_config.num_hidden_layers),
        }
        cfg_img = copy.deepcopy(gen)
        autocast = torch.autocast(
            device_type=self.device.type,
            enabled=self.device.type != "cpu",
            dtype=self.od_config.dtype,
        )
        with torch.no_grad(), autocast:
            resized = resize_input_image(bundle, image)
            gen = update_context_image(bundle, resized, gen, vae=True, vit=True)
            cfg_text = copy.deepcopy(gen)  # snapshot before the prompt text
            gen = self._prefill_text(gen, prompt)
            cfg_img = self._prefill_text(cfg_img, prompt)
        return gen, cfg_text, cfg_img

    def _inject_it2i_contexts(self, req: OmniDiffusionRequest, image: PIL.Image.Image) -> None:
        """Inject it2i KV contexts and requested canvas metadata into a copied request."""
        sp = req.sampling_params
        if sp.height is None or sp.width is None:
            raise ValueError(
                "RLBagelPipeline it2i: sampling_params must carry height/width (the "
                "output canvas the driver's x_T recipe was authored for)."
            )
        image_shape = (int(sp.height), int(sp.width))
        gen, cfg_text, cfg_img = self._build_it2i_contexts(image, self._prompt_text(req))

        sp = copy.copy(sp)
        sp.past_key_values = gen["past_key_values"]
        sp.kv_metadata = {"ropes": gen["ropes"], "image_shape": image_shape}
        sp.cfg_text_past_key_values = cfg_text["past_key_values"]
        sp.cfg_text_kv_metadata = {"ropes": cfg_text["ropes"]}
        sp.cfg_img_past_key_values = cfg_img["past_key_values"]
        sp.cfg_img_kv_metadata = {"ropes": cfg_img["ropes"]}
        req.sampling_params = sp

    # ------------------------------------------------------------------ #
    # arm — every request (stale-leak guards)
    # ------------------------------------------------------------------ #

    def _arm_sde(self, req: OmniDiffusionRequest, image_token_sizes: Optional[list] = None) -> None:
        """This request's SDE strength + sparse step gate + σ_max + storage dtype."""
        sp = req.sampling_params
        eta = float(getattr(sp, "eta", 0.0) or 0.0)
        extra = getattr(sp, "extra_args", None) or {}
        traj_dtype_name = extra.get("trajectory_precision")
        traj_dtype = (
            parse_torch_dtype(traj_dtype_name, field_name="trajectory_precision")
            if traj_dtype_name
            else self._trajectory_dtype
        )
        sigma_max = extra.get("sigma_max")
        self._sde_scheduler.set_for_request(
            eta=eta,
            sde_indices=extra.get("sde_indices"),
            sigma_max=float(sigma_max) if sigma_max is not None else None,
            trajectory_dtype=traj_dtype,
            image_token_sizes=image_token_sizes,
        )

    def _arm_initial_noise(self, req: OmniDiffusionRequest) -> None:
        """This request's driver-authored x_T (batch slice or recipe row)."""
        self._pending_initial_noise = resolve_request_noise(req, caller="RLBagelPipeline._arm_initial_noise")

    def _harvest_trajectory(self, out: DiffusionOutput) -> None:
        """Overwrite upstream's trajectory capture with the SDE scheduler's — the ``build_image_segment`` wire."""
        drain_trajectory_into(out, self._sde_scheduler)

    def _is_batchable_t2i(self, req: OmniDiffusionRequest) -> bool:
        """Packed DiT batching: pure text→image at cfg=1 only. Expects the unwrapped request."""
        fp = req.prompts[0] if getattr(req, "prompts", None) else None
        if isinstance(fp, dict):
            modalities = fp.get("modalities") or []
            if "text" in modalities:
                return False
            if (fp.get("multi_modal_data") or {}).get("image") is not None:
                return False
        extra = getattr(req.sampling_params, "extra_args", None) or {}
        if "cfg_text_scale" not in extra or "cfg_img_scale" not in extra:
            return False
        return extra["cfg_text_scale"] <= 1.0 and extra["cfg_img_scale"] <= 1.0

    def forward(self, req: DiffusionRequestBatch, **kwargs) -> DiffusionOutput:
        """Single-request batch in, single output out — see ``single_request``."""
        one = single_request(req, caller="RLBagelPipeline.forward")
        self._install_sde_scheduler()
        self._install_noise_tap()
        self._install_rope_fp32()
        self._install_rmsnorm_fp32()

        spp = getattr(one.sampling_params, "num_outputs_per_prompt", 1)
        if spp > 1:
            if not self._is_batchable_t2i(one):
                raise RuntimeError(
                    f"RLBagelPipeline: num_outputs_per_prompt={spp} requires pure t2i "
                    f"with cfg_text_scale<=1 and cfg_img_scale<=1 present in "
                    f"sampling_params.extra_args. BagelInputAdapter should leave "
                    f"num_outputs_per_prompt=1 (sample-level layout) when packing "
                    f"is disabled."
                )
            return self._forward_batched(req, spp, **kwargs)

        # it2i: build the conditioning ourselves (trainside-identical) and inject it,
        # so upstream's own img2img prefill never runs. No-op for t2i.
        image = self._source_image(one)
        if image is not None:
            self._inject_it2i_contexts(one, image)

        self._arm_sde(one)
        self._arm_initial_noise(one)

        out = super().forward(req, **kwargs)

        self._harvest_trajectory(out)
        finalize_output(out)
        return out

    def _forward_batched(self, req: DiffusionRequestBatch, spp: int, **kwargs) -> DiffusionOutput:
        """Pack ``spp`` same-prompt images into ONE ``generate_image``."""
        one = single_request(req, caller="RLBagelPipeline._forward_batched")
        self._install_generate_image_tap()
        ds = int(self.bagel.latent_downsample)
        per = (int(one.sampling_params.height) // ds) * (int(one.sampling_params.width) // ds)
        self._arm_sde(one, image_token_sizes=[per] * spp)
        self._arm_initial_noise(one)
        self._pending_spp = spp
        self._pending_batched_latents = None
        try:
            out = super().forward(req, **kwargs)
            self._harvest_trajectory(out)
            finalize_output(out)
            lats = self._pending_batched_latents
            if not lats or len(lats) != spp:
                raise RuntimeError(
                    f"RLBagelPipeline batched forward: generate_image tap captured "
                    f"{0 if not lats else len(lats)} latents, expected spp={spp}."
                )
            raw = out.output
            first = raw[0] if isinstance(raw, (list, tuple)) and raw else raw
            image_shape = (int(one.sampling_params.height), int(one.sampling_params.width))
            if first is not None:
                images = [first] + [
                    self._decode_image_from_latent(self.bagel, self.vae, lat, image_shape) for lat in lats[1:]
                ]
            else:
                images = [self._decode_image_from_latent(self.bagel, self.vae, lat, image_shape) for lat in lats]
            set_payload(out, images)
        finally:
            self._pending_spp = 1
            self._pending_batched_latents = None
        return out


__all__ = ["RLBagelPipeline"]
