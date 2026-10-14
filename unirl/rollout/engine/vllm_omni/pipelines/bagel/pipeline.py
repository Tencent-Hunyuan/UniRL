"""RL-aware BAGEL-7B-MoT pipeline subclass."""

from __future__ import annotations

import copy
import logging
from types import SimpleNamespace
from typing import Any, Dict, Optional

import PIL.Image
import torch
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.bagel.bagel_transformer import NaiveCache
from vllm_omni.diffusion.models.bagel.pipeline_bagel import BagelPipeline
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from unirl.models.bagel.rl_ops import (
    build_image_transforms,
    resize_input_image,
    update_context_image,
    update_context_text,
)
from unirl.rollout.engine.vllm_omni.pipelines._shared.interception import (
    drain_trajectory_into,
    finalize_output,
    resolve_request_noise,
    set_payload,
    single_request,
    stamp_capture,
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
        self._pending_initial_noise: Optional[torch.Tensor] = None
        self._pending_layout: Optional[Dict[str, Any]] = None
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
        rotary = self.bagel.language_model.model.rotary_emb
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

    def _install_noise_tap(self) -> None:
        """Wrap ``bagel.prepare_vae_latent`` to swap the driver-authored x_T in for upstream's RNG-drawn noises."""
        if self._noise_tap_installed:
            return

        orig_prepare_vae_latent = self.bagel.prepare_vae_latent
        pipeline = self

        def tapped(*args: Any, **kwargs: Any) -> Any:
            spp = pipeline._pending_spp
            if spp > 1:
                kwargs = dict(kwargs)
                kwargs["image_sizes"] = list(kwargs["image_sizes"]) * spp
                kwargs["curr_kvlens"] = list(kwargs["curr_kvlens"]) * spp
                kwargs["curr_rope"] = list(kwargs["curr_rope"]) * spp
            height, width = kwargs["image_sizes"][0]
            pipeline._pending_layout = {
                "kv_len": kwargs["curr_kvlens"][0],
                "rope": kwargs["curr_rope"][0],
                "height": height,
                "width": width,
            }
            out = orig_prepare_vae_latent(*args, **kwargs)
            noise = pipeline._pending_initial_noise
            if noise is not None:
                pipeline._pending_initial_noise = None
                ref_noise = out.get("packed_init_noises")
                if ref_noise is None:
                    raise RuntimeError(
                        "RLBagelPipeline noise tap: prepare_vae_latent returned no 'packed_init_noises' to override."
                    )
                if noise.dim() == ref_noise.dim() + 1:
                    noise = noise.reshape(-1, noise.shape[-1])
                if tuple(noise.shape) != tuple(ref_noise.shape):
                    raise RuntimeError(
                        "RLBagelPipeline noise tap: driver x_T shape "
                        f"{tuple(noise.shape)} != worker packed_init_noises shape "
                        f"{tuple(ref_noise.shape)} — check the recipe's "
                        "init_noise_latent_shape (bagel_latent_shape) vs the "
                        "request's height/width."
                    )
                out["packed_init_noises"] = noise.to(dtype=ref_noise.dtype, device=ref_noise.device)
            return out

        self.bagel.prepare_vae_latent = tapped  # type: ignore[assignment]
        self._noise_tap_installed = True

    def _install_generate_image_tap(self) -> None:
        """Wrap ``bagel.generate_image`` once for the grouped (spp>1) path."""
        if self._generate_image_tap_installed:
            return

        original_generate_image = self.bagel.generate_image
        pipeline = self

        def generate_image_grouped(*args: Any, **kwargs: Any) -> Any:
            spp = pipeline._pending_spp
            if spp > 1 and (past_kv := kwargs.get("past_key_values")) is not None:
                kwargs = {**kwargs, "past_key_values": NaiveCache.merge([past_kv] * spp)}
            outputs = original_generate_image(*args, **kwargs)
            if spp > 1:
                pipeline._pending_batched_latents = list(outputs[0])
            return outputs

        self.bagel.generate_image = generate_image_grouped  # type: ignore[assignment]
        self._generate_image_tap_installed = True

    def forward(self, req: DiffusionRequestBatch, **kwargs) -> DiffusionOutput:
        """Single-request batch in, single output out — see ``single_request``."""
        request = single_request(req, caller="RLBagelPipeline.forward")
        self._install_sde_scheduler()
        self._install_noise_tap()
        self._install_rope_fp32()
        self._pending_layout = None

        sampling = request.sampling_params
        image_shape = (sampling.height, sampling.width)
        prompt = request.prompt
        image = (prompt.get("multi_modal_data") or {}).get("image") if isinstance(prompt, dict) else None
        if image is not None:
            if not isinstance(image, PIL.Image.Image):
                raise TypeError(
                    "RLBagelPipeline: multi_modal_data['image'] must be ONE PIL image "
                    f"(BagelAdapter ships one per request); got {type(image).__name__}."
                )
            if self._vae_transform is None:
                self._vae_transform, self._vit_transform = build_image_transforms()
            bundle = SimpleNamespace(
                model=self.bagel,
                vae=self.vae,
                tokenizer=self.tokenizer,
                vae_transform=self._vae_transform,
                vit_transform=self._vit_transform,
                new_token_ids=self.new_token_ids,
                device=self.device,
            )
            clean_prompt = str(prompt.get("prompt") or "").removeprefix("<|im_start|>").removesuffix("<|im_end|>")
            gen_ctx = {
                "kv_lens": [0],
                "ropes": [0],
                "past_key_values": NaiveCache(self.bagel.config.llm_config.num_hidden_layers),
            }
            cfg_img_ctx = copy.deepcopy(gen_ctx)
            autocast = torch.autocast(
                device_type=self.device.type,
                enabled=self.device.type != "cpu",
                dtype=self.od_config.dtype,
            )
            with torch.no_grad(), autocast:
                resized = resize_input_image(bundle, image)
                gen_ctx = update_context_image(bundle, resized, gen_ctx, vae=True, vit=True)
                cfg_text_ctx = copy.deepcopy(gen_ctx)  # snapshot before the prompt text
                gen_ctx = update_context_text(bundle, clean_prompt, gen_ctx)
                cfg_img_ctx = update_context_text(bundle, clean_prompt, cfg_img_ctx)

            sampling = copy.copy(sampling)
            sampling.past_key_values = gen_ctx["past_key_values"]
            sampling.kv_metadata = {"ropes": gen_ctx["ropes"], "image_shape": image_shape}
            sampling.cfg_text_past_key_values = cfg_text_ctx["past_key_values"]
            sampling.cfg_text_kv_metadata = {"ropes": cfg_text_ctx["ropes"]}
            sampling.cfg_img_past_key_values = cfg_img_ctx["past_key_values"]
            sampling.cfg_img_kv_metadata = {"ropes": cfg_img_ctx["ropes"]}
            request.sampling_params = sampling

        extra_args = sampling.extra_args
        spp = sampling.num_outputs_per_prompt
        image_token_sizes = None
        if spp > 1:
            is_text_output = isinstance(prompt, dict) and "text" in (prompt.get("modalities") or [])
            is_batchable = (
                not is_text_output
                and "cfg_text_scale" in extra_args
                and "cfg_img_scale" in extra_args
                and extra_args["cfg_text_scale"] <= 1.0
                and extra_args["cfg_img_scale"] <= 1.0
            )
            if not is_batchable:
                raise RuntimeError(
                    f"RLBagelPipeline: num_outputs_per_prompt={spp} requires image output "
                    f"with cfg_text_scale<=1 and cfg_img_scale<=1 present in "
                    f"sampling_params.extra_args. BagelAdapter should leave "
                    f"num_outputs_per_prompt=1 (sample-level layout) when packing "
                    f"is disabled."
                )
            self._install_generate_image_tap()
            downsample = self.bagel.latent_downsample
            tokens_per_image = (sampling.height // downsample) * (sampling.width // downsample)
            image_token_sizes = [tokens_per_image] * spp

        traj_dtype_name = extra_args.get("trajectory_precision")
        traj_dtype = (
            parse_torch_dtype(traj_dtype_name, field_name="trajectory_precision")
            if traj_dtype_name
            else self._trajectory_dtype
        )
        self._sde_scheduler.set_for_request(
            eta=sampling.eta,
            sde_indices=extra_args.get("sde_indices"),
            sigma_max=extra_args.get("sigma_max"),
            trajectory_dtype=traj_dtype,
            image_token_sizes=image_token_sizes,
        )
        self._pending_initial_noise = resolve_request_noise(request, caller="RLBagelPipeline.forward")
        self._pending_spp = spp
        self._pending_batched_latents = None
        try:
            out = super().forward(req, **kwargs)
            drain_trajectory_into(out, self._sde_scheduler)
            stamp_capture(out, "layout", self._pending_layout)
            finalize_output(out)
            if spp > 1:
                latents = self._pending_batched_latents
                if not latents or len(latents) != spp:
                    raise RuntimeError(
                        f"RLBagelPipeline batched forward: generate_image tap captured "
                        f"{0 if not latents else len(latents)} latents, expected spp={spp}."
                    )
                decoded_tail = [
                    self._decode_image_from_latent(self.bagel, self.vae, lat, image_shape) for lat in latents[1:]
                ]
                set_payload(out, [out.output, *decoded_tail])
        finally:
            self._pending_spp = 1
            self._pending_batched_latents = None
        return out


__all__ = ["RLBagelPipeline"]
