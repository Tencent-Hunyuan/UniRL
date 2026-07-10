"""SD3 SFT task adapter — diffusion flow-matching MSE (text→image).

The diffusion counterpart to :class:`unirl.models.qwen3.sft_task.Qwen3SFTTask`:
same generic SFT skeleton, but the supervised loss is flow-matching velocity
MSE, not next-token cross-entropy. Together they prove the SFT domain is
model-agnostic (one skeleton, AR + diffusion).

Flow-matching SFT (mirrors how the SD3 RL/rollout path defines the field):

    x0    = VAE-encode(image)                       # clean latent
    sigma ~ logit-normal: sigmoid(N(0,1))            # SD3-default noise weighting
    x_t   = (1 - sigma) * x0 + sigma * noise         # forward interpolation
    v     = SD3DiffusionStep.predict_noise(x_t, ..)  # transformer predicts velocity
    loss  = MSE(v, noise - x0)                        # target velocity dx/dsigma

The velocity convention (target = ``noise - x0``, NOT ``noise``) matches
``FlowSDEStrategy.step``'s drift term ``noise_pred * (sigma_next - sigma)`` —
i.e. ``predict_noise`` outputs ``dx/dsigma``. Training draws ``sigma`` from a
logit-normal distribution (``sigmoid(z), z~N(0,1)``) — the SD3 default (Esser et
al. 2024) used by diffusers' ``train_dreambooth_lora_sd3``, concentrating
samples on the informative mid-noise band. Reuses:
:meth:`SD3Bundle.from_config`, :class:`SD3TextEmbedStage` (text→embeds),
:meth:`SD3DiffusionStep.predict_noise` (CFG-aware transformer forward, here with
``guidance_scale=1.0`` so only the conditional branch runs), and
:func:`unirl.sde.runtime.get_sigma_schedule` (the static SD3 ``shift=3.0`` grid,
used by :meth:`sample` for the few-step eval denoiser).
Image→latent VAE encode is written here (inverse of ``SD3VAEDecodeStage``).

Batch note: the SFT policy calls ``compute_loss`` per record (B = 1).

Record schema (one JSONL row): ``{"sample_id": str, "image_path": str, "prompt": str}``.
``image_path`` is relative to the manifest dir (``_root`` injected by the data source).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from unirl.models.sd3.bundle import SD3Bundle
from unirl.models.sd3.conditions import SD3Conditions
from unirl.models.sd3.config import SD3PipelineConfig
from unirl.models.sd3.diffusion import SD3DiffusionStep
from unirl.models.sd3.text_embed import SD3TextEmbedStage
from unirl.sde.runtime import get_sigma_schedule
from unirl.train.sft.task import SFTTaskBase
from unirl.types.primitives import Texts
from unirl.utils.dtypes import parse_torch_dtype

logger = logging.getLogger(__name__)


class SD3SFTTask(SFTTaskBase):
    """Text→image SFT for SD3.5: flow-matching velocity MSE."""

    block_class_names: Tuple[str, ...] = ("JointTransformerBlock",)

    def __init__(self, *, bundle: SD3Bundle, config: SD3PipelineConfig) -> None:
        self.bundle = bundle
        self.config = config
        self.text_embed = SD3TextEmbedStage(bundle)
        self.diffusion = SD3DiffusionStep()
        self.shift = float(getattr(config, "shift", 3.0))
        self.autocast_dtype = parse_torch_dtype(
            getattr(config, "autocast_precision", "bf16"), field_name="SD3SFTTask.autocast_precision"
        )

    @classmethod
    def from_config(cls, config: SD3PipelineConfig) -> "SD3SFTTask":
        return cls(bundle=SD3Bundle.from_config(config), config=config)

    # ------------------------------------------------------------------
    # Data loading (worker-side; the record carries paths, tensors load here)
    # ------------------------------------------------------------------

    def load_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        from PIL import Image as PILImage

        prompt = record.get("prompt")
        image_path = record.get("image_path")
        if not isinstance(prompt, str) or not isinstance(image_path, str):
            raise ValueError(
                f"SD3SFTTask record needs str 'prompt' and 'image_path'; "
                f"got prompt={type(prompt).__name__}, image_path={type(image_path).__name__}"
            )
        root = record.get("_root", "")
        img = PILImage.open(os.path.join(root, image_path)).convert("RGB")
        arr = torch.from_numpy(_pil_to_float01(img))  # [3, H, W] in [0, 1]
        loaded = dict(record)
        loaded["pixels"] = arr
        return loaded

    # ------------------------------------------------------------------
    # VAE encode (inverse of SD3VAEDecodeStage: decode does lat/scaling + shift)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_image(self, pixels: torch.Tensor) -> torch.Tensor:
        """``[3, H, W]`` in [0, 1] → SD3 model-space latent ``[1, C, H/8, W/8]``."""
        device = self.bundle.device
        vae_f32 = self.bundle.vae.to(torch.float32)
        x = pixels.to(device=device, dtype=torch.float32).unsqueeze(0)  # [1, 3, H, W]
        x = x * 2.0 - 1.0  # [0, 1] → [-1, 1]
        z = vae_f32.encode(x).latent_dist.mode()  # native VAE latent
        scaling_factor = vae_f32.config.scaling_factor
        shift_factor = getattr(vae_f32.config, "shift_factor", None)
        if shift_factor is not None:
            z = z - float(shift_factor)
        z = z * scaling_factor
        return z.to(dtype=self.bundle.dtype)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(
        self, loaded: Dict[str, Any], *, generator: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Flow-matching velocity MSE for ONE image/prompt pair."""
        device = self.bundle.device
        x0 = self._encode_image(loaded["pixels"])  # [1, C, h, w]

        conditions = SD3Conditions(text=self.text_embed.embed(Texts(texts=[loaded["prompt"]])))

        # Sample the training noise level from a logit-normal distribution:
        # sigma = sigmoid(z), z ~ N(0, 1). This is the SD3 default (Esser et al.
        # 2024 found logit-normal timestep weighting best across schemes) and
        # what diffusers' train_dreambooth_lora_sd3 uses. It concentrates samples
        # near the informative mid-noise band (sigma ~ 0.5), unlike a uniform
        # draw over the inference schedule (which is shift=3-biased toward high
        # sigma — an RL rollout-schedule convention, not an SFT training one).
        z = torch.randn(1, generator=generator, device=device, dtype=torch.float32)
        sigma = torch.sigmoid(z).view(1)

        noise = torch.randn(x0.shape, generator=generator, device=device, dtype=torch.float32)
        s = sigma.view(1, *([1] * (x0.ndim - 1)))
        x_t = (1.0 - s) * x0.float() + s * noise
        v_target = noise - x0.float()

        self.bundle.transformer.train()
        v_pred = self.diffusion.predict_noise(self.bundle, x_t, sigma, conditions, guidance_scale=1.0)
        loss = F.mse_loss(v_pred.float(), v_target)
        metrics = {"loss/total": float(loss.detach().item()), "train/sigma": float(sigma.item())}
        return loss, metrics

    # ------------------------------------------------------------------
    # Eval sampling — all FSDP ranks enter (collective weights); rank 0 kept.
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self, loaded: Dict[str, Any], *, generator: Optional[torch.Generator] = None
    ) -> Dict[str, Any]:
        # A few-step Euler flow-match sampler over the same schedule, decoded via
        # the VAE. Kept minimal — eval is a qualitative smoke, not the training
        # objective. Returns pixels [3, H, W] in [0, 1].
        device = self.bundle.device
        conditions = SD3Conditions(text=self.text_embed.embed(Texts(texts=[loaded["prompt"]])))
        x0_ref = self._encode_image(loaded["pixels"])
        num_steps = int(getattr(self.config, "sample_num_inference_steps", 20))
        sigmas = get_sigma_schedule(num_steps, shift=self.shift, device=device)
        x = torch.randn(x0_ref.shape, generator=generator, device=device, dtype=torch.float32)
        self.bundle.transformer.eval()
        for i in range(len(sigmas) - 1):
            sigma = sigmas[i].to(dtype=torch.float32).view(1)
            v = self.diffusion.predict_noise(self.bundle, x, sigma, conditions, guidance_scale=1.0).float()
            dt = (sigmas[i + 1] - sigmas[i]).to(dtype=torch.float32)
            x = x + v * dt  # Euler step; dt < 0 (sigma decreasing)
        pixels = self._decode_latent(x)
        self.bundle.transformer.train()
        return {"image": pixels}

    @torch.no_grad()
    def _decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        vae_f32 = self.bundle.vae.to(torch.float32)
        scaling_factor = vae_f32.config.scaling_factor
        shift_factor = getattr(vae_f32.config, "shift_factor", None)
        z = latent.to(torch.float32) / scaling_factor
        if shift_factor is not None:
            z = z + float(shift_factor)
        pixels = vae_f32.decode(z).sample  # [1, 3, H, W] in [-1, 1]
        pixels = (pixels.clamp(-1, 1) + 1.0) / 2.0
        return pixels[0]


def _pil_to_float01(img: Any):
    """PIL RGB → numpy ``[3, H, W]`` float32 in [0, 1] (no torchvision dep)."""
    import numpy as np

    arr = np.asarray(img, dtype="float32") / 255.0  # [H, W, 3]
    return arr.transpose(2, 0, 1).copy()


__all__ = ["SD3SFTTask"]
