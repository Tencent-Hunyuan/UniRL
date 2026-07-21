"""Remote roles for the WAN22 REFL recipe."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, List, Optional

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from recipes.common.roles import Role
from unirl.distributed.group.dispatch import distributed
from unirl.distributed.tensor.batch import Batch, concat_field, shared_field
from unirl.sde.runtime import get_sigma_schedule
from unirl.types.primitives import Images, Texts
from unirl.types.rollout_req import RolloutReq


@dataclass
class REFLGenerated(Batch):
    """Generated BPTT payload carrying live-grad decoded pixels and KL loss."""

    decoded: torch.Tensor = concat_field(default_factory=lambda: torch.empty(0))
    kl_loss: torch.Tensor = shared_field(default_factory=lambda: torch.zeros(1))


@dataclass
class REFLLossMetrics(Batch):
    """Per-DP-shard REFL scalar metrics."""

    loss: List[float] = concat_field(default_factory=list)
    reward_loss: List[float] = concat_field(default_factory=list)
    kl_loss: List[float] = concat_field(default_factory=list)
    reward_mean: List[float] = concat_field(default_factory=list)


def _maybe_instantiate(value: Any) -> Any:
    if OmegaConf.is_config(value) and value.get("_target_") is not None:
        return instantiate(value)
    return value


class ReflActorRole(Role):
    """Actor role: bundle + pipeline + backend + REFL BPTT logic."""

    bundle: Any
    pipeline: Any
    backend: Any
    algo_cfg: Any
    sampling_params: Any

    def initialize(self) -> None:
        super().initialize()
        self.algo_cfg = self.cfg.algorithm
        self.sampling_params = _maybe_instantiate(self.algo_cfg.get("sampling_params"))

    @distributed
    def generate_samples(self, req: RolloutReq) -> REFLGenerated:
        """Run live-grad diffusion sampling and VAE decode."""
        stage = self.pipeline.diffusion
        decode_stage = self.pipeline.vae_decode
        if not hasattr(stage, "diffuse_with_grad"):
            raise RuntimeError("ReflActorRole: pipeline.diffusion lacks diffuse_with_grad(...).")
        if not hasattr(decode_stage, "decode_with_grad"):
            raise RuntimeError("ReflActorRole: pipeline.vae_decode lacks decode_with_grad(...).")

        texts = req.primitives.get("text") if req.primitives else None
        if not isinstance(texts, Texts):
            raise TypeError(
                f"ReflActorRole.generate_samples: req.primitives['text'] must be Texts, "
                f"got {type(texts).__name__ if texts is not None else 'None'}"
            )
        negatives_raw = req.primitives.get("negative_text") if req.primitives else None
        negatives = negatives_raw if isinstance(negatives_raw, Texts) else None
        if negatives is not None and len(negatives.texts) != len(texts.texts):
            raise ValueError(
                f"ReflActorRole.generate_samples: negative_text length {len(negatives.texts)} "
                f"!= text length {len(texts.texts)}"
            )

        params = self.sampling_params
        primary_g = float(getattr(params, "guidance_scale", 1.0))
        secondary_g = getattr(params, "guidance_scale_2", None)
        effective_guidance = max(primary_g, float(secondary_g)) if secondary_g is not None else primary_g
        conditions = self.pipeline.build_conditions(texts, negatives=negatives, guidance_scale=effective_guidance)

        images_prim = req.primitives.get("image") if req.primitives else None
        if images_prim is not None:
            if not isinstance(images_prim, Images):
                raise TypeError(
                    f"ReflActorRole.generate_samples: req.primitives['image'] must be Images, "
                    f"got {type(images_prim).__name__}"
                )
            if int(images_prim.pixels.shape[0]) != len(texts.texts):
                raise ValueError(
                    f"ReflActorRole.generate_samples: image count {images_prim.pixels.shape[0]} "
                    f"!= text count {len(texts.texts)}"
                )
            from unirl.models.wan21.clip_vision_encode import WAN21CLIPVisionEncodeStage
            from unirl.models.wan21.image_encode import WAN21ImageLatentEncodeStage

            image_latent_cond = WAN21ImageLatentEncodeStage(
                self.pipeline.bundle,
                num_frames=int(params.num_frames),
                height=int(params.height),
                width=int(params.width),
            ).encode(images_prim)
            image_embed_cond = (
                WAN21CLIPVisionEncodeStage(self.pipeline.bundle).encode(images_prim)
                if getattr(self.pipeline.bundle, "uses_clip_vision", False)
                else None
            )
            if image_latent_cond is not None or image_embed_cond is not None:
                conditions = dataclasses.replace(
                    conditions,
                    image_latent=image_latent_cond,
                    image_embed=image_embed_cond,
                )

        device = getattr(getattr(self.pipeline, "bundle", None), "device", None)
        schedule = get_sigma_schedule(
            int(params.num_inference_steps),
            shift=float(getattr(self.pipeline, "shift", 5.0)),
            device=device,
        )

        train_model = getattr(self.backend, "model", None)
        if train_model is not None and hasattr(train_model, "train"):
            train_model.train()
        self.backend.zero_grad()

        if bool(getattr(params, "init_same_noise", False)) and not getattr(params, "noise_group_ids", None):
            params = dataclasses.replace(params, noise_group_ids=list(req.group_ids))

        result = stage.diffuse_with_grad(
            conditions,
            schedule=schedule,
            params=params,
        )
        kl_loss = result.kl_loss
        pixels = decode_stage.decode_with_grad(result.z_final)
        return REFLGenerated(decoded=pixels, kl_loss=kl_loss.unsqueeze(0) if kl_loss.ndim == 0 else kl_loss)

    @distributed
    def forward_backward_loss(
        self,
        *,
        rewards: torch.Tensor,
        kl_loss: Optional[torch.Tensor] = None,
    ) -> REFLLossMetrics:
        """Assemble REFL loss and run backward on the actor graph."""
        algo = self.algo_cfg
        reward_weight = float(algo.get("reward_weight", 1.0))
        reward_baseline = float(algo.get("reward_baseline", 0.0))
        reward_scale = float(algo.get("reward_scale", 1.0))
        kl_weight = float(algo.get("kl_weight", 0.0))

        reward = rewards.to(dtype=torch.bfloat16)
        reward_loss = (-(reward - reward_baseline) / reward_scale * reward_weight).mean()
        if kl_loss is not None and kl_weight != 0.0:
            kl_term = kl_weight * kl_loss.squeeze()
        else:
            kl_term = torch.zeros((), device=reward.device, dtype=reward_loss.dtype)
        loss = reward_loss + kl_term
        loss.backward()
        return REFLLossMetrics(
            loss=[float(loss.detach().item())],
            reward_loss=[float(reward_loss.detach().item())],
            kl_loss=[float(kl_term.detach().item())],
            reward_mean=[float(reward.detach().mean().item())],
        )


__all__ = ["ReflActorRole"]
