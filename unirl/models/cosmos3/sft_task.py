"""Cosmos3 SFT task adapters.

Two tasks over one packed-forward core:

- :class:`Cosmos3VideoSFTTask` — text(+first-frame)-conditioned video/image
  flow-matching SFT (t2i via 1-frame clips, t2v, video prediction).
- :class:`Cosmos3ActionBCTask` — policy-mode behavior cloning: latent frame 0
  = clean observation, future video frames + the whole action chunk are
  noised and jointly denoised (the pretraining objective of
  Cosmos3-Nano-Policy-DROID).

A task owns the Cosmos3-specific pieces (bundle, tokenization, packing, loss);
the generic SFT loop (:mod:`unirl.train.sft`) only calls
``from_config`` / ``load_record`` / ``compute_loss`` / ``sample``.

Dataset records (one JSONL row each, see ``unirl/utils/prepare_droid100.py``)::

    {"sample_id": ..., "instruction": str, "frames_path": "frames/x.pt",
     "actions_path": "actions/x.pt" | null, "fps": float}

``frames_path`` -> uint8 ``[T, 3, H, W]``; ``actions_path`` -> float32
``[chunk, D_raw]`` (already normalized by the prep script). H/W should be an
exact Cosmos3 resolution bin so training and tier-based action inference see
the same canvas (the prep script guarantees this).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from unirl.models.cosmos3.bundle import Cosmos3Bundle
from unirl.models.cosmos3.config import Cosmos3SFTConfig
from unirl.models.cosmos3.packing import (
    noise_action_latents,
    noise_vision_latents,
    pack_joint_sequence,
    pad_action_chunk,
    sample_train_sigma,
)

logger = logging.getLogger(__name__)


class Cosmos3SFTTaskBase:
    """Shared packed-forward core; subclasses fix the task mode."""

    block_class_names: Tuple[str, ...] = ("Cosmos3VLTextMoTDecoderLayer",)
    train_action: bool = False
    action_mode: Optional[str] = None  # tokenize_prompt template selector

    def __init__(self, *, bundle: Cosmos3Bundle, config: Cosmos3SFTConfig) -> None:
        self.bundle = bundle
        self.config = config
        self.pipe = bundle.build_pipeline()
        self._token_cache: Dict[Any, list] = {}

    @classmethod
    def from_config(cls, config: Cosmos3SFTConfig) -> "Cosmos3SFTTaskBase":
        return cls(bundle=Cosmos3Bundle.from_config(config), config=config)

    # ------------------------------------------------------------------
    # Data loading (worker-side; records carry paths, tensors load here)
    # ------------------------------------------------------------------

    def load_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        root = record.get("_root", "")
        loaded = dict(record)
        frames = torch.load(os.path.join(root, record["frames_path"]), weights_only=True)
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError(f"frames must be [T, 3, H, W], got {tuple(frames.shape)}")
        loaded["frames"] = frames
        if self.train_action:
            if not record.get("actions_path"):
                raise ValueError(f"record {record.get('sample_id')} lacks actions_path for action BC")
            actions = torch.load(os.path.join(root, record["actions_path"]), weights_only=True)
            chunk = self.config.action_chunk_size
            if actions.shape != (chunk, self.config.raw_action_dim):
                raise ValueError(
                    f"actions must be [{chunk}, {self.config.raw_action_dim}], got {tuple(actions.shape)}"
                )
            if frames.shape[0] != chunk + 1:
                raise ValueError(
                    f"policy BC pairs {chunk} actions with {chunk + 1} frames, got {frames.shape[0]} frames"
                )
            loaded["actions"] = actions
        return loaded

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _tokenize(self, prompt: str, *, num_frames: int, height: int, width: int, fps: float) -> list:
        key = (prompt, num_frames, height, width, round(float(fps), 3))
        ids = self._token_cache.get(key)
        if ids is None:
            ids, _ = self.pipe.tokenize_prompt(
                prompt,
                None,
                num_frames=num_frames,
                height=height,
                width=width,
                fps=fps,
                use_system_prompt=self.config.use_system_prompt,
                add_resolution_template=self.config.add_resolution_template,
                add_duration_template=self.config.add_duration_template,
                action_mode=self.action_mode,
                action_view_point=self.config.action_view_point if self.action_mode else None,
            )
            self._token_cache[key] = ids
        return ids

    def compute_loss(
        self, record: Dict[str, Any], *, generator: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """One packed forward + masked flow-matching MSE for ONE sample."""
        cfg = self.config
        device = torch.device(cfg.device)
        frames = record["frames"]
        num_frames, _, height, width = frames.shape
        fps = float(record.get("fps", cfg.fps))

        pixels = frames.to(device=device, dtype=torch.float32).div(127.5).sub(1.0)
        pixels = pixels.permute(1, 0, 2, 3).unsqueeze(0)  # [1, 3, T, H, W]
        with torch.no_grad():
            x0 = self.pipe._encode_video(pixels)  # [1, C, T_lat, H_lat, W_lat], float32

        condition_frames = [0] if (cfg.condition_on_first_frame and x0.shape[2] > 1) else []
        sigma = sample_train_sigma(
            time_dist=cfg.time_dist,
            logitnormal_mean=cfg.logitnormal_mean,
            logitnormal_std=cfg.logitnormal_std,
            shift=self.bundle.flow_shift,
            generator=generator,
            device=device,
        )
        x_t, v_target = noise_vision_latents(x0, sigma, condition_frames, generator)

        action_kwargs: Dict[str, Any] = {}
        v_target_action: Optional[torch.Tensor] = None
        if self.train_action:
            from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import _EMBODIMENT_TO_DOMAIN_ID

            x0_action = pad_action_chunk(
                record["actions"].to(device=device, dtype=torch.float32),
                int(self.bundle.transformer.config.action_dim),
            )
            x_t_action, v_target_action = noise_action_latents(
                x0_action, sigma, cfg.raw_action_dim, generator
            )
            action_kwargs = {
                "action_tokens": x_t_action,
                "action_condition_frame_indexes": (),  # policy BC: fully noisy chunk
                "action_domain_id": torch.tensor(
                    [_EMBODIMENT_TO_DOMAIN_ID[cfg.action_domain_name]], dtype=torch.long, device=device
                ),
                "action_fps": fps,
            }

        input_ids = self._tokenize(record["instruction"], num_frames=num_frames, height=height, width=width, fps=fps)
        kwargs, meta = pack_joint_sequence(
            self.pipe,
            input_ids=input_ids,
            vision_tokens=x_t,
            condition_frame_indexes=condition_frames,
            vision_fps=fps,
            device=device,
            **action_kwargs,
        )
        timestep = float(sigma.item()) * float(self.pipe.scheduler.config.num_train_timesteps)
        kwargs["vision_timesteps"] = torch.full((meta["num_noisy_vision_tokens"],), timestep, device=device)
        if self.train_action:
            kwargs["action_timesteps"] = torch.full((meta["num_noisy_action_tokens"],), timestep, device=device)

        preds_vision, _preds_sound, preds_action = self.bundle.transformer(**kwargs)

        pred_v = preds_vision[0]
        if pred_v.dim() == 4:  # [C, T_lat, H, W] -> [1, C, T_lat, H, W]
            pred_v = pred_v.unsqueeze(0)
        noisy_frames = meta["vision_noisy_frames"]
        vision_loss = F.mse_loss(pred_v[:, :, noisy_frames].float(), v_target[:, :, noisy_frames])
        loss = cfg.vision_loss_weight * vision_loss
        metrics = {
            "loss/vision": float(vision_loss.detach().item()),
            "train/sigma": float(sigma.item()),
        }

        if self.train_action:
            pred_a = preds_action[0]
            if pred_a.dim() == 3:  # [1, T, D] -> [T, D]
                pred_a = pred_a.squeeze(0)
            raw = cfg.raw_action_dim
            action_loss = F.mse_loss(pred_a[:, :raw].float(), v_target_action[:, :raw])
            loss = loss + cfg.action_loss_weight * action_loss
            metrics["loss/action"] = float(action_loss.detach().item())

        metrics["loss/total"] = float(loss.detach().item())
        return loss, metrics

    # ------------------------------------------------------------------
    # Eval sampling (all FSDP ranks must enter together — collective weights)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(self, record: Dict[str, Any], *, generator: Optional[torch.Generator] = None) -> Dict[str, Any]:
        from PIL import Image as PILImage

        cfg = self.config
        frames = record["frames"]
        num_frames, _, height, width = frames.shape
        first_frame = PILImage.fromarray(frames[0].permute(1, 2, 0).numpy())
        common = {
            "num_inference_steps": cfg.sample_num_inference_steps,
            "output_type": "pt",
            "generator": generator,
            "enable_safety_check": False,
            "use_system_prompt": cfg.use_system_prompt,
        }
        if self.train_action:
            from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import CosmosActionCondition

            # NB: CosmosActionCondition resolves the canonical embodiment width
            # (droid_lerobot -> 10); when the BC data uses a narrower layout the
            # extra predicted columns are just the model's zero-padding.
            condition = CosmosActionCondition(
                mode="policy",
                chunk_size=cfg.action_chunk_size,
                domain_name=cfg.action_domain_name,
                image=first_frame,
                view_point=cfg.action_view_point,
            )
            out = self.pipe(
                prompt=record["instruction"], action=condition, guidance_scale=1.0, fps=record.get("fps", cfg.fps), **common
            )
            action = out.action[0] if out.action else None
            return {"video": out.video.cpu(), "action": action}
        out = self.pipe(
            prompt=record["instruction"],
            image=first_frame if cfg.condition_on_first_frame else None,
            num_frames=num_frames,
            height=height,
            width=width,
            fps=record.get("fps", cfg.fps),
            guidance_scale=cfg.sample_guidance_scale,
            **common,
        )
        return {"video": out.video.cpu()}


class Cosmos3VideoSFTTask(Cosmos3SFTTaskBase):
    """Text(+first-frame)-conditioned video/image flow-matching SFT."""

    train_action = False
    action_mode = None


class Cosmos3ActionBCTask(Cosmos3SFTTaskBase):
    """Policy-mode behavior cloning: obs frame 0 + instruction -> action chunk
    (+ co-denoised future video, the pretraining objective)."""

    train_action = True
    action_mode = "policy"


__all__ = ["Cosmos3ActionBCTask", "Cosmos3SFTTaskBase", "Cosmos3VideoSFTTask"]
