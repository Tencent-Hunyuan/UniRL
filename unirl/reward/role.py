"""Reward role abstraction for trainer-managed reward remotes."""

from __future__ import annotations

import logging
import torch
from typing import Any

from hydra.utils import instantiate

from unirl.distributed.group.remote import Remote
from unirl.distributed.group.dispatch import distributed
from unirl.reward.service import RewardService
from unirl.types.rollout_req import RolloutReq
from unirl.types.primitives import Texts
from unirl.reward.base import DifferentiableReward

logger = logging.getLogger(__name__)


class RewardRole(RewardService):
    """A trainer role that owns one reward backend.

    ``RewardService`` normally receives an already-instantiated backend in its
    constructor. Trainer-managed roles are constructed with a role config first,
    then ``initialize()`` runs after ``Remote.setup`` injects the worker device.
    This class bridges those lifecycles while preserving ``RewardService``'s
    scoring and offload/onload/dispose methods.
    """

    def __init__(self, cfg: Any) -> None:
        Remote.__init__(self)
        self.cfg = cfg

    def _cfg_get(self, key: str, default: Any = None) -> Any:
        if hasattr(self.cfg, "get"):
            return self.cfg.get(key, default)
        return getattr(self.cfg, key, default)

    def initialize(self) -> None:
        if self.device is None:
            raise RuntimeError(f"{type(self).__name__}.initialize called before Remote.setup injected device.")

        backend_cfg = self._cfg_get("backend")
        if backend_cfg is None:
            raise ValueError(f"{type(self).__name__} requires cfg.backend.")

        self.backend = instantiate(backend_cfg, base_device=self.device)
        self.truncated_reward = str(self._cfg_get("truncated_reward", "zero"))
        self.overlong_buffer_len = int(self._cfg_get("overlong_buffer_len", 4096))
        self.overlong_penalty_factor = float(self._cfg_get("overlong_penalty_factor", 1.0))
        if self.truncated_reward not in ("zero", "keep", "soft"):
            raise ValueError(f"truncated_reward must be zero|keep|soft, got {self.truncated_reward!r}")

        if hasattr(self.backend, "initialize"):
            self.backend.initialize()

        logger.info(
            "RewardRole initialized with backend=%s, truncated_reward=%s",
            self.backend.get_model_name() or type(self.backend).__name__,
            self.truncated_reward,
        )

    @distributed
    def score_differentiable(self, *, req: RolloutReq, generated: Any) -> torch.Tensor:
        """Score live-grad decoded media through the official differentiable reward path."""
        decoded = generated.decoded
        if not isinstance(decoded, torch.Tensor):
            raise TypeError(
                f"ReflRewardRole.score_differentiable: generated.decoded must be Tensor, "
                f"got {type(decoded).__name__}."
            )
        texts = req.primitives.get("text") if req.primitives else None
        if not isinstance(texts, Texts):
            raise TypeError("ReflRewardRole.score_differentiable: req.primitives['text'] must be Texts.")

        if not isinstance(self.backend, DifferentiableReward):
            raise TypeError(
                f"{type(self.backend).__name__} must implement compute_rewards_differentiable "
                "for REFL reward backprop."
            )

        records = list(req.metadata) if req.metadata else None
        kind = self.preferred_input_kind
        if kind == "image" and decoded.ndim != 4:
            raise ValueError(f"image backend expects [B,C,H,W], got {tuple(decoded.shape)}")
        if kind == "video" and decoded.ndim != 5:
            raise ValueError(f"video backend expects [B,C,T,H,W], got {tuple(decoded.shape)}")
        if kind not in {"image", "video"}:
            raise ValueError(f"ReflRewardRole.score_differentiable: unsupported input_kind={kind!r}.")

        return self.backend.compute_rewards_differentiable(
            decoded,
            list(texts.texts),
            records=records,
        )


__all__ = ["RewardRole"]
