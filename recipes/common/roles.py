"""Recipe-level Role abstractions for trainer-managed Remote roles."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional

import torch
from hydra.utils import instantiate

from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.distributed.group.remote import Remote
from unirl.reward.base import DifferentiableReward
from unirl.reward.service import RewardService
from unirl.types.primitives import Texts
from unirl.types.rollout_req import RolloutReq

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoleStepResult:
    """Generic result of one role-local optimizer step."""

    metrics: Mapping[str, object]
    grad_norm: float
    lr: float


class Role(Remote):
    """Recipe-level role that runs as a UniRL Remote inside a Worker.

    ``Role`` stores the role config and initializes common role-local
    components (model/runtime bundle / pipeline / backend) as ordinary
    Worker-local objects. Recipe roles may override ``initialize`` for
    recipe-specific config, but should call ``super().initialize()`` first.
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg = cfg

    def _cfg_get(self, key: str) -> Any:
        if hasattr(self.cfg, "get"):
            return self.cfg.get(key)
        return getattr(self.cfg, key, None)

    def initialize(self) -> None:
        """Initialize common role-local components after Worker setup."""
        model_cfg = self._cfg_get("model")
        bundle_cfg = model_cfg if model_cfg is not None else self._cfg_get("bundle")
        if bundle_cfg is not None:
            self.bundle = instantiate(bundle_cfg)

        pipeline_cfg = self._cfg_get("pipeline")
        if pipeline_cfg is not None:
            if hasattr(self, "bundle"):
                self.pipeline = instantiate(pipeline_cfg, bundle=self.bundle)
            else:
                self.pipeline = instantiate(pipeline_cfg)

        backend_cfg = self._cfg_get("backend")
        if backend_cfg is not None:
            if hasattr(self, "bundle"):
                if self.device is None or self.rank_info is None:
                    raise RuntimeError(
                        f"{type(self).__name__}.initialize called before Remote.setup injected device/rank_info."
                    )
                self.backend = instantiate(
                    backend_cfg,
                    bundle=self.bundle,
                    device=torch.device(self.device),
                    rank=int(self.rank_info.rank),
                )
            else:
                self.backend = instantiate(backend_cfg)

    @distributed
    def step(self) -> RoleStepResult:
        """Clip gradients and run one backend optimizer step."""
        if not hasattr(self, "backend") or not hasattr(self.backend, "optimizer_step"):
            raise RuntimeError(f"{type(self).__name__}.step requires a backend with optimizer_step(...).")
        grad_norm = float(self.backend.optimizer_step(max_grad_norm=float(self.algo_cfg.get("max_grad_norm", 1.0))))
        lr = 0.0
        try:
            sched = getattr(self.backend, "scheduler", None)
            if sched is not None:
                last = sched.get_last_lr()
                lr = float(last[0]) if last else 0.0
        except Exception:
            lr = 0.0
        return RoleStepResult(
            metrics={"grad_norm": grad_norm, "lr": lr},
            grad_norm=grad_norm,
            lr=lr,
        )

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def save_checkpoint(self, path: str, step: Optional[int] = None, mode: str = "auto") -> None:
        """Save backend checkpoint when the role backend supports it."""
        if hasattr(self, "backend") and hasattr(self.backend, "save"):
            self.backend.save(path, step=step, mode=mode)

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def load_checkpoint(self, path: str) -> int:
        """Load backend checkpoint when the role backend supports it."""
        if hasattr(self, "backend") and hasattr(self.backend, "load"):
            return int(self.backend.load(path) or 0)
        return 0


class RewardRole(RewardService):
    """Common differentiable reward role for recipe-local reward backends."""

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

        logger.info(
            "RewardRole initialized with backend=%s",
            self.backend.get_model_name() or type(self.backend).__name__,
        )

    @distributed
    def score_differentiable(self, *, req: RolloutReq, generated: Any) -> torch.Tensor:
        """Score live-grad decoded media through a differentiable reward backend."""
        decoded = generated.decoded
        if not isinstance(decoded, torch.Tensor):
            raise TypeError(
                f"RewardRole.score_differentiable: generated.decoded must be Tensor, "
                f"got {type(decoded).__name__}."
            )
        texts = req.primitives.get("text") if req.primitives else None
        if not isinstance(texts, Texts):
            raise TypeError("RewardRole.score_differentiable: req.primitives['text'] must be Texts.")

        if not isinstance(self.backend, DifferentiableReward):
            raise TypeError(
                f"{type(self.backend).__name__} must implement compute_rewards_differentiable "
                "for reward backprop."
            )

        records = list(req.metadata) if req.metadata else None
        kind = self.preferred_input_kind
        if kind == "image" and decoded.ndim != 4:
            raise ValueError(f"image backend expects [B,C,H,W], got {tuple(decoded.shape)}")
        if kind == "video" and decoded.ndim != 5:
            raise ValueError(f"video backend expects [B,C,T,H,W], got {tuple(decoded.shape)}")
        if kind not in {"image", "video"}:
            raise ValueError(f"RewardRole.score_differentiable: unsupported input_kind={kind!r}.")

        return self.backend.compute_rewards_differentiable(
            decoded,
            list(texts.texts),
            records=records,
        )


__all__ = ["Role", "RoleStepResult", "RewardRole"]
