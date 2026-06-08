from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable

import torch

from unirl.train.configs import EmaFullConfig, EmaLoraConfig
from unirl.train.fsdp_utils import local_view
from unirl.train.shadow import Shadow

logger = logging.getLogger(__name__)


@dataclass
class EMA:
    """Per-step shadow updater.  The only runtime class.

    Stateless — shadow values live on the model tree.  EMA just knows
    when to update (timing) and how fast (decay_fn).
    """

    shadow: Shadow
    decay_fn: Callable[[int], float]
    timing: str  # "optimizer_step" | "rollout_end"
    name: str = "ema"

    def step(self, t: int) -> None:
        if self.timing == "optimizer_step":
            self._run(self.decay_fn(t))

    def on_rollout_end(self, t: int) -> None:
        if self.timing == "rollout_end":
            self._run(self.decay_fn(t))

    @torch.no_grad()
    def _run(self, decay: float) -> None:
        if decay <= 0.0:
            for live, shd in self.shadow.iter_pairs():
                local_view(shd).copy_(local_view(live))
            return
        for live, shd in self.shadow.iter_pairs():
            local_shd = local_view(shd)
            local_shd.mul_(decay).add_(local_view(live), alpha=1.0 - decay)

    @contextmanager
    def use_shadow(self):
        """Swap shadow into live position for inference / export."""
        self.shadow.swap_in()
        try:
            yield
        finally:
            self.shadow.swap_out()

    def apply_shadow(self) -> None:
        """RPC-friendly swap-in (no context manager). Must be paired with
        :meth:`restore_shadow`."""
        self.shadow.swap_in()

    def restore_shadow(self) -> None:
        """RPC-friendly swap-out (restore live params after :meth:`apply_shadow`)."""
        self.shadow.swap_out()


def make_decay_fn(cfg: EmaLoraConfig | EmaFullConfig) -> Callable[[int], float]:
    """Build a ``t -> decay`` callable from an EMA config."""
    if isinstance(cfg, EmaFullConfig):
        target = float(cfg.target_decay)
        return lambda t: min((1 + t) / (10 + t), target)

    decay_type = str(cfg.ema_decay_type)
    ema_decay = float(cfg.ema_decay)
    flat_steps = int(cfg.ema_flat_steps)
    uprate = float(cfg.ema_uprate)
    uphold = float(cfg.ema_uphold)

    if decay_type == "linear":
        return lambda t: float(min(t * uprate, uphold))
    if decay_type == "warmup":
        return lambda t: 0.0 if t < flat_steps else float(min((t - flat_steps) * uprate, uphold))
    return lambda t: ema_decay


__all__ = ["EMA", "make_decay_fn"]
