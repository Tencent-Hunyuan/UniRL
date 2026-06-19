"""EMA feature: shadow structure injection + runtime shadow updates.

Owns the whole story of "a shadow copy of the trainable weights":

* :func:`inject_nft` — dual LoRA adapters (trainable ``default`` + frozen
  ``old`` shadow) for NFT-style adapter EMA; build-time, returns a
  :class:`Shadow`.
* :func:`inject_mirror` — full-model ``shadow_*`` parameters for full-weight
  EMA; build-time, returns a :class:`Shadow`.
* :class:`Shadow` — how to access (live, shadow) parameter pairs, regardless
  of whether shadows are peft adapters or mirror parameters.
* :class:`EMA` — the runtime updater: when to update (timing) and how fast
  (``make_decay_fn``), plus swap-in/swap-out for rollout/export.

Injection runs in the backend constructor while the model may still be on
meta device; post-materialize work is stamped via ``unirl.train.deferred``.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Callable, Iterator, List, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn.parameter import Parameter

from unirl.train.configs import EmaFullConfig, EmaLoraConfig
from unirl.train.deferred import _stamp
from unirl.train.lora import _reset_adapter

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Shadow handle
# ------------------------------------------------------------------


@dataclass(frozen=True)
class Shadow:
    """How to access (live, shadow) parameter pairs on the model tree.

    Returned by :func:`inject_nft` / :func:`inject_mirror`.  Consumed by
    :class:`EMA`.  The closures capture the model reference and the
    specifics of how shadows are stored.  EMA calls these without knowing
    whether shadows are peft adapters or mirror parameters.
    """

    iter_pairs: Callable[[], Iterator[Tuple[Tensor, Tensor]]]
    swap_in: Callable[[], None]
    swap_out: Callable[[], None]


# ------------------------------------------------------------------
# Runtime updater
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# inject_nft — returns Shadow handle
# ------------------------------------------------------------------


def inject_nft(
    model: nn.Module,
    *,
    rank: int,
    alpha: int,
    target_modules: Sequence[str],
    default: str = "default",
    shadow: str = "old",
    dropout: float = 0.0,
    bias: str = "none",
    task_type: str = "FEATURE_EXTRACTION",
) -> Shadow:
    """Inject dual LoRA adapters for NFT-style EMA.  Returns Shadow."""
    from peft import LoraConfig, inject_adapter_in_model

    peft_cfg = LoraConfig(
        r=int(rank),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        target_modules=list(target_modules),
        bias=str(bias),
        task_type=str(task_type),
    )
    inject_adapter_in_model(peft_cfg, model, adapter_name=default)
    inject_adapter_in_model(peft_cfg, model, adapter_name=shadow)

    # peft's inject_adapter_in_model installs the LoRA layers but does not flip
    # diffusers' PeftAdapterMixin `_hf_peft_config_loaded` flag, so the model-level
    # `set_adapter` raises "No adapter loaded". Activate `default` the same
    # per-LoraLayer way swap_out does (works for diffusers + plain modules), and
    # mark the flag so downstream diffusers adapter ops stay consistent.
    if hasattr(model, "_hf_peft_config_loaded"):
        model._hf_peft_config_loaded = True
    _activate_keep_grad(model, default, trainable=default, frozen=shadow)

    if _current_rank() == 0:
        n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
        logger.info(
            "inject_nft: adapters %r + %r (rank=%d, alpha=%d) — %d trainable params",
            default,
            shadow,
            rank,
            alpha,
            n_trainable,
        )

    _stamp(model, partial(_reset_adapter, name=default))
    _stamp(model, partial(_reset_adapter, name=shadow))
    _stamp(model, partial(_copy_adapter, src=default, dst=shadow))

    return Shadow(
        iter_pairs=lambda: _adapter_pairs(model, default, shadow),
        swap_in=lambda: _activate_keep_grad(model, shadow, trainable=default, frozen=shadow),
        swap_out=lambda: _activate_keep_grad(model, default, trainable=default, frozen=shadow),
    )


# ------------------------------------------------------------------
# inject_mirror — returns Shadow handle
# ------------------------------------------------------------------


def inject_mirror(
    model: nn.Module,
    *,
    prefix: str = "shadow_",
) -> Shadow:
    """Register shadow_* parameters for full-model EMA.  Returns Shadow."""
    pairs: List[Tuple[nn.Module, str, str]] = []

    for fqn, p in list(model.named_parameters()):
        if not p.requires_grad:
            continue
        parent, attr = _parent_and_attr(model, fqn)
        shadow_attr = prefix + attr
        shadow_param = Parameter(torch.empty_like(p), requires_grad=False)
        parent.register_parameter(shadow_attr, shadow_param)
        pairs.append((parent, attr, shadow_attr))

    if _current_rank() == 0:
        logger.info("inject_mirror: registered %d shadow parameters (prefix=%r)", len(pairs), prefix)

    _stamp(model, partial(_copy_mirror, pairs=pairs))

    return Shadow(
        iter_pairs=lambda: ((getattr(m, a), getattr(m, s)) for m, a, s in pairs),
        swap_in=lambda: _swap_mirror(pairs),
        swap_out=lambda: _swap_mirror(pairs),
    )


# ------------------------------------------------------------------
# Peft helpers
# ------------------------------------------------------------------


def _set_adapter_requires_grad(model: nn.Module, name: str, requires_grad: bool) -> None:
    from peft.tuners.lora import LoraLayer

    for m in model.modules():
        if not isinstance(m, LoraLayer):
            continue
        for key in ("lora_A", "lora_B"):
            bank = getattr(m, key, {})
            if name in bank:
                bank[name].weight.requires_grad = requires_grad


def _copy_adapter(model: nn.Module, *, src: str, dst: str) -> None:
    from peft.tuners.lora import LoraLayer

    n_copied = 0
    for m in model.modules():
        if not isinstance(m, LoraLayer):
            continue
        for key in ("lora_A", "lora_B"):
            bank = getattr(m, key, {})
            if src in bank and dst in bank:
                for sp, dp in zip(bank[src].parameters(), bank[dst].parameters()):
                    dp.data.copy_(sp.data)
                n_copied += 1
    if n_copied == 0:
        raise RuntimeError(f"_copy_adapter: no adapter pairs found for {src!r} -> {dst!r}")


def _adapter_pairs(
    model: nn.Module,
    default: str,
    shadow: str,
) -> list[Tuple[torch.Tensor, torch.Tensor]]:
    from peft.tuners.lora import LoraLayer

    pairs: list[Tuple[torch.Tensor, torch.Tensor]] = []
    for m in model.modules():
        if not isinstance(m, LoraLayer):
            continue
        for key in ("lora_A", "lora_B"):
            bank = getattr(m, key, {})
            if default in bank and shadow in bank:
                for sp, dp in zip(bank[default].parameters(), bank[shadow].parameters()):
                    pairs.append((sp, dp))
    return pairs


def _activate(model: nn.Module, adapter_name: str) -> None:
    from peft.tuners.lora import LoraLayer

    for m in model.modules():
        if isinstance(m, LoraLayer):
            m.set_adapter(adapter_name)


def _activate_keep_grad(model: nn.Module, active: str, *, trainable: str, frozen: str) -> None:
    """Switch the active adapter, then RESTORE the canonical requires_grad split.

    peft's ``set_adapter`` couples the active adapter with ``requires_grad``
    (active -> True, all others -> False). DiffusionNFT swaps to the frozen
    ``old`` adapter for off-policy rollout, which flips the trainable ``default``
    adapter to ``requires_grad=False``. Under FSDP2 with ``reshard_after_forward
    =False`` the rollout's all-gather then caches the unsharded ``default`` param
    grad-less, and the later training forward yields a loss with no ``grad_fn``
    ("element 0 ... does not require grad"). Re-asserting the split keeps the
    trainable adapter grad-enabled no matter which adapter is active, so the
    cached compute param is always gathered with ``requires_grad=True``."""
    _activate(model, active)
    _set_adapter_requires_grad(model, trainable, True)
    _set_adapter_requires_grad(model, frozen, False)


# ------------------------------------------------------------------
# Mirror helpers
# ------------------------------------------------------------------


def _copy_mirror(model: nn.Module, *, pairs: List[Tuple[nn.Module, str, str]]) -> None:
    for mod, live_attr, shadow_attr in pairs:
        getattr(mod, shadow_attr).data.copy_(getattr(mod, live_attr).data)


def _swap_mirror(pairs: List[Tuple[nn.Module, str, str]]) -> None:
    for mod, live_attr, shadow_attr in pairs:
        live = getattr(mod, live_attr)
        shd = getattr(mod, shadow_attr)
        live.data, shd.data = shd.data, live.data


# ------------------------------------------------------------------
# General helpers
# ------------------------------------------------------------------


def local_view(tensor: Tensor) -> Tensor:
    """DTensor -> local shard.  Identity for non-DTensors."""
    if hasattr(tensor, "_local_tensor"):
        return tensor._local_tensor
    return tensor


def _parent_and_attr(model: nn.Module, fqn: str) -> Tuple[nn.Module, str]:
    parts = fqn.rsplit(".", 1)
    if len(parts) == 1:
        return model, parts[0]
    parent = model
    for part in parts[0].split("."):
        parent = getattr(parent, part)
    return parent, parts[1]


def _current_rank() -> int:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return 0


__all__ = ["EMA", "make_decay_fn", "Shadow", "inject_nft", "inject_mirror"]
