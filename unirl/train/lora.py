"""Plain LoRA adapter injection.

Build-time structural mutation only: :func:`inject_lora` installs a single
peft adapter on the trainable stage and stamps the post-materialize reset via
``unirl.train.deferred``.  No Shadow, no EMA — the dual-adapter NFT variant
lives in ``unirl.train.ema``.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Sequence

from torch import nn

from unirl.train.deferred import _stamp

logger = logging.getLogger(__name__)


def inject_lora(
    model: nn.Module,
    *,
    rank: int,
    alpha: int,
    target_modules: Sequence[str],
    dropout: float = 0.0,
    bias: str = "none",
    task_type: str = "FEATURE_EXTRACTION",
    adapter_name: str = "default",
) -> None:
    """Inject a single LoRA adapter.  No Shadow, no EMA."""
    from peft import LoraConfig, inject_adapter_in_model

    peft_cfg = LoraConfig(
        r=int(rank),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        target_modules=list(target_modules),
        bias=str(bias),
        task_type=str(task_type),
    )
    inject_adapter_in_model(peft_cfg, model, adapter_name=adapter_name)

    if _current_rank() == 0:
        n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
        logger.info(
            "inject_lora: adapter %r (rank=%d, alpha=%d, target_modules=%s) — %d trainable params",
            adapter_name,
            rank,
            alpha,
            tuple(target_modules),
            n_trainable,
        )

    _stamp(model, partial(_reset_adapter, name=adapter_name))


def _reset_adapter(model: nn.Module, *, name: str) -> None:
    from peft.tuners.lora import LoraLayer

    n_reset = 0
    for m in model.modules():
        if isinstance(m, LoraLayer):
            m.reset_lora_parameters(name, init_lora_weights=True)
            n_reset += 1
    if _current_rank() == 0:
        logger.info("_reset_adapter(%r): %d LoraLayer(s)", name, n_reset)


def _current_rank() -> int:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return 0


__all__ = ["inject_lora"]
