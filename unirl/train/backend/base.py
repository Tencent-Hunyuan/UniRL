"""Backend-agnostic schema dataclasses for the training stack.

Config-layer schemas (torch-free — shared by every train backend):

* :class:`OptimizerConfig` — AdamW hyperparameters
* :class:`LrSchedulerConfig` — LR schedule hyperparameters
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OptimizerConfig:
    """AdamW-style optimizer hyperparameters consumed by the training actor."""

    learning_rate: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    weight_decay: float


@dataclass
class LrSchedulerConfig:
    """Learning-rate scheduler hyperparameters."""

    type: str
    warmup_steps: int
    total_steps: int


def resolve_trainable_module(bundle: object, trainable_attr: str):
    """The module a backend wraps + optimizes + checkpoints.

    A bundle may expose ``trainable_module()`` to hand the backend a *nested*
    submodule (e.g. hunyuan_image3's bare decoder ``transformer.model``). The
    backend then shards / optimizes / checkpoints exactly that trainable subtree,
    and the composite's frozen aux (diffusion heads, VAE, ViT) stays *outside*
    the wrap — on meta until the bundle materializes it, and out of the
    optimizer / checkpoint scope. Crucially this is what lets the composite run
    under VeOmni, whose ``parallelize`` root-shards + whole-root-``to_empty``s
    whatever module it is given (a heterogeneous composite is out of scope, a
    single decoder is not).

    Bundles that do not expose ``trainable_module()`` fall back to the named
    attribute (the common single-module case), so existing recipes are
    unaffected.
    """
    tm = getattr(bundle, "trainable_module", None)
    return tm() if callable(tm) else getattr(bundle, trainable_attr)


__all__ = [
    "LrSchedulerConfig",
    "OptimizerConfig",
    "resolve_trainable_module",
]
