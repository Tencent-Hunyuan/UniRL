"""Hydra config helpers."""

from __future__ import annotations

from typing import Any

from hydra.utils import get_method
from omegaconf import DictConfig, OmegaConf

from unirl.distributed.group.placement import remote

_RESERVED_KEYS = frozenset({"_target_", "_partial_", "_recursive_", "_convert_", "_args_"})


def parse_hydra_cfg(cfg: DictConfig) -> dict[str, Any]:
    """Resolve a Hydra ``_target_`` config into ``remote()``-ready kwargs."""
    if not OmegaConf.is_config(cfg):
        raise TypeError(f"expected a DictConfig, got {type(cfg).__name__}")

    target = cfg.get("_target_")
    if target is None:
        raise ValueError("cfg has no _target_")

    role_cls = get_method(target)

    container = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(container, dict)

    kwargs = {k: v for k, v in container.items() if k not in _RESERVED_KEYS}
    if "role_cls" in kwargs:
        raise ValueError("cfg field 'role_cls' collides with remote()'s parameter name")
    return {"role_cls": role_cls, **kwargs}


def remote_hydra(cfg: DictConfig, **kwargs: Any) -> Any:
    """Sugar for ``remote(**parse_hydra_cfg(cfg), **kwargs)``."""
    return remote(**parse_hydra_cfg(cfg), **kwargs)


__all__ = ["parse_hydra_cfg", "remote_hydra"]
