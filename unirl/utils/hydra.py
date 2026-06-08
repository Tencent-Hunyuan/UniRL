"""Hydra config helpers.

Small utilities for working with Hydra/OmegaConf configs outside the main
``hydra.utils.instantiate`` flow. The driver routes config; the worker
materializes objects — ``parse_hydra_cfg`` resolves only the top-level
``_target_`` (to a class for ``remote()`` to register a ``Handle``) and
leaves every nested ``_target_`` block as plain data for worker-side
resolution.
"""

from __future__ import annotations

from typing import Any

from hydra.utils import get_method
from omegaconf import DictConfig, OmegaConf

from unirl.distributed.group.placement import remote

_RESERVED_KEYS = frozenset({"_target_", "_partial_", "_recursive_", "_convert_", "_args_"})


def parse_hydra_cfg(cfg: DictConfig) -> dict[str, Any]:
    """Resolve a Hydra ``_target_`` config into ``remote()``-ready kwargs.

    The returned dict has ``role_cls`` (the class resolved from the
    top-level ``_target_``) plus the cfg's own fields as plain Python
    via ``OmegaConf.to_container(resolve=True)``. Nested ``_target_``
    blocks are **not** instantiated here — they pass through as plain
    dicts so the worker can construct them in its own CUDA context
    (see ``Worker.add_remote`` and its ``_resolve_init_kwargs`` walker).

    Designed to be star-star-unpacked into ``remote(...)`` — see
    ``remote_hydra`` for the common sugar::

        self.bundle = remote(**parse_hydra_cfg(bundle_cfg))

    Raises
    ------
    ValueError
        If ``cfg`` has no ``_target_`` or the target is not a class.
    TypeError
        If ``cfg`` is not a DictConfig.
    """
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
    """Sugar for ``remote(**parse_hydra_cfg(cfg), **kwargs)``.

    Extra ``**kwargs`` are merged on top of the cfg's own fields (typically
    sibling ``Handle``s like ``bundle=self.bundle``) before being forwarded
    to ``remote()``.
    """
    return remote(**parse_hydra_cfg(cfg), **kwargs)


__all__ = ["parse_hydra_cfg", "remote_hydra"]
