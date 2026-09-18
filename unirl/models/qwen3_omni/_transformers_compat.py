"""Narrow Transformers compatibility fixes for Qwen3-Omni."""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from importlib.metadata import version
from typing import Any

from packaging.version import Version

logger = logging.getLogger(__name__)

_AFFECTED_TRANSFORMERS_RELEASE = (5, 6, 0)
_FLASH_ATTENTION_BACKENDS = ("flash_attention_2", "flash_attention_3", "flash_attention_4")
_PATCH_MARKER = "_unirl_optional_s_aux_compat"


class _NullAttentionSink:
    """Adapt ``None`` to the buggy Transformers 5.6.0 ``s_aux.to(...)`` call."""

    def to(self, *_args: Any, **_kwargs: Any) -> None:
        return None


_NULL_ATTENTION_SINK = _NullAttentionSink()


def _needs_optional_s_aux_compat(transformers_version: str) -> bool:
    return Version(transformers_version).release == _AFFECTED_TRANSFORMERS_RELEASE


def _wrap_optional_s_aux(attention_forward: Callable[..., Any]) -> Callable[..., Any]:
    """Preserve ``s_aux=None`` semantics through Transformers 5.6.0's FA dispatcher."""
    if getattr(attention_forward, _PATCH_MARKER, False):
        return attention_forward

    @functools.wraps(attention_forward)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("s_aux") is None:
            kwargs["s_aux"] = _NULL_ATTENTION_SINK
        return attention_forward(*args, **kwargs)

    setattr(wrapped, _PATCH_MARKER, True)
    return wrapped


def _patch_attention_registry(registry: Any) -> tuple[str, ...]:
    patched: list[str] = []
    for backend in _FLASH_ATTENTION_BACKENDS:
        try:
            attention_forward = registry[backend]
        except KeyError:
            continue
        wrapped = _wrap_optional_s_aux(attention_forward)
        if wrapped is attention_forward:
            continue
        registry.register(backend, wrapped)
        patched.append(backend)
    return tuple(patched)


def install_transformers_flash_attention_compat() -> tuple[str, ...]:
    """Install the Transformers 5.6.0 optional-attention-sink fix once per process."""
    if not _needs_optional_s_aux_compat(version("transformers")):
        return ()

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    patched = _patch_attention_registry(ALL_ATTENTION_FUNCTIONS)
    if patched:
        logger.info(
            "Installed Transformers 5.6.0 optional s_aux compatibility for: %s",
            ", ".join(patched),
        )
    return patched


__all__ = ["install_transformers_flash_attention_compat"]
