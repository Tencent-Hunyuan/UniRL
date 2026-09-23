"""Fail-closed, two-phase registry for parity plugin installers."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SymbolResult:
    """Before/after evidence for one concrete patched symbol."""

    symbol: str
    before_provider: str
    before_signature: str | None
    before_identity: int | None
    after_provider: str
    after_signature: str | None
    after_identity: int | None
    verified: bool


@dataclass(frozen=True)
class PatchResult:
    """Structured result returned by every patch installer."""

    name: str
    symbols: tuple[SymbolResult, ...]

    def to_manifest(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Installer:
    preflight: Callable[[bool], None]
    install: Callable[[bool], PatchResult]


_INSTALLED: dict[str, PatchResult] = {}


def _provider_name(value: object) -> str:
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if module and qualname:
        return f"{module}.{qualname}"
    if isinstance(value, str):
        return value
    return repr(value)


def _signature(value: object) -> str | None:
    try:
        return str(inspect.signature(value))
    except (TypeError, ValueError):
        return None


def symbol_result(
    symbol: str,
    provider: object,
    *,
    before: object,
    actual: object,
) -> SymbolResult:
    """Describe and verify the provider currently bound to ``symbol``."""
    verified = actual is provider
    if not verified:
        raise RuntimeError(
            f"parity provider verification failed for {symbol}: expected identity {provider!r}, got {actual!r}"
        )
    return SymbolResult(
        symbol=symbol,
        before_provider=_provider_name(before),
        before_signature=_signature(before),
        before_identity=id(before),
        after_provider=_provider_name(provider),
        after_signature=_signature(provider),
        after_identity=id(actual),
        verified=True,
    )


def value_result(
    symbol: str,
    provider: str,
    *,
    before: object,
    actual: object,
    verified: bool,
) -> SymbolResult:
    if not verified:
        raise RuntimeError(f"parity value verification failed for {symbol}")
    return SymbolResult(
        symbol=symbol,
        before_provider=_provider_name(before),
        before_signature=None,
        before_identity=id(before),
        after_provider=provider,
        after_signature=None,
        after_identity=id(actual),
        verified=True,
    )


def install_selected(
    names: tuple[str, ...],
    *,
    strict: bool,
    installers: Mapping[str, Installer],
) -> tuple[PatchResult, ...]:
    unknown = sorted(set(names) - installers.keys())
    if unknown:
        raise ValueError(f"unknown parity patches {unknown}; known={sorted(installers)}")

    pending = tuple(name for name in names if name not in _INSTALLED)
    # Complete every compatibility check before mutating process-global state.
    for name in pending:
        installers[name].preflight(strict=strict)
    for name in pending:
        result = installers[name].install(strict=strict)
        if result.name != name:
            raise RuntimeError(f"parity installer {name!r} returned result for {result.name!r}")
        _INSTALLED[name] = result
    return tuple(_INSTALLED[name] for name in names)


def installed_manifest() -> tuple[PatchResult, ...]:
    return tuple(_INSTALLED[name] for name in sorted(_INSTALLED))


__all__ = [
    "Installer",
    "PatchResult",
    "SymbolResult",
    "install_selected",
    "installed_manifest",
    "symbol_result",
    "value_result",
]
