"""Small fail-closed registry for parity plugin installers."""

from __future__ import annotations

from collections.abc import Callable

_INSTALLERS: dict[str, Callable[[], None]] = {}
_INSTALLED: set[str] = set()


def installer(name: str):
    def register(function: Callable[[], None]):
        if name in _INSTALLERS:
            raise RuntimeError(f"duplicate parity plugin installer {name!r}")
        _INSTALLERS[name] = function
        return function

    return register


def install_selected(names: tuple[str, ...], *, strict: bool) -> tuple[str, ...]:
    for name in names:
        if name in _INSTALLED:
            continue
        function = _INSTALLERS.get(name)
        if function is None:
            if strict:
                raise ValueError(f"unknown parity patch {name!r}; known={sorted(_INSTALLERS)}")
            continue
        function()
        _INSTALLED.add(name)
    return tuple(sorted(_INSTALLED))


def installed_manifest() -> tuple[str, ...]:
    return tuple(sorted(_INSTALLED))


__all__ = ["install_selected", "installed_manifest", "installer"]
