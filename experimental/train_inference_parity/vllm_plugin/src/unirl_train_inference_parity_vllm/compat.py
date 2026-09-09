"""Pinned runtime and private-symbol contracts for the verified stack."""

from __future__ import annotations

import importlib
import inspect
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version

_VERSION_ALLOWLIST = {
    "vllm": ("0.22.0",),
    "torch": ("2.11.0",),
    "transformers": ("5.6.0",),
}
_REQUIRED = "<required>"


@dataclass(frozen=True)
class RuntimeVersion:
    package: str
    installed: str
    allowed: tuple[str, ...]

    def to_manifest(self) -> dict[str, object]:
        return asdict(self)


def validate_runtime_versions() -> tuple[RuntimeVersion, ...]:
    results = []
    failures = []
    for package, allowed in _VERSION_ALLOWLIST.items():
        try:
            installed = version(package)
        except PackageNotFoundError:
            failures.append(f"{package}=<not installed>; allowed={list(allowed)}")
            continue
        public_version = installed.split("+", 1)[0]
        if public_version not in allowed:
            failures.append(f"{package}={installed}; allowed={list(allowed)}")
        results.append(RuntimeVersion(package=package, installed=installed, allowed=allowed))
    if failures:
        raise RuntimeError("unsupported UniRL parity runtime: " + "; ".join(failures))
    return tuple(results)


def _signature_shape(
    value: object,
) -> tuple[tuple[str, str, str], ...]:
    parameters = inspect.signature(value).parameters.values()
    return tuple(
        (
            parameter.name,
            parameter.kind.name,
            _REQUIRED if parameter.default is inspect.Parameter.empty else repr(parameter.default),
        )
        for parameter in parameters
    )


def require_symbol(
    module_name: str,
    symbol_path: str,
    *,
    parameters: tuple[tuple[str, str] | tuple[str, str, str], ...] | None = None,
    origin: str | None = None,
    strict: bool,
) -> object:
    """Resolve a required symbol and optionally enforce its exact signature."""
    try:
        value: object = importlib.import_module(module_name)
    except ImportError as error:
        raise RuntimeError(f"required parity module {module_name!r} could not be imported") from error
    for component in symbol_path.split("."):
        try:
            value = getattr(value, component)
        except AttributeError as error:
            raise RuntimeError(f"required parity symbol {module_name}.{symbol_path} is missing") from error

    qualified_name = ".".join(
        part
        for part in (
            getattr(value, "__module__", None),
            getattr(value, "__qualname__", None),
        )
        if part
    )
    if origin is not None and qualified_name != origin:
        raise RuntimeError(
            f"conflicting provider at {module_name}.{symbol_path}: "
            f"expected {origin}, got {qualified_name or type(value).__name__}"
        )

    if parameters is not None:
        try:
            actual = _signature_shape(value)
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"cannot inspect required parity symbol {module_name}.{symbol_path}") from error
        expected = tuple(
            (
                parameter[0],
                (inspect.Parameter.POSITIONAL_OR_KEYWORD.name if len(parameter) == 2 else parameter[1]),
                parameter[-1],
            )
            for parameter in parameters
        )
        if strict and actual != expected:
            raise RuntimeError(f"signature drift at {module_name}.{symbol_path}: expected {expected}, got {actual}")
    return value


def require_value(
    module_name: str,
    symbol_path: str,
    *,
    expected_type: type,
) -> object:
    value = require_symbol(module_name, symbol_path, strict=True)
    if not isinstance(value, expected_type):
        raise RuntimeError(
            f"invalid parity symbol {module_name}.{symbol_path}: "
            f"expected {expected_type.__name__}, got {type(value).__name__}"
        )
    return value


__all__ = [
    "RuntimeVersion",
    "require_symbol",
    "require_value",
    "validate_runtime_versions",
]
