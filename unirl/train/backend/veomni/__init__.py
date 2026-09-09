"""VeOmni training backend."""

from typing import Any

__all__ = ["VeOmniBackend"]


def __getattr__(name: str) -> Any:
    if name == "VeOmniBackend":
        from unirl.train.backend.veomni.backend import VeOmniBackend

        return VeOmniBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
