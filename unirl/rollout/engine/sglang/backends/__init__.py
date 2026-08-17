"""The backend seam package — the runtime boundary of the engine."""

from unirl.rollout.engine.sglang.backends.base import Backend, RawResult
from unirl.rollout.engine.sglang.backends.http import HTTPBackend
from unirl.rollout.engine.sglang.backends.native import NativeBackend

__all__ = ["Backend", "HTTPBackend", "NativeBackend", "RawResult"]
