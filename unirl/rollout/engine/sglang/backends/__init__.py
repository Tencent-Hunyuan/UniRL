"""The backend seam package — the runtime boundary of the engine.

``base.py`` holds the ``Backend`` protocol + the wire types (the contract every
collaborator binds to); ``http.py`` is the impl over a spawned SGLang SRT
server; ``native.py`` is the in-process ``sglang.Engine`` impl (the promised
landing spot). Each impl is the only module importing sglang on its path — boot
included. Consumers import from this package, so adding an impl touches no
engine/adapter/weight-sync code.
"""

from unirl.rollout.engine.sglang.backends.base import Backend, RawResult
from unirl.rollout.engine.sglang.backends.http import HTTPBackend
from unirl.rollout.engine.sglang.backends.native import NativeBackend

__all__ = ["Backend", "HTTPBackend", "NativeBackend", "RawResult"]
