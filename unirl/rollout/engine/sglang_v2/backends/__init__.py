"""The backend seam package — the runtime boundary of the engine.

``base.py`` holds the ``Backend`` protocol + the wire types (the contract every
collaborator binds to); ``http.py`` is the real impl over a spawned SGLang SRT
server (the only module importing sglang or doing I/O — boot included). A
future in-process impl would land beside it — consumers import from this
package, so adding one touches no engine/adapter/weight-sync code.
"""

from unirl.rollout.engine.sglang_v2.backends.base import Backend, RawResult
from unirl.rollout.engine.sglang_v2.backends.http import HTTPBackend

__all__ = ["Backend", "HTTPBackend", "RawResult"]
