"""The backend seam package — the runtime boundary of the engine.

``base.py`` holds the ``Backend`` protocol + the wire types (the contract every
collaborator binds to); ``native.py`` is the real impl over ``DiffGenerator`` +
the ZMQ scheduler client (local-mode spawn and remote scheduler connect). An
HTTP-server impl would land beside it as ``http.py`` — consumers import from this
package, so adding one touches no engine/adapter/weight-sync code.
"""

from unirl.rollout.engine.sglang_diffusion.backends.base import (
    Backend,
    EncoderOutputs,
    MediaPayload,
    RawResult,
)
from unirl.rollout.engine.sglang_diffusion.backends.native import SGLangBackend

__all__ = ["Backend", "SGLangBackend", "RawResult", "EncoderOutputs", "MediaPayload"]
