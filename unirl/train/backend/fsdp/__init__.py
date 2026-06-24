"""FSDP training backend.

``backend.py`` holds :class:`FSDPBackend` (the training-state Remote);
``wrap.py`` the FSDP2 model wrapping; ``state.py`` the sharded-state
helpers (state-dict gather/load, grad clipping, onload/offload).
"""

from unirl.train.backend.fsdp.backend import FSDPBackend

__all__ = ["FSDPBackend"]
