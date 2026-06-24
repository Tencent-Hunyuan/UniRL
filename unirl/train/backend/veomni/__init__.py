"""VeOmni training backend.

``backend.py`` holds :class:`VeOmniBackend` (the training-state Remote);
``wrap.py`` the VeOmni FSDP2 parallelization; ``state.py`` the sharded-state
helpers; ``_compat.py`` the selective-import shim (the ONLY veomni import
site in unirl).

Unlike the fsdp package, the re-export is lazy (PEP 562): ``backend.py``
imports torch at module level, but recipe tooling (Hydra compose checks,
config linting) must be able to import this package on torch-less machines.
``unirl.train.backend.veomni.VeOmniBackend`` still resolves as a Hydra
``_target_`` — ``hydra.utils.get_method``/``get_class`` trigger the lazy
attribute load.
"""

from typing import Any

__all__ = ["VeOmniBackend"]


def __getattr__(name: str) -> Any:
    if name == "VeOmniBackend":
        from unirl.train.backend.veomni.backend import VeOmniBackend

        return VeOmniBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
