"""Per-mode generate functions for ``HunyuanImage3Pipeline``.

Each submodule exports a single ``generate(pipeline, req) -> RolloutResp``
function that orchestrates the stages for its task topology
(``t2t`` / ``i2t`` / ``t2i`` / ``it2i`` / ``t2ti``). The dispatcher in
:mod:`unirl.models.hunyuan_image3.pipeline` looks up
``stage_params["task"]`` and delegates to the matching submodule.

Splitting per-mode keeps the core ``pipeline.py`` slim (~80 LoC
dispatcher) and makes each task's request → response wiring a
self-contained file that's easy to test and extend independently.
"""

from . import i2t, it2i, t2i, t2t, t2ti

__all__ = ["i2t", "it2i", "t2i", "t2t", "t2ti"]
