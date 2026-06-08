"""WAN 2.2 T2V pipeline — new four-tier architecture, dual-transformer.

Re-expression of the legacy WAN 2.2 implementation against the typed
``Bundle`` / ``Pipeline`` / ``DiffusionStage`` / ``DecodeStage``
protocols. This package is the framework direction for
``train.py`` + ``model: wan22``. The legacy ``unirl/models``
path remains in tree for active old recipes, code comparison, and a
smaller merge surface with ``main`` while those recipes are retired.

WAN 2.2's twist over WAN 2.1 is dual-transformer boundary routing: the
``high_noise`` transformer handles ``sigma >= boundary_ratio`` (coarse
structure), the ``low_noise`` transformer handles ``sigma <
boundary_ratio`` (detail refinement). Both are exposed through a single
:class:`unirl.models.wan22.bundle.WanDualTransformer` so the
new policy stack has one trainable module to inspect. FSDPPolicy does
block-only wrapping: it discovers ``WanTransformerBlock`` instances under
both branches and shards those blocks individually; the composite root is
not itself fully sharded.

The text-embedding and VAE-decode stages reuse the WAN 2.1
implementations verbatim — only the diffusion stage swaps in. This
follows SD3's one-package-per-model convention; WAN 2.2 is NOT a
subclass of WAN 2.1's pipeline.

Importing this package re-exports its bundle / pipeline / config classes;
recipes wire them by ``_target_`` dotpath.
"""

from unirl.models.wan22.bundle import WAN22Bundle, WanDualTransformer
from unirl.models.wan22.config import (
    DEFAULT_BOUNDARY_RATIO,
    WAN22PipelineConfig,
)
from unirl.models.wan22.diffusion import (
    WAN22DiffusionStage,
    WAN22DiffusionStep,
)
from unirl.models.wan22.pipeline import WAN22Pipeline

__all__ = [
    "DEFAULT_BOUNDARY_RATIO",
    "WAN22Bundle",
    "WAN22DiffusionStage",
    "WAN22DiffusionStep",
    "WAN22Pipeline",
    "WAN22PipelineConfig",
    "WanDualTransformer",
]
