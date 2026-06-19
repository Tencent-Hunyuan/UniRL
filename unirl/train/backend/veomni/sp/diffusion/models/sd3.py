"""SP boundary adapter for SD3Transformer2DModel (processor-injection mechanism; no RoPE)."""

import logging

from unirl.train.backend.veomni.sp.diffusion.ulysses import _install_boundary_hooks, register

logger = logging.getLogger(__name__)


@register("SD3Transformer2DModel")
def _wrap_sd3(model, sp_group):
    # SD3 has learned positional embeddings baked into the patches (no RoPE).
    _install_boundary_hooks(model, sp_group, "transformer_blocks", "norm_out")
    logger.info("diffusion SP: sd3 boundary hooks installed")
