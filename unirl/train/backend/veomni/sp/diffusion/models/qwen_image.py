"""SP boundary adapter for QwenImageTransformer2DModel (dispatch-patch mechanism)."""

import logging

from unirl.train.backend.veomni.sp.diffusion.ulysses import (
    _install_boundary_hooks,
    _make_rope_slice_hook,
    register,
)

logger = logging.getLogger(__name__)


@register("QwenImageTransformer2DModel")
def _wrap_qwen_image(model, sp_group):
    # qwen pos_embed returns (vid, txt) freqs; slice each on dim 0 (the per-stream freq dim).
    _install_boundary_hooks(
        model,
        sp_group,
        "transformer_blocks",
        "norm_out",
        rope_hook=_make_rope_slice_hook(sp_group, dim=0),
        rope_attr="pos_embed",
    )
    logger.info("diffusion SP: qwen-image boundary hooks installed")
