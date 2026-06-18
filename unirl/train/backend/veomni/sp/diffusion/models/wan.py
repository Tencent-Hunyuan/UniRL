"""SP boundary adapter for WanTransformer3DModel (dispatch-patch; image self-attn + text cross-attn)."""

import logging

from unirl.train.backend.veomni.sp.diffusion.ulysses import (
    _install_boundary_hooks,
    _make_rope_slice_hook,
    register,
)

logger = logging.getLogger(__name__)


@register("WanTransformer3DModel")
def _wrap_wan(model, sp_group):
    # Wan: image self-attn (slice image) + text cross-attn (text stays FULL; the
    # dispatch cross-attn guard skips its all-to-all). Block call is positional.
    # Wan rotary: (cos, sin), each (1, S_img, 1, D); slice the image seq dim (dim 1).
    _install_boundary_hooks(
        model,
        sp_group,
        "blocks",
        "norm_out",
        rope_hook=_make_rope_slice_hook(sp_group, dim=1),
        rope_attr="rope",
        slice_encoder=False,
    )
    logger.info("diffusion SP: wan boundary hooks installed")
