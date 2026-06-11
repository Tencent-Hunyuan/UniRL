"""FLUX-family image adapters: plain FLUX + FLUX.2-Klein.

``FluxAdapter`` is the default image path (5-D passthrough). ``Flux2KleinAdapter``
is a packed-token family — Klein's transformer is a pure sequence model emitting
packed ``[B, T, H*W, C_packed]`` tokens — so it rides the shared
``PackedImageAdapter`` unpack and overrides only ``_depack`` (the token→grid
reshape, channels unchanged at patch resolution) plus the schedule policy (Klein
needs a model-specific ``compute_mu`` the generic FlowMatch path can't synthesize).
The Dance-GRPO SDE label rides on the base ``resolve_sde_label``.
"""

from __future__ import annotations

from unirl.config.require import require
from unirl.rollout.engine.sglang_diffusion.adapters.base import register_adapter
from unirl.rollout.engine.sglang_diffusion.adapters.image import ImageAdapter
from unirl.rollout.engine.sglang_diffusion.adapters.packed_image import PackedImageAdapter


@register_adapter("flux")
class FluxAdapter(ImageAdapter):
    """FLUX — image-form 5-D trajectory throughout; default path."""

    pass


@register_adapter("flux2_klein")
class Flux2KleinAdapter(PackedImageAdapter):
    """FLUX.2-Klein — packed sequence-style trajectory + model-specific schedule."""

    #: Klein rejects H/W not divisible by the VAE×patchify downsample (default 16).
    _packed_require_divisible = True

    def validate(self) -> None:
        super().validate()
        require(
            callable(getattr(self.model_config, "build_schedule_policy", None)),
            "flux2_klein adapter requires model_config.build_schedule_policy() "
            "(Klein needs a model-specific compute_mu the generic FlowMatch path "
            "cannot synthesize from scheduler_config.json).",
        )

    def schedule_policy(self):
        return self.model_config.build_schedule_policy()

    def _depack(self, flat, *, h_pat, w_pat):
        """Klein keeps the packed channels at patch resolution (token→spatial reshape)."""
        from unirl.models.flux2_klein.flux2_klein_utils import unpack_latents

        return unpack_latents(flat, h_pat, w_pat)


__all__ = ["FluxAdapter", "Flux2KleinAdapter"]
