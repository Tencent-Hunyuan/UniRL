"""``PackedImageAdapter`` — image base for packed-sequence DiT families.

Output on the wire is a 4-D packed-token trajectory ``[B, T+1, S, C_packed]`` (a
pure-sequence transformer's denoising state: ``S`` patch tokens × ``C_packed``
packed channels). This base unpacks it to the 5-D image form
``[B, T+1, C_out, H_out, W_out]`` that :class:`ImageAdapter` / ``build_latent_segment``
consume, then assembles the segment via the shared template. Concrete packed
families (FLUX.2-Klein, Qwen-Image) implement ONLY :meth:`_depack` — the per-model
token→image-grid reshape — and inherit the identical collect / guard / token-count
check / reshape scaffolding.

The token grid is ``H // _packed_downsample × W // _packed_downsample`` for every
packed family; what differs is how each turns those tokens back into channels and
spatial extent (Klein keeps the packed channels at patch resolution; Qwen
depatchifies 2×2 to the true 16-channel latent grid). 5-D arrivals pass through
untouched (image-form results), exactly as the former per-family overrides did.

This is the packed-family analogue of the planned ``VideoAdapter`` base: a
per-output-shape intermediate that owns one stage's variation behind a single seam.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import List, Optional

from unirl.config.require import require
from unirl.rollout.engine.sglang_diffusion import utils
from unirl.rollout.engine.sglang_diffusion.adapters.image import ImageAdapter
from unirl.rollout.engine.sglang_diffusion.backends import RawResult
from unirl.types.rollout_req import RolloutReq
from unirl.types.sampling import get_diffusion_params


class PackedImageAdapter(ImageAdapter):
    """Image base for packed-token DiT families: unpack ``[B,T,S,C]`` → image form.

    Subclasses set the two class knobs if they diverge from the defaults and
    implement :meth:`_depack`; everything else (the ``build_segment`` stage and the
    ``_unpack_packed`` scaffolding) is inherited identically.
    """

    #: pixel / (vae_scale_factor × patchify_factor); token grid = H//d × W//d.
    _packed_downsample: int = 16
    #: Whether to reject H/W not divisible by the downsample. FLUX.2-Klein guards
    #: this; Qwen-Image historically does not (it accepts multiples of 8).
    _packed_require_divisible: bool = False

    def build_segment(
        self,
        req: RolloutReq,
        results: List[RawResult],
        *,
        num_steps: int,
        sde_indices: Optional[List[int]],
        emit_native_logprob: bool,
    ):
        """Stage override: unpack the packed trajectory, then assemble the segment.

        Identical to the former per-family overrides — only the unpack
        (:meth:`_depack`, via :meth:`_unpack_packed`) is model-specific.
        """
        traj = utils.collect_trajectory_latents(results)
        traj = self._unpack_packed(traj, req)
        return utils.build_latent_segment(
            traj,
            results=results,
            expected_sigmas=req.sigmas,
            num_steps=num_steps,
            sde_indices=sde_indices,
            emit_native_logprob=emit_native_logprob,
            segment_factory=self.segment_factory,
        )

    def _unpack_packed(self, traj, req: RolloutReq):
        """Convert SGLang's packed ``[B, T, S, C_packed]`` to image-form ``[B, T, C, H, W]``.

        Shared scaffolding around the per-model :meth:`_depack`: 5-D arrivals pass
        through; the rank / height-width / divisibility / token-count guards are
        model-agnostic (the token grid is the same ``H//d × W//d`` for every packed
        family); the final ``reshape`` only restores the ``(B, T)`` leading dims that
        the flat ``B*T`` unpack collapsed.
        """
        if traj.ndim == 5:
            return traj
        if traj.ndim != 4:
            raise ValueError(
                f"{self.model_family}: SGLang trajectory has rank {traj.ndim}, want 4 "
                f"(packed) or 5 (image-form); shape={tuple(traj.shape)}."
            )
        diffusion = get_diffusion_params(req.sampling_params)
        height = int(diffusion.height) if diffusion.height is not None else None
        width = int(diffusion.width) if diffusion.width is not None else None
        if height is None or width is None:
            raise ValueError(
                f"{self.model_family}: need height/width from req.sampling_params to "
                f"unpack the packed [B, T, S, C_packed] trajectory; both must be set."
            )
        d = self._packed_downsample
        if self._packed_require_divisible and (height % d or width % d):
            raise ValueError(
                f"{self.model_family}: height ({height}) and width ({width}) must be "
                f"divisible by the VAE×patchify downsample ({d})."
            )
        h_pat = height // d
        w_pat = width // d
        B, T, S, C_packed = traj.shape
        require(
            S == h_pat * w_pat,
            f"{self.model_family}: packed token count S={S} != h_pat*w_pat={h_pat * w_pat} "
            f"(from height={height}, width={width}). Schedule/recipe drift — fix the "
            f"source rather than silently reshape to a wrong spatial layout.",
        )
        flat = traj.reshape(B * T, S, C_packed)
        out = self._depack(flat, h_pat=h_pat, w_pat=w_pat)
        return out.reshape(B, T, *out.shape[1:]).contiguous()

    @abstractmethod
    def _depack(self, flat, *, h_pat, w_pat):
        """Reshape flat packed tokens ``[B*T, S, C_packed]`` → ``[B*T, C_out, H_out, W_out]``.

        The single packed-family seam. ``h_pat`` / ``w_pat`` are the shared token grid
        (``H//_packed_downsample``, ``W//_packed_downsample``); the base restores the
        leading ``(B, T)`` dims and calls ``.contiguous()`` afterwards.
        """


__all__ = ["PackedImageAdapter"]
