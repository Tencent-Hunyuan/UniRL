"""Navit-forward adapter over the PRISTINE official Bagel modeling.

The official ``ByteDance-Seed/Bagel`` ``_forward_flow`` is the velocity predictor
the RL path needs, but it (a) consumes a *packed* (navit) sequence + three KV-cache
contexts rather than a dense ``predict_noise(sample, sigma)`` and (b) carries an
upstream ``@torch.no_grad``. This module is the **thin adapter** that bridges those
two facts to UniRL's shared diffusion runtime — and nothing more:

- :func:`forward_flow`           grad-capable velocity via the pristine
                                 ``Bagel._forward_flow`` (bypasses ``@torch.no_grad``
                                 through ``functools.wraps``' ``__wrapped__``).
- :func:`disable_inference_cache` turns off TaylorSeer (per-step determinism for replay).

Everything else the RL loop needs is UniRL's, NOT a flow_grpo port:

- the SDE transition + log-prob  → :class:`unirl.sde.kernels.FlowSDEStrategy`
- which steps run SDE            → :meth:`DiffusionSamplingParams.resolve_sde_indices`
                                   (``unirl.utils.scheduler_utils.AllSDEScheduler``)
- the σ / timestep schedule      → :class:`unirl.sde.runtime.FlowMatchSchedulePolicy`
- the initial noise x_T          → :class:`unirl.types.noise_recipe.NoiseRecipe`

so :class:`unirl.models.bagel.diffusion.BagelDiffusionStage` reads exactly like
``SD3DiffusionStage`` (central schedule + sde_indices + kernel + noise), with this
adapter supplying only the model-specific velocity call. ``vendor/`` stays
byte-pristine; an upstream bump is a re-vendor + import-rewrite with this file
untouched.

Gradients
---------
``Bagel._forward_flow`` carries ``@torch.no_grad`` upstream. :func:`forward_flow`
reaches the undecorated function via ``functools.wraps``' ``__wrapped__`` so replay
can backprop while the vendored file stays unedited (verified on torch 2.11: the
decorated form blocks grad even under ``enable_grad``; ``__wrapped__`` restores it).
Under an outer ``torch.no_grad()`` (e.g. rollout) it stays grad-free, so the same
function serves rollout, the ratio test, and training.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "forward_flow",
    "disable_inference_cache",
]


def disable_inference_cache(model: Any) -> None:
    """Turn off the TaylorSeer cache for the RL path (per-step determinism).

    The pristine ``_forward_flow`` reads ``self.language_model.model.enable_taylorseer``;
    the official ``generate_image`` sets it, but the RL loop calls ``_forward_flow``
    directly so we set the flag here (the cache would break per-step determinism →
    replay would not be bit-exact). Best-effort; ignored if the attribute path is
    absent (e.g. a fake model in unit tests).
    """
    try:
        model.language_model.model.enable_taylorseer = False
    except AttributeError:
        pass


def _raw_forward_flow(model: Any):
    """The undecorated ``Bagel._forward_flow`` (bypasses upstream ``@torch.no_grad``)."""
    fn = type(model)._forward_flow
    return getattr(fn, "__wrapped__", fn)


def forward_flow(model: Any, **kwargs: Any) -> Any:
    """Velocity prediction via the pristine vendored ``Bagel._forward_flow``.

    Bypasses upstream's ``@torch.no_grad`` (via ``__wrapped__``) so gradients flow
    during replay; under an outer ``torch.no_grad()`` it is still grad-free. The
    TaylorSeer cache kwargs (``model_pred_*``) are left at their ``None`` defaults —
    the RL path disables that cache (see :func:`disable_inference_cache`).

    ``model._forward_flow`` already does the CFG combine internally (gen / cfg_text /
    cfg_img contexts + ``cfg_text_scale`` / ``cfg_img_scale`` / ``cfg_renorm_*``), so
    the returned velocity is the CFG-combined ``v_t`` the SDE kernel consumes.
    """
    return _raw_forward_flow(model)(model, **kwargs)
