"""HunyuanImage3DiffusionState — KV-cache thread across diffusion steps.

The unified-MM transformer's prompt-token K/V projections are
identical at every diffusion step (the prompt tokens never change;
only the noisy latent at the ``<img>`` positions and the timestep
token's value flip per step). Caching them on step 0 and reusing on
steps 1..T-1 cuts the per-step cost from O(L²) to O(L_img × L) — see
``hunyuan.py:1991-2003`` for the upstream first-step / not-first-step
fork that materializes this win.

The unirl per-step kernel is **stateless by Protocol** —
``predict_noise(model, sample, sigma, conditions, *, guidance_scale)``
has no place to put cache state. So we thread an optional
``HunyuanImage3DiffusionState`` instance through the loop:

- Lifetime: born inside one ``HunyuanImage3DiffusionStage.diffuse(...)``
  call, dies at loop exit. Not transported, not shared across rollouts.
- ``replay()`` keeps ``state=None`` because each replay step starts
  from a stored intermediate latent — temporal KV reuse from the prior
  step is meaningless across replay calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass
class HunyuanImage3DiffusionState:
    """Mutable per-rollout state for KV-cache reuse across diffusion steps.

    All fields are filled by the first ``predict_noise`` call (when
    ``step_index == 0`` and the kernel runs forward with
    ``first_step=True, use_cache=True``). Subsequent calls
    (``step_index > 0``) read from + write back to the same instance.

    Field shapes (with ``N = batch * cfg``, ``L`` = full sequence
    length on step 0; ``L'`` = "currently processing" slice length on
    subsequent steps -- typically ``L_img + 1`` for the image block
    plus the timestep token, much smaller than ``L``; ``D`` = head_dim):

        past_key_values            : upstream cache obj (typically
                                     ``HunyuanStaticCache`` or a list
                                     of (K, V) tuples per layer)
        position_ids               : [N, L']      long  -- gathered
                                     down to the changed positions by
                                     ``_update_model_kwargs_for_generation``
        attention_mask             : [N, 1, L', L] bool -- cross-attn
                                     between the L' "currently
                                     processing" rows and the cached L
                                     prompt+image columns
        gen_timestep_scatter_index : [N, K]       long  -- re-derived
                                     per step; relative to the L'
                                     frame of reference
    """

    past_key_values: Optional[Any] = None
    position_ids: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    gen_timestep_scatter_index: Optional[torch.Tensor] = None


__all__ = ["HunyuanImage3DiffusionState"]
