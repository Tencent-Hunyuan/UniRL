"""Eval-time diffusion sampling: the ``eval_sampling:`` overlay resolver."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Mapping, Optional, Set

from omegaconf import OmegaConf

from unirl.types.sampling import BaseSamplingParams

# Per-field eval knobs the overlay replaced, and the params field each maps to.
_RETIRED_EVAL_KEYS = {
    "eval_num_inference_steps": "num_inference_steps",
    "eval_height": "height",
    "eval_width": "width",
    "eval_shift": "schedule_shift",
    "eval_mu": "schedule_mu",
}


def cfg_scale_of(params: Any) -> float:
    """The CFG scale a diffusion params object will actually be sampled with.

    BAGEL-family params carry ``cfg_text_scale``; every other family carries
    ``guidance_scale``. Log lines read the scale through here so they report the
    field the pipeline actually consumes.
    """
    scale = getattr(params, "cfg_text_scale", None)
    return float(params.guidance_scale if scale is None else scale)


def reject_retired_eval_keys(cfg: Any) -> None:
    """Fail fast on the per-field ``eval_*`` knobs that ``eval_sampling:`` replaced.

    Ignoring them would silently evaluate at the rollout's own setting — the exact
    train/eval mismatch the overlay exists to make explicit.
    """
    present = sorted(key for key in _RETIRED_EVAL_KEYS if cfg is not None and cfg.get(key) is not None)
    if not present:
        return
    moves = "\n".join(f"  {key}: X   ->   eval_sampling: {{{_RETIRED_EVAL_KEYS[key]}: X}}" for key in present)
    raise ValueError(
        "These per-field eval knobs were replaced by the `eval_sampling:` overlay, which accepts "
        f"ANY DiffusionSamplingParams field:\n{moves}"
    )


def build_eval_sampling(
    sampling_params: Dict[str, BaseSamplingParams],
    *,
    cfg_text_scale: float,
    eta: float = 0.0,
    samples_per_prompt: Optional[int] = None,
    overrides: Any = None,
) -> Dict[str, BaseSamplingParams]:
    """Return ``sampling_params`` with its ``diffusion`` entry rebuilt for evaluation.

    Eval INHERITS the training ``sampling:`` block and overlays only what the
    recipe asks for, later winning over earlier:

    1. ``eta`` — recipe ``eval_eta`` (default ``0.0``: deterministic ODE eval).
    2. ``samples_per_prompt`` when given — recipe ``eval_samples_per_prompt``.
    3. the CFG scale — recipe ``eval_cfg_text_scale``, written onto whichever
       field this params family carries.
    4. ``overrides`` — the recipe's ``eval_sampling:`` block: ANY
       :class:`~unirl.types.sampling.DiffusionSamplingParams` field
       (``num_inference_steps``, ``height`` / ``width``, ``schedule_mu``,
       ``seed``, ...). Unknown keys raise rather than being silently dropped.

    A resolved ``eta <= 0`` then clears the SDE gate (``sde_indices=[]``,
    ``scheduler=None``): eta=0 with gated steps is a contradictory request — the
    central kernel degrades such steps to ODE, and worker-resident schedulers
    (BAGEL) refuse the pair outright. A resolved ``eta > 0`` keeps the training
    gate, whose indices are resolved against the ROLLOUT's step count, so a step
    override is rejected here rather than addressing a schedule it cannot reach.

    The rollout's params are never mutated, so eval settings cannot leak into the
    trajectories the policy is trained on.
    """
    base = sampling_params.get("diffusion")
    if base is None:
        raise ValueError("build_eval_sampling: sampling params carry no `diffusion` entry to override.")
    field_names = {f.name for f in dataclasses.fields(base)}

    updates: Dict[str, Any] = {"eta": float(eta)}
    if samples_per_prompt is not None:
        updates["samples_per_prompt"] = int(samples_per_prompt)
    updates["cfg_text_scale" if "cfg_text_scale" in field_names else "guidance_scale"] = float(cfg_text_scale)
    updates.update(_resolve_overrides(overrides, field_names))

    steps = int(updates.get("num_inference_steps", base.num_inference_steps))
    if float(updates["eta"]) <= 0.0:
        updates["sde_indices"] = []
        updates["scheduler"] = None
    elif steps != int(base.num_inference_steps):
        raise ValueError(
            f"eval eta={updates['eta']} leaves the SDE gate on, but eval_sampling.num_inference_steps"
            f"={steps} differs from the rollout's {base.num_inference_steps}: the gated step indices "
            "are resolved against the rollout's step count and cannot address the eval schedule. "
            "Set eval_eta: 0, or drop the step override."
        )
    return {**sampling_params, "diffusion": dataclasses.replace(base, **updates)}


def _resolve_overrides(overrides: Any, field_names: Set[str]) -> Dict[str, Any]:
    """Validate a recipe ``eval_sampling:`` block into plain ``dataclasses.replace`` kwargs."""
    if overrides is None:
        return {}
    if OmegaConf.is_config(overrides):
        overrides = OmegaConf.to_container(overrides, resolve=True)
    if not isinstance(overrides, Mapping):
        raise TypeError(
            "eval_sampling must be a mapping of diffusion sampling fields, "
            f"got {type(overrides).__name__}. It overlays `sampling:`, so it takes no `_target_`."
        )
    unknown = sorted(set(overrides) - field_names)
    if unknown:
        raise ValueError(f"eval_sampling has unknown field(s) {unknown}; valid fields are {sorted(field_names)}.")
    return dict(overrides)


__all__ = ["build_eval_sampling", "cfg_scale_of", "reject_retired_eval_keys"]
