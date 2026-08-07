"""Eval-time diffusion sampling: the ``eval_sampling:`` overlay resolver."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Mapping, Optional, Set

from omegaconf import OmegaConf

from unirl.types.sampling import BaseSamplingParams

# Per-field eval knobs the overlay replaced, and what to write instead.
_RETIRED_EVAL_KEYS = {
    "eval_cfg_text_scale": "eval_sampling: {guidance_scale: X}   (BAGEL family: cfg_text_scale)",
    "eval_num_inference_steps": "eval_sampling: {num_inference_steps: X}",
    "eval_height": "eval_sampling: {height: X}",
    "eval_width": "eval_sampling: {width: X}",
    "eval_media_max_items": "logging: {log_media: true, media_max_items: X}",
    "eval_shift": "eval_sampling: {schedule_shift: X}",
    "eval_mu": "eval_sampling: {schedule_mu: X}",
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
    moves = "\n".join(f"  {key}: X   ->   {_RETIRED_EVAL_KEYS[key]}" for key in present)
    raise ValueError(
        "These per-field eval knobs were replaced by the `eval_sampling:` overlay, which accepts "
        f"ANY DiffusionSamplingParams field:\n{moves}"
    )


def build_eval_sampling(
    sampling_params: Dict[str, BaseSamplingParams],
    *,
    eta: float = 0.0,
    samples_per_prompt: Optional[int] = None,
    overrides: Any = None,
) -> Dict[str, BaseSamplingParams]:
    """Return ``sampling_params`` with its ``diffusion`` entry rebuilt for evaluation.

    Eval INHERITS the training ``sampling:`` block and overlays only what the
    recipe asks for, later winning over earlier:

    1. ``eta`` — recipe ``eval_eta`` (default ``0.0``: deterministic ODE eval).
    2. ``samples_per_prompt`` when given — recipe ``eval_samples_per_prompt``.
    3. ``overrides`` — the recipe's ``eval_sampling:`` block: ANY
       :class:`~unirl.types.sampling.DiffusionSamplingParams` field
       (``guidance_scale``, ``num_inference_steps``, ``height`` / ``width``,
       ``schedule_mu``, ``seed``, ...). Unknown keys raise rather than being
       silently dropped.

    CFG needs no knob of its own: an unmentioned ``guidance_scale`` inherits the
    training guidance, so a CFG-off run cannot silently evaluate with CFG on, and
    naming it decouples the two. It is the field the pipeline consumes, so a
    family that reads ``cfg_text_scale`` must be given THAT one — the inert
    sibling raises instead of being accepted and ignored.

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
    updates.update(_resolve_overrides(overrides, field_names))

    # Only the cfg_text_scale families declare both; elsewhere the sibling is not a
    # field at all and _resolve_overrides already rejected it.
    if "cfg_text_scale" in field_names and "guidance_scale" in updates:
        raise ValueError(
            f"eval_sampling sets `guidance_scale`, which {type(base).__name__} declares but its "
            "pipeline discards — the eval would silently run at the training CFG. "
            "Set `cfg_text_scale` instead."
        )

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
