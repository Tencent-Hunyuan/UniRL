"""``RolloutReq`` → SGLang ``DiffGenerator.generate(sampling_params_kwargs=...)`` translator.

Single free function ``_to_sglang_kwargs(req, *, cfg, sde_label, initial_noise)``
that mirrors the layered kwargs build from the legacy
``samplers/sglang/request.py:to_kwargs``:

1. **Escape hatch** — ``cfg.sampling.sampler_kwargs`` (raw SGLang overrides).
2. **Typed / computed fields** — prompts (with optional K de-expansion),
   sigma schedule, geometry, sample count.
3. **Engine pins** — trajectory return shape, RNG seed-out, no group-share
   inside SGLang.
4. **SDE-mode kernel kwargs** — only when ``sampling_params.diffusion.
   sde_indices`` is non-None.

``initial_noise`` is taken as an argument (engine-computed or pre-shipped via
``req.request_conditions['initial_latents']``); the translator is pure and
performs no allocation.

De-expansion uses ``req.group_ids``: when every group has the same repeat
count ``k`` and the per-group prompts are identical strings, the translator
collapses to unique prompts + ``num_outputs_per_prompt=k`` so SGLang runs one
text-encode pass per group instead of K. Falls through unchanged for any
deviation (heterogeneous K, mismatched prompts within a group).

The negative-prompt CFG invariant from legacy ``samplers/sglang/request.py:128-137``
ports over verbatim — set ``return_negative_prompt_embeds=True`` whenever
``negative_prompt`` is provided.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from unirl.config.require import require
from unirl.rollout.engine.sglang.config import SGLangEngineConfig
from unirl.types.primitives import Texts
from unirl.types.rollout_req import RolloutReq
from unirl.types.sampling import get_diffusion_params


def _deexpand_prompts_from_groups(
    prompts: List[str],
    group_ids: List[str],
) -> Tuple[List[str], int]:
    """Collapse K-expanded prompts back to unique prompts when groups agree.

    Returns ``(unique_prompts, k)`` where ``k`` is the repeat count to set
    as ``num_outputs_per_prompt`` on SGLang. Falls through to ``(prompts, 1)``
    when the structure doesn't admit a clean collapse: heterogeneous K per
    group, mismatched prompt strings within a group, or empty groups.
    """
    n = len(prompts)
    if n == 0 or len(group_ids) != n:
        return list(prompts), 1

    # Walk in order, recording per-group sample positions.
    groups: Dict[str, List[int]] = {}
    order: List[str] = []
    for i, gid in enumerate(group_ids):
        if gid not in groups:
            order.append(gid)
            groups[gid] = []
        groups[gid].append(i)

    if not groups:
        return list(prompts), 1

    k_per_group = {gid: len(idxs) for gid, idxs in groups.items()}
    k_values = set(k_per_group.values())
    if len(k_values) != 1:
        return list(prompts), 1

    k = next(iter(k_values))
    if k <= 1:
        return list(prompts), 1

    unique_prompts: List[str] = []
    for gid in order:
        idxs = groups[gid]
        base = prompts[idxs[0]]
        if any(prompts[i] != base for i in idxs[1:]):
            return list(prompts), 1
        unique_prompts.append(base)
    return unique_prompts, k


def _to_sglang_kwargs(
    req: RolloutReq,
    *,
    cfg: SGLangEngineConfig,
    sde_label: Optional[str],
    initial_noise: Optional[torch.Tensor],
) -> Dict[str, Any]:
    """Translate a ``RolloutReq`` into the kwargs dict ``DiffGenerator.generate`` consumes."""
    text_prim = req.primitives.get("text")
    if not isinstance(text_prim, Texts):
        raise TypeError(
            f"_to_sglang_kwargs: req.primitives['text'] must be Texts; got "
            f"{type(text_prim).__name__ if text_prim is not None else 'None'}"
        )
    prompts = list(text_prim.texts)
    require(bool(prompts), "_to_sglang_kwargs: req.primitives['text'] must be non-empty")
    require(
        len(prompts) == len(req.sample_ids),
        f"_to_sglang_kwargs: text count {len(prompts)} != sample_ids count {len(req.sample_ids)}",
    )

    diffusion = get_diffusion_params(req.sampling_params)
    require(
        diffusion is not None,
        "_to_sglang_kwargs: req.sampling_params must contain diffusion params (set at request construction)",
    )

    num_inference_steps = int(diffusion.num_inference_steps)
    guidance_scale = float(diffusion.guidance_scale)
    height = int(diffusion.height)
    width = int(diffusion.width)
    eta = float(diffusion.eta)
    sde_indices_raw = diffusion.sde_indices
    sde_indices = sorted(int(v) for v in sde_indices_raw) if sde_indices_raw is not None else None

    # σ is the SSOT field on ``RolloutReq``. The engine populates it via
    # :func:`unirl.sde.runtime.ensure_req_sigmas` BEFORE calling
    # this translator; we never recompute here so any drift between the
    # schedule training will replay against and what SGLang executes is
    # impossible. ``req.sigmas`` is a length-``T+1`` tensor (terminal 0
    # included); SGLang's ``set_timesteps(sigmas=...)`` expects the
    # interior ``T`` values (terminal 0 is implicit), so slice it off
    # right before serializing.
    require(
        req.sigmas is not None,
        "_to_sglang_kwargs: req.sigmas must be set by the engine before "
        "calling the translator (see "
        "unirl.sde.runtime.ensure_req_sigmas).",
    )
    require(
        int(req.sigmas.shape[0]) == num_inference_steps + 1,
        f"_to_sglang_kwargs: req.sigmas length {int(req.sigmas.shape[0])} != "
        f"num_inference_steps+1 ({num_inference_steps + 1}). Engine must "
        f"populate σ for the resolved num_inference_steps.",
    )
    sigmas = req.sigmas.detach().cpu().tolist()[:-1]

    # De-expansion: collapse K-expanded prompts on group boundaries when clean.
    unique_prompts, k = _deexpand_prompts_from_groups(prompts, list(req.group_ids))
    prompt_payload: Any = unique_prompts if len(unique_prompts) > 1 else unique_prompts[0]
    num_outputs_per_prompt: Optional[int] = k if k > 1 else None

    sampler_kwargs: Dict[str, Any] = dict(diffusion.sampler_kwargs or {})

    # Negative-prompt CFG invariant (ported from samplers/sglang/request.py:128-137):
    # SGLang gates CFG on guidance_scale>1 independently of
    # return_negative_prompt_embeds. If we accept ``negative_prompt`` without
    # also pinning ``return_negative_prompt_embeds=True``, rollout conditions on
    # the negative prompt while training-side replay falls back to zero negative
    # embeds — silent GRPO ratio mismatch. Fail fast at the boundary.
    neg_prompt = sampler_kwargs.get("negative_prompt")
    return_neg_embeds = bool(sampler_kwargs.get("return_negative_prompt_embeds", False))
    require(
        neg_prompt is None or return_neg_embeds,
        "_to_sglang_kwargs: sampler_kwargs.negative_prompt is set but "
        "return_negative_prompt_embeds is not True. SGLang gates CFG on "
        "guidance_scale>1 independently of return_negative_prompt_embeds, so the "
        "rollout would condition on the negative prompt while training-side "
        "replay falls back to zero negative embeds — silent GRPO ratio mismatch. "
        "Set sampler_kwargs.return_negative_prompt_embeds=True to keep rollout "
        "and replay aligned.",
    )

    # Layer 1: caller escape-hatch (lowest priority).
    kwargs: Dict[str, Any] = dict(sampler_kwargs)

    # Layers 2 + 3: typed/computed + engine pins (override layer 1).
    kwargs.update(
        {
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "height": height,
            "width": width,
            "num_frames": int(diffusion.num_frames),
            "sigmas": sigmas,
            "prompt": prompt_payload,
            # Stock upstream has no ``init_same_noise`` field; its default draws
            # per-output noise (== the fork's ``init_same_noise=False``), and any
            # group-sharing pattern is already carried by ``initial_noise`` below,
            # so the fork flag is simply dropped.
            # Output shape policy: latents + prompt embeds, no per-step decoded
            # frames. Pass the sampling seed (not None): when ``initial_noise`` is
            # supplied it still determines x_T verbatim, so the seed only feeds
            # SGLang's per-step SDE noise — and SGLang derives that deterministically
            # (its ``_make_step_generators``, keyed on denoise_seeds) ONLY when
            # ``seed is not None``; a None seed silently falls back to global RNG and
            # breaks per-step reproducibility / cross-engine alignment.
            "seed": int(diffusion.seed) if getattr(diffusion, "seed", None) is not None else 0,
            "save_output": False,
            "return_file_paths_only": False,
            "return_trajectory_latents": True,
            "return_trajectory_decoded": False,
        }
    )

    # ``return_prompt_embeds`` / ``return_negative_prompt_embeds`` are the
    # conditions-path opt-in flags re-hosted onto stock upstream by
    # ``_patches/patch_conditions`` (+ ``patch_sampling_io`` makes them genuine
    # ``SamplingParams`` fields). Only request them when the engine populates
    # conditions; positives are always emitted under ``populate_conditions``
    # (GRPO's ``typed_conditions`` consumes ``conditions['text']``), negatives
    # only when CFG is actually active in the rollout.
    if cfg.populate_conditions:
        kwargs["return_prompt_embeds"] = True
        # SGLang only populates negative embeds when CFG runs, i.e.
        # ``guidance_scale > 1`` AND a negative prompt is present (its CFG gate).
        # Mirror that here so the flag tracks the rollout's actual CFG state and
        # the negative branch (``conditions['negative_text']``) is emitted exactly
        # when training-side replay will consume it -- keeping rollout/replay
        # aligned (the same invariant the ``require`` above enforces from the
        # opposite direction). When CFG is off (SD3 pilot: guidance_scale=1.0),
        # negatives are neither requested nor produced.
        if guidance_scale > 1.0 and neg_prompt is not None:
            kwargs["return_negative_prompt_embeds"] = True

    if initial_noise is not None:
        kwargs["initial_noise"] = initial_noise

    # Per-step SDE noise key. Keyed on sample_ids (unique per sample) so each
    # sample explores its own per-step SDE noise; the fork keys on group_ids
    # (same-group samples share per-step noise). x_T is already per-sample via the
    # initial_noise injection above, so within-group diversity does not depend on
    # this; it is a secondary exploration knob (NOT the flat-reward root cause --
    # that was the grouped-forward trajectory collapse, see patch_rollout_trajectory).
    if req.sample_ids:
        kwargs["denoise_seeds"] = [str(sid) for sid in req.sample_ids]

    # Layer 4: rollout mode. The response builder ALWAYS rebuilds the
    # ``LatentSegment`` from ``rollout_trajectory_data.dit_trajectory.latents``
    # (DiffusionNFT reads the clean final latent; GRPO replays the trajectory), so
    # request rollout + the dit-trajectory unconditionally (collection is gated by
    # ``rollout_return_dit_trajectory``, independently of the SDE kernel).
    kwargs["rollout"] = True
    kwargs["rollout_return_dit_trajectory"] = True
    if sde_indices:
        # GRPO: per-step SDE noise on the resolved step indices. ``num_sde_steps=0``
        # resolves to ``[]`` (not ``None``) via ``resolve_sde_indices``, so gate on
        # non-emptiness, not ``is not None``.
        require(
            sde_label is not None,
            "_to_sglang_kwargs: SDE mode requires sde_label (resolved by engine ctor)",
        )
        kwargs["rollout_sde_type"] = sde_label
        kwargs["rollout_noise_level"] = eta
        kwargs["rollout_sde_step_indices"] = sde_indices
    else:
        # DiffusionNFT / forward-process: deterministic ODE rollout — no per-step
        # SDE noise, so ``rollout_sde_step_indices`` is omitted (the kernel
        # ignores it for ode and reads the field via ``getattr(..., None)``).
        #
        # ``rollout_log_prob_no_const=True`` is REQUIRED whenever
        # ``rollout_sde_type="ode"``: an ODE step is deterministic, so its
        # normalized log-prob is undefined and the kernel asserts this flag is set
        # ("p_ode is always 0, true log_prob is meaningless"). It only makes the
        # kernel emit a zero-log-prob placeholder, which this path never reads
        # (``emit_native_logprob`` is False when ``sde_indices`` is empty). Keep it
        # OFF the SDE branch above, where True would strip the normalization
        # constants from GRPO's native log-prob.
        kwargs["rollout_sde_type"] = "ode"
        kwargs["rollout_noise_level"] = eta
        kwargs["rollout_log_prob_no_const"] = True

    if num_outputs_per_prompt is not None:
        kwargs["num_outputs_per_prompt"] = num_outputs_per_prompt

    return kwargs


__all__ = ["_to_sglang_kwargs", "_deexpand_prompts_from_groups"]
