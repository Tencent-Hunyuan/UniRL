"""HunyuanImage-3 family: input/output sub-adapters + the six modality classes.

The modality classes are thin binders — identity knobs + two constructor
calls — and delegate the conversion verbs to their sub-adapters:

- :class:`Hi3InputAdapter` builds the AR-bearing request side shared by
  t2i / it2i / i2t / t2t / ar_recaption. ``build_inputs`` mirrors the
  official vllm-omni end-to-end inference example
  (``examples/offline_inference/hunyuan_image3/end2end.py``) — the canonical
  reference for the per-prompt dict shape::

      {"prompt_token_ids": ids, "prompt": raw_user_text,
       "use_system_prompt": sys_type, "modalities": [...],
       # image-conditioned: "multi_modal_data": {"image": pil},
       "height": h, "width": w}

- :class:`Hi3DitRecaptionInputAdapter` is the two-engine trainer's
  standalone-DiT request side (externally-injected recaption).
- :class:`Hi3TextOutputAdapter` fills the AR generation Part;
  :class:`Hi3ImageOutputAdapter` fills the AR and diffusion Parts independently;
  :class:`Hi3DitRecaptionOutputAdapter` fills one diffusion Part — the latter
  two derive from the shared
  :class:`~.dit.DitOutputAdapter` skeleton.

The HI3 chat-template knowledge (``task_key`` / ``sys_type`` /
``output_modalities``, mirroring upstream ``_TASK_PRESETS``) rides the
:class:`Hi3InputAdapter` constructor — one row per modality binder.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from unirl.models.hunyuan_image3.conditions import HunyuanImage3FusedMultimodalCondition
from unirl.rollout.engine.vllm_omni.adapters.base import ModelAdapter, register_adapter
from unirl.rollout.engine.vllm_omni.adapters.dit import DitOutputAdapter
from unirl.rollout.engine.vllm_omni.backends import (
    STAGE_KIND_AR,
    STAGE_KIND_DIFFUSION,
    GenerateCall,
    OmniRawResult,
    StageSampling,
)
from unirl.rollout.engine.vllm_omni.utils import (
    build_ar_segment,
    collect_dit_outputs,
    decoded_text_from_ar,
    seed_from_sample_id,
)
from unirl.rollout.engine.vllm_omni.utils.diff_kwargs import core_diff_kwargs, sde_extra_args
from unirl.types.primitives import Texts
from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams, DiffusionSamplingParams


def _trailing_gen_parts(sample: Sample, *params_types: type, caller: str) -> Tuple[Part, ...]:
    """Validate and return the current generation-stage suffix.

    HI3 has both one-stage ``[..., AR]`` / ``[..., Diffusion]`` requests and
    two-stage ``[..., AR, Diffusion]`` requests. Matching from the whole
    trajectory would select an earlier agent turn when either type repeats.
    """
    count = len(params_types)
    if len(sample.parts) < count:
        raise ValueError(f"{caller}: expected {count} trailing generation Parts, got {len(sample.parts)} total Parts")
    start = len(sample.parts) - count
    trailing = tuple(sample.parts[start:])
    for offset, (part, params_type) in enumerate(zip(trailing, params_types)):
        index = start + offset
        if not part.is_gen:
            raise ValueError(f"{caller}: trailing Part at index {index} is not generated")
        if not isinstance(part.sampling_params, params_type):
            raise ValueError(
                f"{caller}: trailing Part at index {index} has sampling_params of type "
                f"{type(part.sampling_params).__name__}, expected {params_type.__name__}"
            )
    return trailing


def _build_prompt_entries(
    texts: Texts,
    *,
    task: str,
    sys_type: str,
    modalities_field: Tuple[str, ...],
    tokenize_fn: Optional[Callable[..., List[int]]],
    decorate: Callable[[Dict[str, Any], int], None],
) -> List[Dict[str, Any]]:
    """Build the HI3 per-prompt dicts shared by the AR-bearing modalities.

    Each entry carries the official ``end2end.py`` base fields
    (``prompt_token_ids`` / ``prompt`` / ``use_system_prompt`` /
    ``modalities``); the ``decorate`` callback then attaches the
    modality-specific extras (``multi_modal_data``, ``height`` / ``width``).
    """
    if tokenize_fn is None:
        raise RuntimeError("build_prompt_entries: tokenize_fn not provided (AR modalities need the driver tokenizer)")
    prompts: List[Dict[str, Any]] = []
    for i, text in enumerate(texts.texts):
        token_ids = tokenize_fn(text, task=task, sys_type=sys_type)
        entry: Dict[str, Any] = {
            "prompt_token_ids": token_ids,
            "prompt": text,
            "use_system_prompt": sys_type,
            "modalities": list(modalities_field),
        }
        decorate(entry, i)
        prompts.append(entry)
    return prompts


def hi3_fused_conditions(diff_outputs: List[OmniRawResult], *, modality: str) -> Dict[str, Any]:
    """The HI3 DiT replay conditions — concat the ``fused_mm_capture`` dicts.

    Reads ``custom_output["fused_mm_capture"]`` — written by
    ``RLHunyuanImage3Pipeline`` after intercepting
    ``prepare_inputs_for_generation``. For think_recaption mode different
    prompts produce different AR output lengths → different ``L`` per
    capture; right-pad shorter sequences to ``max_L`` (pad 0 for input_ids,
    False for masks) so the dim-0 concat works. t2i scope: the it2i
    ``cond_*`` fields stay unpopulated. ``rope_cache`` is deliberately NOT
    shipped: the engine's rope tables use vllm-omni's own layout and are not
    compatible with the HF-side replay forward — replay rebuilds rope
    natively from ``gen_image_mask`` (see ``models/hunyuan_image3/diffusion.py``).
    """
    captures = [(getattr(d, "custom_output", None) or {}).get("fused_mm_capture") for d in diff_outputs]
    if any(c is None for c in captures):
        raise RuntimeError(
            f"build_response: HI3 rollout (modality={modality!r}) "
            "returned no 'fused_mm_capture' on DiffusionOutput.custom_output. "
            "Check that RLHunyuanImage3Pipeline.prepare_inputs_for_generation "
            "hook ran in every DiT worker — the subclass swap may not have "
            "taken effect (verify custom_pipeline_args.pipeline_class in "
            "the stage YAML)."
        )

    sequence_lengths = [int(c["input_ids"].shape[-1]) for c in captures]
    max_L = max(sequence_lengths)

    def _pad_to(t: Any, target_L: int, dim: int = -1, value: Any = 0) -> Any:
        if t is None or not isinstance(t, torch.Tensor):
            return t
        cur_L = t.shape[dim]
        if cur_L >= target_L:
            return t
        pad_size = target_L - cur_L
        ndim = t.ndim
        pad_spec = [0] * (2 * ndim)
        actual_dim = dim if dim >= 0 else ndim + dim
        pad_idx = (ndim - 1 - actual_dim) * 2
        pad_spec[pad_idx + 1] = pad_size
        return torch.nn.functional.pad(t, pad_spec, value=value)

    def _pad_attn_mask(mask: Any, target_L: int) -> Any:
        """Pad attention_mask [N, 1, L, L] → [N, 1, target_L, target_L]."""
        if mask is None or not isinstance(mask, torch.Tensor):
            return mask
        if mask.shape[-1] >= target_L:
            return mask
        N, H, L, _ = mask.shape
        padded = torch.zeros(N, H, target_L, target_L, dtype=mask.dtype, device=mask.device)
        padded[:, :, :L, :L] = mask
        return padded

    padded_captures = []
    for c, L_i in zip(captures, sequence_lengths):
        if L_i == max_L:
            padded_captures.append(c)
        else:
            padded_captures.append(
                {
                    "input_ids": _pad_to(c["input_ids"], max_L, dim=-1, value=0),
                    "attention_mask": _pad_attn_mask(c.get("attention_mask"), max_L),
                    "position_ids": _pad_to(c.get("position_ids"), max_L, dim=-1, value=0),
                    "gen_image_mask": _pad_to(c.get("gen_image_mask"), max_L, dim=-1, value=False),
                    "gen_timestep_scatter_index": c.get("gen_timestep_scatter_index"),
                }
            )

    fused_dict: Dict[str, Any] = {
        "input_ids": torch.cat([c["input_ids"] for c in padded_captures], dim=0),
        "attention_mask": torch.cat([c["attention_mask"] for c in padded_captures], dim=0),
        "position_ids": torch.cat([c["position_ids"] for c in padded_captures], dim=0),
        "gen_image_mask": torch.cat([c["gen_image_mask"] for c in padded_captures], dim=0),
        "gen_timestep_scatter_index": torch.cat([c["gen_timestep_scatter_index"] for c in padded_captures], dim=0),
    }
    return {"fused": HunyuanImage3FusedMultimodalCondition.from_dict(fused_dict)}


def hi3_ar_fused_conditions(per_request: List[List[OmniRawResult]]) -> Dict[str, Any]:
    """The AR Part replay conditions for the recaption producer.

    ARGRPO.replay teacher-forces over prompt+response; it needs the prompt
    token ids (``conditions['fused'].input_ids``). vLLM runs prompts
    per-request with no batch padding, so each Stage-0 output's
    ``prompt_token_ids`` is the sample's TRUE, un-padded prompt. Right-pad to
    ``[B, max_len]`` and carry each sample's true length in the dedicated 1D
    ``prompt_lengths`` [B] field (NOT ``attention_mask`` — that's typed 4D
    and its concat does a 4D unpack). The teacher-forced replay slices
    ``input_ids[b, :prompt_lengths[b]]``, so the right-pad never leaks.
    Returns ``{}`` if no Stage-0 output carries prompt tokens.
    """
    rows: List[List[int]] = []
    for outputs in per_request:
        ids = None
        for out in outputs:
            if getattr(out, "stage_id", None) == 0:
                ids = getattr(out, "prompt_token_ids", None)
                break
        rows.append([int(t) for t in ids] if ids else [])

    if not any(rows):
        return {}

    bsz = len(rows)
    max_len = max(len(r) for r in rows)
    input_ids = torch.zeros((bsz, max_len), dtype=torch.long)
    prompt_lengths = torch.zeros((bsz,), dtype=torch.long)
    for b, r in enumerate(rows):
        if r:
            input_ids[b, : len(r)] = torch.tensor(r, dtype=torch.long)
            prompt_lengths[b] = len(r)
    return {"fused": HunyuanImage3FusedMultimodalCondition(input_ids=input_ids, prompt_lengths=prompt_lengths)}


class Hi3InputAdapter:
    """Request ``Sample`` → HI3 AR-bearing :class:`GenerateCall` (one, whole batch).

    One class covers every AR-bearing HI3 modality; the constructor row says
    what varies:

    - ``task_key`` / ``sys_type`` / ``output_modalities`` — the chat-template
      preset (upstream ``_TASK_PRESETS`` mirror).
    - ``stages`` — ``("ar",)`` or ``("ar", "dit")``: whether a DiT sampling
      stage rides along.
    - ``image_input`` — the request carries ``primitives['image']``; the
      entry gets ``multi_modal_data`` + the PIL's own dims (upstream reads
      h/w off the prompt dict for the image-conditioned paths, matching
      ``end2end.py:185-187``; i2t carries them for parity even without a DiT).
    - ``carries_target_size`` — the entry gets the request's generation
      ``height``/``width`` (t2i's target canvas; ar_recaption's recaption
      prompt needs them although THIS engine never renders).
    - ``bot_task_base`` — when set, ``task_config["bot_task"]``
      think/recaption swaps the trigger tag (``f"{base}_{bot}"``). Kept
      separate from ``modality`` (registry keys are family-namespaced; the
      upstream task vocabulary is not). AR-only modalities leave it ``None``
      — only the two-stage t2i/it2i templates have think/recaption/vanilla
      variants.
    - ``vanilla_task`` — t2i only: ``bot_task == "vanilla"`` pins BOTH the
      task and the system preset (upstream pairs t2i_vanilla with
      en_vanilla).
    """

    def __init__(
        self,
        modality: str,
        *,
        tokenize_fn: Optional[Callable[..., List[int]]],
        task_key: str,
        output_modalities: Tuple[str, ...],
        stages: Tuple[str, ...],
        image_input: bool = False,
        carries_target_size: bool = False,
        bot_task_base: Optional[str] = None,
        vanilla_task: Optional[Tuple[str, str]] = None,
        sys_type: str = "en_unified",
    ) -> None:
        self.modality = modality
        self.tokenize_fn = tokenize_fn
        self.task_key = task_key
        self.output_modalities = tuple(output_modalities)
        self.stages = tuple(stages)
        self.image_input = image_input
        self.carries_target_size = carries_target_size
        self.bot_task_base = bot_task_base
        self.vanilla_task = vanilla_task
        self.sys_type = sys_type

    def _resolve_task(self, task_config: Dict[str, Any]) -> Tuple[str, str]:
        """Resolve ``(task_key, sys_type)`` with the ``task_config`` overrides."""
        sys_type = task_config.get("sys_type") or self.sys_type
        bot_task = task_config.get("bot_task")
        if self.bot_task_base and bot_task:
            if bot_task == "vanilla" and self.vanilla_task is not None:
                return self.vanilla_task
            if bot_task in ("think", "recaption"):
                return f"{self.bot_task_base}_{bot_task}", sys_type
        return self.task_key, sys_type

    def _stage_parts(self, sample: Sample) -> Tuple[Part, Optional[Part]]:
        """Return this call's AR Part and optional diffusion Part from its suffix."""
        caller = f"{type(self).__name__}({self.modality})"
        if "dit" in self.stages:
            ar_part, diff_part = _trailing_gen_parts(
                sample,
                ARSamplingParams,
                DiffusionSamplingParams,
                caller=caller,
            )
            return ar_part, diff_part
        if self.carries_target_size:
            diff_part, ar_part = _trailing_gen_parts(
                sample,
                DiffusionSamplingParams,
                ARSamplingParams,
                caller=caller,
            )
            return ar_part, diff_part
        return sample.frontier_gen_part(ARSamplingParams), None

    def build_prompts(self, sample: Sample) -> List[Dict[str, Any]]:
        """The HI3 chat-templated per-prompt entries (+ the image gates).

        Request routing (the ``bot_task`` / ``sys_type`` controls) now rides
        the input Part's ``control`` bag (see ``unirl/types/README.md``).
        """
        task, sys_type = self._resolve_task(sample.parts[0].control or {})

        if self.image_input:
            turns, images = sample.vision_conditioning()
            texts = next(t.content for t in turns if isinstance(t.content, Texts))
            pil_images = images[0].to_pils()
        else:
            texts = sample.text_conditioning()[0].content
            pil_images = []

        _, diff_part = self._stage_parts(sample)
        diff_params = diff_part.sampling_params if diff_part is not None else None
        return _build_prompt_entries(
            texts,
            task=task,
            sys_type=sys_type,
            modalities_field=self.output_modalities,
            tokenize_fn=self.tokenize_fn,
            decorate=lambda entry, i: self._decorate(entry, i, pil_images=pil_images, diff_params=diff_params),
        )

    def build_sampling(self, sample: Sample) -> List[StageSampling]:
        """AR always; a DiT stage rides along iff ``"dit" in self.stages``."""
        ar_part, diff_part = self._stage_parts(sample)
        ar_params = ar_part.sampling_params
        sampling = [self._ar_sampling(ar_params)]
        if "dit" in self.stages:
            assert diff_part is not None
            sampling.append(self._dit_sampling(diff_part, diff_part.sampling_params))
        return sampling

    def build(self, sample: Sample) -> List[GenerateCall]:
        prompts = self.build_prompts(sample)
        ar_part, diff_part = self._stage_parts(sample)
        if "dit" not in self.stages or diff_part is None:
            return [GenerateCall(prompts=prompts, sampling=self.build_sampling(sample))]

        diff_params = diff_part.sampling_params
        explicit_keys = list(diff_part.init_noise_group_ids or [])
        share = bool(getattr(diff_params, "init_same_noise", False))
        recipe_keys = explicit_keys or (diff_part.group_ids if share else list(diff_part.sample_ids))
        if bool(getattr(diff_params, "disable_driver_xt", False)):
            recipe_keys = []
        if not recipe_keys:
            return [GenerateCall(prompts=prompts, sampling=self.build_sampling(sample))]
        if len(recipe_keys) != len(prompts):
            raise ValueError(
                f"{self.modality}: x_T recipe key count {len(recipe_keys)} != prompt count {len(prompts)}."
            )

        # Slice the generation part so each B=1 request gets its own noise key.
        calls: List[GenerateCall] = []
        for index, prompt in enumerate(prompts):
            one = diff_part.slice(index, index + 1)
            calls.append(
                GenerateCall(
                    prompts=[prompt],
                    sampling=[
                        self._ar_sampling(ar_part.sampling_params),
                        self._dit_sampling(one, one.sampling_params),
                    ],
                )
            )
        return calls

    def _decorate(self, entry: Dict[str, Any], i: int, *, pil_images: List[Any], diff_params: Any) -> None:
        """The per-entry extras, derived from the constructor flags."""
        if self.image_input:
            pil = pil_images[i]
            entry["multi_modal_data"] = {"image": pil}
            entry["height"] = pil.height
            entry["width"] = pil.width
        elif self.carries_target_size:
            entry["height"] = int(diff_params.height)
            entry["width"] = int(diff_params.width)

    def _ar_sampling(self, ar_params: Any) -> StageSampling:
        """AR sampling intent (Stage 0). ``logprobs=1`` makes vLLM emit
        per-token logp on the sampled token (read by ``build_ar_segment``).
        ``ar_params`` is the request's ``ARSamplingParams`` — the engine keeps
        no AR sampling defaults (NB the dataclass field is ``max_new_tokens``)."""
        return StageSampling(
            kind=STAGE_KIND_AR,
            kwargs=dict(
                temperature=float(ar_params.temperature),
                top_p=float(ar_params.top_p),
                top_k=int(ar_params.top_k),
                max_tokens=int(ar_params.max_new_tokens),
                logprobs=1,
            ),
        )

    def _dit_sampling(self, gen_part: Part, diff_params: Any) -> StageSampling:
        diff_kwargs = core_diff_kwargs(diff_params)
        seed = getattr(diff_params, "seed", None)
        if seed is not None:
            diff_kwargs["seed"] = int(seed)

        extra_args = sde_extra_args(diff_params)

        seg = gen_part.segment
        if getattr(seg, "initial_latents", None) is not None:
            raise NotImplementedError(
                f"{type(self).__name__}: modality={self.modality!r} cannot consume a "
                f"pre-materialized LatentSegment.initial_latents tensor "
                f"(HI3 DiT latent shape is AR-dynamic). Ship the x_T RECIPE via the "
                f"gen Part's lineage instead."
            )

        # Ship noise IDs and seed; the pipeline fills shape before regeneration.
        explicit_keys = list(gen_part.init_noise_group_ids or [])
        share = bool(getattr(diff_params, "init_same_noise", False))
        keys = explicit_keys or (gen_part.group_ids if share else list(gen_part.sample_ids))
        if keys and not bool(getattr(diff_params, "disable_driver_xt", False)):
            extra_args["init_noise_group_ids"] = [str(g) for g in keys]
            extra_args["init_noise_seed"] = int(seed) if seed is not None else 0

        if extra_args:
            diff_kwargs["extra_args"] = extra_args

        return StageSampling(kind=STAGE_KIND_DIFFUSION, kwargs=diff_kwargs)


class Hi3DitRecaptionInputAdapter:
    """Standalone HI3 DiT request side — eats an externally-injected recaption.

    The two-engine trainer chains the AR-generated recaption per sample as a
    ``cot_text`` input Part (``Texts``, aligned 1:1 with the prompt Part; see
    :func:`cot_text_from_sample`). Each per-prompt dict carries
    ``extra['ar_generated_text']`` — exactly the
    key the upstream DiT ``forward`` reads as ``cot_text`` — plus
    ``use_system_prompt`` so the DiT rebuilds the same system prefix the AR
    used.

    **One call per prompt, seeded here.** Per-image distinct seeds cannot
    travel through the sampling params: vllm-omni requires one params object
    per STAGE (not per prompt) and shares it across all prompts of a
    ``generate()`` call — ``OmniDiffusionRequest.__post_init__`` assigns a
    random seed only on the FIRST request and the mutated object poisons the
    rest (byte-identical images → diffusion advantage 0). So ``build``
    emits one single-prompt :class:`GenerateCall` per sample with its own
    ``seed_from_sample_id`` seed and its own x_T recipe gid slice (a shared
    full-batch gid list would make the worker's ``NoiseRecipe.for_batch(1)``
    hand gids[0] to EVERY image).

    Deliberately does NOT use the ``build_prompts`` / ``build_sampling``
    pair: prompts and sampling are paired per single-prompt call (seed + gid
    slice decided together), so a wholesale ``build`` keeps them co-located.
    """

    def __init__(self, modality: str, *, sys_type: str = "en_unified") -> None:
        self.modality = modality
        self.sys_type = sys_type

    def build(self, sample: Sample) -> List[GenerateCall]:
        if sample.has_image_input():
            raise ValueError(f"modality={self.modality!r} does not accept an image input Part")

        turns = sample.text_conditioning()
        texts = turns[0].content
        cot = turns[1].content
        gen_part = sample.frontier_gen_part(DiffusionSamplingParams)
        diff_params = gen_part.sampling_params
        sys_type = (sample.parts[0].control or {}).get("sys_type") or self.sys_type

        base_kwargs = core_diff_kwargs(diff_params)
        height = int(base_kwargs["height"])
        width = int(base_kwargs["width"])

        base_extra = sde_extra_args(diff_params)
        share = bool(getattr(diff_params, "init_same_noise", False))
        recipe_gids = gen_part.group_ids if share else list(gen_part.sample_ids)
        if recipe_gids and not bool(getattr(diff_params, "disable_driver_xt", False)):
            base_extra["init_noise_group_ids"] = [str(g) for g in recipe_gids]
            base_extra["init_noise_seed"] = (
                int(diff_params.seed) if getattr(diff_params, "seed", None) is not None else 0
            )

        calls: List[GenerateCall] = []
        for idx, (sample_id, text, recap) in enumerate(zip(gen_part.sample_ids, texts.texts, cot.texts)):
            prompt = {
                "prompt": text,
                "height": height,
                "width": width,
                "use_system_prompt": sys_type,
                "extra": {"ar_generated_text": recap},
            }
            kwargs = dict(base_kwargs)
            kwargs["seed"] = seed_from_sample_id(sample_id)
            extra_args = dict(base_extra)
            gid = recipe_gids[idx] if idx < len(recipe_gids) else None
            if gid is not None and extra_args.get("init_noise_group_ids"):
                extra_args["init_noise_group_ids"] = [str(gid)]
            if extra_args:
                kwargs["extra_args"] = extra_args
            calls.append(
                GenerateCall(
                    prompts=[prompt],
                    sampling=[StageSampling(kind=STAGE_KIND_DIFFUSION, kwargs=kwargs)],
                    group_by_request_id=False,
                )
            )
        return calls


class Hi3TextOutputAdapter:
    """Per-request AR results → the filled AR generation Part.

    Same direct-Part three-hook shape as :class:`~.dit.DitOutputAdapter`:
    segment, decoded primitive, and replay conditions are derived for the typed
    AR shell and written only to that Part. ``build_decoded`` is deliberately
    NOT best-effort — text is the product here, so broken extraction must raise.
    """

    def __init__(self, modality: str) -> None:
        self.modality = modality

    def build_segment(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Any:
        del sample
        return build_ar_segment(per_request)

    def build_decoded(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Texts:
        del sample
        return decoded_text_from_ar(per_request)

    def build_conditions(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Dict[str, Any]:
        """AR Part conditions. Default: none (no replay capture in scope)."""
        del sample, per_request
        return {}

    def build(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Sample:
        if not per_request or not any(per_request):
            raise ValueError("build_response: empty per-request outputs (Omni.generate returned nothing surfaceable).")
        segment = self.build_segment(sample, per_request)
        decoded = self.build_decoded(sample, per_request)
        conditions = self.build_conditions(sample, per_request)
        if segment is None:
            return sample
        frontier = sample.frontier_gen_part(ARSamplingParams)
        return sample.replace_frontier(
            frontier.fill(
                segment=segment,
                primitives={"text": decoded},
                conditions=dict(conditions),
            )
        )


class Hi3ArRecaptionOutputAdapter(Hi3TextOutputAdapter):
    """AR Part response + the ARGRPO fused prompt-capture conditions."""

    def build_conditions(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Dict[str, Any]:
        del sample
        return hi3_ar_fused_conditions(per_request)


class Hi3ImageOutputAdapter(DitOutputAdapter):
    """Two-stage HI3 response: fill AR and diffusion Parts independently."""

    def __init__(self, modality: str) -> None:
        super().__init__(modality, stage_id=1)

    def build_conditions(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Dict[str, Any]:
        del sample
        diff_outputs, _, _ = collect_dit_outputs(
            per_request, final_output_type=self.final_output_type, stage_id=self.stage_id, modality=self.modality
        )
        return hi3_fused_conditions(diff_outputs, modality=self.modality)

    def build(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Sample:
        filled = super().build(sample, per_request)

        ar_segment = build_ar_segment(per_request)
        if ar_segment is None:
            return filled

        ar_decoded = None
        try:
            ar_decoded = decoded_text_from_ar(per_request)
        except Exception:
            # Best-effort parity: a decoded-text failure must not lose the image rollout.
            ar_decoded = None
        ar_conditions = hi3_ar_fused_conditions(per_request)
        ar_part, _ = _trailing_gen_parts(
            filled,
            ARSamplingParams,
            DiffusionSamplingParams,
            caller=f"{type(self).__name__}({self.modality}).build",
        )
        idx = len(filled.parts) - 2
        parts = list(filled.parts)
        parts[idx] = ar_part.fill(
            segment=ar_segment,
            primitives={"text": ar_decoded} if ar_decoded is not None else {},
            conditions=dict(ar_conditions),
        )
        return filled.with_parts(parts)


class Hi3DitRecaptionOutputAdapter(DitOutputAdapter):
    """Single diffusion-Part response of the standalone HI3 DiT (Stage 0)."""

    def build_conditions(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Dict[str, Any]:
        del sample
        diff_outputs, _, _ = collect_dit_outputs(
            per_request, final_output_type=self.final_output_type, stage_id=self.stage_id, modality=self.modality
        )
        return hi3_fused_conditions(diff_outputs, modality=self.modality)


@register_adapter("hi3_t2i")
class Hi3T2iAdapter(ModelAdapter):
    """HI3 text → AR think → DiT image."""

    stage_yaml = "hunyuan_image3_t2i_rl.yaml"
    omni_mode = "text-to-image"
    ar_lora_passthrough = True
    clear_cuda_visible = True

    def __init__(self, config: Any, model_config: Any, *, strategy: Any = None, tokenize_fn: Any = None) -> None:
        super().__init__(config, model_config, strategy=strategy, tokenize_fn=tokenize_fn)
        self.input_adapter = Hi3InputAdapter(
            self.modality,
            tokenize_fn=self.tokenize_fn,
            task_key="t2i_think",
            output_modalities=("image",),
            stages=("ar", "dit"),
            carries_target_size=True,
            bot_task_base="t2i",
            vanilla_task=("t2i_vanilla", "en_vanilla"),
        )
        self.output_adapter = Hi3ImageOutputAdapter(self.modality)

    def validate_request(self, sample: Sample) -> None:
        if sample.has_image_input():
            raise ValueError(
                f"modality={self.modality!r} rejects image-bearing requests; use an image-conditioned modality instead."
            )

    def build_inputs(self, sample: Sample) -> List[GenerateCall]:
        return self.input_adapter.build(sample)

    def build_response(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Sample:
        return self.output_adapter.build(sample, per_request)


@register_adapter("hi3_it2i")
class Hi3It2iAdapter(ModelAdapter):
    """HI3 image+text → AR recaption → DiT edited image."""

    stage_yaml = "hunyuan_image3_it2i_rl.yaml"
    omni_mode = "text-to-image"
    ar_lora_passthrough = True
    clear_cuda_visible = True

    def __init__(self, config: Any, model_config: Any, *, strategy: Any = None, tokenize_fn: Any = None) -> None:
        super().__init__(config, model_config, strategy=strategy, tokenize_fn=tokenize_fn)
        self.input_adapter = Hi3InputAdapter(
            self.modality,
            tokenize_fn=self.tokenize_fn,
            task_key="it2i_think",
            output_modalities=("image",),
            stages=("ar", "dit"),
            image_input=True,
            bot_task_base="it2i",
        )
        self.output_adapter = Hi3ImageOutputAdapter(self.modality)

    def validate_request(self, sample: Sample) -> None:
        if not sample.has_image_input():
            raise ValueError(
                f"modality={self.modality!r} requires an image input Part (Images primitive) "
                "chained off the prompt (Part.input_child); none found."
            )

    def build_inputs(self, sample: Sample) -> List[GenerateCall]:
        return self.input_adapter.build(sample)

    def build_response(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Sample:
        return self.output_adapter.build(sample, per_request)


@register_adapter("hi3_i2t")
class Hi3I2tAdapter(ModelAdapter):
    """HI3 image+text → AR text (upstream comprehension YAML)."""

    stage_yaml = "hunyuan_image3_i2t.yaml"
    stage_yaml_source = "upstream"
    needs_sigmas = False
    ar_lora_passthrough = True
    clear_cuda_visible = True

    def __init__(self, config: Any, model_config: Any, *, strategy: Any = None, tokenize_fn: Any = None) -> None:
        super().__init__(config, model_config, strategy=strategy, tokenize_fn=tokenize_fn)
        self.input_adapter = Hi3InputAdapter(
            self.modality,
            tokenize_fn=self.tokenize_fn,
            task_key="i2t",
            output_modalities=("text",),
            stages=("ar",),
            image_input=True,
        )
        self.output_adapter = Hi3TextOutputAdapter(self.modality)

    def validate_request(self, sample: Sample) -> None:
        if not sample.has_image_input():
            raise ValueError(
                f"modality={self.modality!r} requires an image input Part (Images primitive) "
                "chained off the prompt (Part.input_child); none found."
            )

    def build_inputs(self, sample: Sample) -> List[GenerateCall]:
        return self.input_adapter.build(sample)

    def build_response(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Sample:
        return self.output_adapter.build(sample, per_request)


@register_adapter("hi3_t2t")
class Hi3T2tAdapter(ModelAdapter):
    """HI3 text → AR text (upstream comprehension YAML)."""

    stage_yaml = "hunyuan_image3_t2t.yaml"
    stage_yaml_source = "upstream"
    needs_sigmas = False
    ar_lora_passthrough = True
    clear_cuda_visible = True

    def __init__(self, config: Any, model_config: Any, *, strategy: Any = None, tokenize_fn: Any = None) -> None:
        super().__init__(config, model_config, strategy=strategy, tokenize_fn=tokenize_fn)
        self.input_adapter = Hi3InputAdapter(
            self.modality,
            tokenize_fn=self.tokenize_fn,
            task_key="t2t",
            output_modalities=("text",),
            stages=("ar",),
        )
        self.output_adapter = Hi3TextOutputAdapter(self.modality)

    def validate_request(self, sample: Sample) -> None:
        if sample.has_image_input():
            raise ValueError(
                f"modality={self.modality!r} rejects image-bearing requests; use modality='hi3_i2t' instead."
            )

    def build_inputs(self, sample: Sample) -> List[GenerateCall]:
        return self.input_adapter.build(sample)

    def build_response(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Sample:
        return self.output_adapter.build(sample, per_request)


@register_adapter("hi3_ar_recaption")
class Hi3ArRecaptionAdapter(ModelAdapter):
    """Two-engine trainer's AR think/recaption producer.

    Builds the same think/recaption prompt as ``t2i`` (``task_key``
    ``t2i_think``) but is served by an AR-only stage. Needs composed
    sampling: the recaption prompt carries the DiT generation dims, read off
    the request's ``diffusion.height`` / ``diffusion.width``.
    """

    stage_yaml = "hunyuan_image3_ar_recaption_rl.yaml"
    needs_sigmas = False
    ar_lora_passthrough = True
    clear_cuda_visible = True
    lora_copy_transport = True

    def __init__(self, config: Any, model_config: Any, *, strategy: Any = None, tokenize_fn: Any = None) -> None:
        super().__init__(config, model_config, strategy=strategy, tokenize_fn=tokenize_fn)
        self.input_adapter = Hi3InputAdapter(
            self.modality,
            tokenize_fn=self.tokenize_fn,
            task_key="t2i_think",
            output_modalities=("image",),
            stages=("ar",),
            carries_target_size=True,
        )
        self.output_adapter = Hi3ArRecaptionOutputAdapter(self.modality)

    def build_inputs(self, sample: Sample) -> List[GenerateCall]:
        return self.input_adapter.build(sample)

    def build_response(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Sample:
        return self.output_adapter.build(sample, per_request)


@register_adapter("hi3_dit_recaption")
class Hi3DitRecaptionAdapter(ModelAdapter):
    """Standalone HI3 DiT — the two-engine trainer's image half."""

    stage_yaml = "hunyuan_image3_dit_recaption_rl.yaml"
    omni_mode = "text-to-image"
    clear_cuda_visible = True
    # HI3 two-engine stages are TP>1 — wake-time LoRA re-push must use the byte-copy transport.
    lora_copy_transport = True

    def __init__(self, config: Any, model_config: Any, *, strategy: Any = None, tokenize_fn: Any = None) -> None:
        super().__init__(config, model_config, strategy=strategy, tokenize_fn=tokenize_fn)
        self.input_adapter = Hi3DitRecaptionInputAdapter(self.modality)
        self.output_adapter = Hi3DitRecaptionOutputAdapter(self.modality)

    def build_inputs(self, sample: Sample) -> List[GenerateCall]:
        return self.input_adapter.build(sample)

    def build_response(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Sample:
        return self.output_adapter.build(sample, per_request)


__all__ = [
    "Hi3ArRecaptionAdapter",
    "Hi3ArRecaptionOutputAdapter",
    "Hi3DitRecaptionAdapter",
    "Hi3DitRecaptionInputAdapter",
    "Hi3DitRecaptionOutputAdapter",
    "Hi3I2tAdapter",
    "Hi3ImageOutputAdapter",
    "Hi3InputAdapter",
    "Hi3It2iAdapter",
    "Hi3T2iAdapter",
    "Hi3T2tAdapter",
    "Hi3TextOutputAdapter",
    "hi3_ar_fused_conditions",
    "hi3_fused_conditions",
]
