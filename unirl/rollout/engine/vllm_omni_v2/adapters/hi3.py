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
- :class:`Hi3TextOutputAdapter` packs the single-"ar"-track response;
  :class:`Hi3ImageOutputAdapter` the two-track (ar root + image child)
  response; :class:`Hi3DitRecaptionOutputAdapter` the single-"image"-track
  response — the latter two derive from the shared
  :class:`~.dit.DitOutputAdapter` skeleton.

The HI3 chat-template knowledge (``task_key`` / ``sys_type`` /
``output_modalities``, mirroring upstream ``_TASK_PRESETS``) rides the
:class:`Hi3InputAdapter` constructor — one row per modality binder.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from unirl.rollout.engine.vllm_omni_v2.adapters.base import ModelAdapter, register_adapter
from unirl.rollout.engine.vllm_omni_v2.adapters.dit import DitOutputAdapter
from unirl.rollout.engine.vllm_omni_v2.backends import (
    STAGE_KIND_AR,
    STAGE_KIND_DIFFUSION,
    GenerateCall,
    OmniRawResult,
    StageSampling,
)
from unirl.rollout.engine.vllm_omni_v2.utils import (
    assemble_tracks,
    build_ar_fused_condition,
    build_ar_segment,
    build_fused_mm_condition,
    decoded_text_from_ar,
    pil_images_from_req,
    seed_from_sample_id,
    texts_from_req,
)
from unirl.rollout.engine.vllm_omni_v2.utils.diff_kwargs import core_diff_kwargs, sde_extra_args
from unirl.types.primitives import Texts
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp
from unirl.types.sampling import get_ar_params, get_diffusion_params

# --------------------------------------------------------------------------- #
# Chat-template prompt construction
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# Replay-condition extractors
# --------------------------------------------------------------------------- #


def hi3_fused_conditions(diff_outputs: List[OmniRawResult], *, modality: str) -> Dict[str, Any]:
    """The HI3 DiT replay conditions — fused multimodal capture."""
    fused = build_fused_mm_condition(diff_outputs)
    if fused is None:
        raise RuntimeError(
            f"build_response: HI3 rollout (modality={modality!r}) "
            "returned no 'fused_mm_capture' on DiffusionOutput.custom_output. "
            "Check that RLHunyuanImage3Pipeline.prepare_inputs_for_generation "
            "hook ran in every DiT worker — the subclass swap may not have "
            "taken effect (verify custom_pipeline_args.pipeline_class in "
            "the stage YAML)."
        )
    return {"fused": fused}


def hi3_ar_fused_conditions(per_request: List[List[OmniRawResult]]) -> Dict[str, Any]:
    """The AR-track replay conditions for the recaption producer.

    ARGRPO.replay teacher-forces over prompt+response; it needs the prompt
    token ids (``conditions['fused'].input_ids``). vLLM processes each
    request's prompt independently (no batch padding), so the output's
    ``prompt_token_ids`` is the sample's true, un-padded prompt.
    """
    ar_fused = build_ar_fused_condition(per_request)
    return {"fused": ar_fused} if ar_fused is not None else {}


# --------------------------------------------------------------------------- #
# Input sub-adapters
# --------------------------------------------------------------------------- #


class Hi3InputAdapter:
    """``RolloutReq`` → HI3 AR-bearing :class:`GenerateCall` (one, whole batch).

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
    - ``bot_task_base`` — when set, ``stage_config["bot_task"]``
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

    def _resolve_task(self, stage_config: Dict[str, Any]) -> Tuple[str, str]:
        """Resolve ``(task_key, sys_type)`` with the ``stage_config`` overrides."""
        sys_type = stage_config.get("sys_type") or self.sys_type
        bot_task = stage_config.get("bot_task")
        if self.bot_task_base and bot_task:
            if bot_task == "vanilla" and self.vanilla_task is not None:
                return self.vanilla_task
            if bot_task in ("think", "recaption"):
                return f"{self.bot_task_base}_{bot_task}", sys_type
        return self.task_key, sys_type

    def build(self, req: RolloutReq) -> List[GenerateCall]:
        task, sys_type = self._resolve_task(req.stage_config or {})

        texts = texts_from_req(req)
        n = len(texts.texts)

        pil_images = pil_images_from_req(req, n) if self.image_input else []
        if self.image_input and not pil_images:
            raise ValueError(f"modality={self.modality!r} requires req.primitives['image']")
        if not self.image_input and req.primitives.get("image") is not None:
            raise ValueError(f"modality={self.modality!r} does not accept req.primitives['image']")

        diff_params = get_diffusion_params(req.sampling_params)
        ar_params = get_ar_params(req.sampling_params)

        prompts = _build_prompt_entries(
            texts,
            task=task,
            sys_type=sys_type,
            modalities_field=self.output_modalities,
            tokenize_fn=self.tokenize_fn,
            decorate=lambda entry, i: self._decorate(entry, i, pil_images=pil_images, diff_params=diff_params),
        )

        sampling = [self._ar_sampling(ar_params)]
        if "dit" in self.stages:
            sampling.append(self._dit_sampling(req, diff_params))
        return [GenerateCall(prompts=prompts, sampling=sampling)]

    def _decorate(self, entry: Dict[str, Any], i: int, *, pil_images: List[Any], diff_params: Any) -> None:
        """The per-entry extras, derived from the constructor flags."""
        if self.image_input:
            # Upstream HI3 reads h/w off the prompt dict for the
            # image-conditioned paths — the PIL dims, not the request's.
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

    def _dit_sampling(self, req: RolloutReq, diff_params: Any) -> StageSampling:
        diff_kwargs = core_diff_kwargs(req, diff_params)
        seed = getattr(diff_params, "seed", None)
        if seed is not None:
            diff_kwargs["seed"] = int(seed)

        extra_args = sde_extra_args(diff_params)

        # HI3's DiT latent shape is AR-dynamic (only known in-worker after
        # stage 0), so the driver cannot ship a materialized x_T tensor.
        if (req.request_conditions or {}).get("initial_latents") is not None:
            raise NotImplementedError(
                f"{type(self).__name__}: modality={self.modality!r} cannot consume a "
                f"pre-materialized request_conditions['initial_latents'] tensor "
                f"(HI3 DiT latent shape is AR-dynamic). Ship the x_T RECIPE via "
                f"req.init_noise_group_ids instead."
            )

        # Driver-authoritative x_T RECIPE: per-image gids (+ seed; NO shape —
        # the pipeline's prepare_latents hook fills the AR-resolved shape and
        # regenerates the byte-identical x_T via NoiseRecipe.for_batch).
        if req.init_noise_group_ids:
            extra_args["init_noise_group_ids"] = [str(g) for g in req.init_noise_group_ids]
            extra_args["init_noise_seed"] = int(seed) if seed is not None else 0

        if extra_args:
            diff_kwargs["extra_args"] = extra_args

        return StageSampling(kind=STAGE_KIND_DIFFUSION, kwargs=diff_kwargs)


class Hi3DitRecaptionInputAdapter:
    """Standalone HI3 DiT request side — eats an externally-injected recaption.

    The two-engine trainer puts the AR-generated recaption per sample on
    ``req.primitives['cot_text']`` (aligned 1:1 with ``primitives['text']``).
    Each per-prompt dict carries ``extra['ar_generated_text']`` — exactly the
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
    """

    def __init__(self, modality: str, *, sys_type: str = "en_unified") -> None:
        self.modality = modality
        #: System-prompt preset for ``use_system_prompt`` — the only piece of
        #: the HI3 chat-template row this DiT-only stage consumes (no task:
        #: the recaption text is injected via ``extra['ar_generated_text']``).
        self.sys_type = sys_type

    def build(self, req: RolloutReq) -> List[GenerateCall]:
        if req.primitives.get("image") is not None:
            raise ValueError(f"modality={self.modality!r} does not accept req.primitives['image']")

        texts = texts_from_req(req)
        cot = req.primitives.get("cot_text")
        if not isinstance(cot, Texts):
            raise TypeError(
                f"modality={self.modality!r} requires req.primitives['cot_text'] (Texts of recaptions); "
                f"got {type(cot).__name__ if cot is not None else 'None'}."
            )
        if len(cot.texts) != len(texts.texts):
            raise ValueError(f"{self.modality}: cot_text count {len(cot.texts)} != prompt count {len(texts.texts)}.")

        sys_type = (req.stage_config or {}).get("sys_type") or self.sys_type
        diff_params = get_diffusion_params(req.sampling_params)

        base_kwargs = core_diff_kwargs(req, diff_params)
        height = int(base_kwargs["height"])
        width = int(base_kwargs["width"])

        # Base extra_args mirror the v1 builder: sparse SDE indices + the
        # WHOLE batch's x_T recipe gids (+ the regen base seed — distinct from
        # the per-image SAMPLING seed below; per-image x_T variety comes from
        # the gid, not this seed). NO init_noise_latent_shape — HI3's DiT
        # latent shape is AR-dynamic and resolved in the worker.
        base_extra = sde_extra_args(diff_params)
        recipe_gids = list(req.init_noise_group_ids or [])
        if recipe_gids:
            base_extra["init_noise_group_ids"] = [str(g) for g in recipe_gids]
            base_extra["init_noise_seed"] = (
                int(diff_params.seed) if getattr(diff_params, "seed", None) is not None else 0
            )

        calls: List[GenerateCall] = []
        for idx, (sample_id, text, recap) in enumerate(zip(req.sample_ids, texts.texts, cot.texts)):
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
            # Each single-prompt generate runs with batch_size=1 in the
            # worker, so ship ONLY this sample's x_T recipe gid.
            gid = recipe_gids[idx] if idx < len(recipe_gids) else None
            if gid is not None and extra_args.get("init_noise_group_ids"):
                extra_args["init_noise_group_ids"] = [str(gid)]
            if extra_args:
                kwargs["extra_args"] = extra_args
            calls.append(
                GenerateCall(
                    prompts=[prompt],
                    sampling=[StageSampling(kind=STAGE_KIND_DIFFUSION, kwargs=kwargs)],
                    # Single-prompt call: its flat output list IS the group.
                    group_by_request_id=False,
                )
            )
        return calls


# --------------------------------------------------------------------------- #
# Output sub-adapters
# --------------------------------------------------------------------------- #


class Hi3TextOutputAdapter:
    """Per-request AR results → the single-"ar"-track :class:`RolloutResp`.

    ``conditions`` is the optional replay-condition extractor over the raw
    per-request groups — ``None`` for the comprehension modalities,
    :func:`hi3_ar_fused_conditions` for the recaption producer.
    """

    def __init__(
        self,
        modality: str,
        *,
        conditions: Optional[Callable[[List[List[OmniRawResult]]], Dict[str, Any]]] = None,
    ) -> None:
        self.modality = modality
        self._conditions = conditions

    def build(self, req: RolloutReq, per_request: List[List[OmniRawResult]]) -> RolloutResp:
        if not per_request or not any(per_request):
            raise ValueError("build_response: empty per-request outputs (Omni.generate returned nothing surfaceable).")

        decoded_text = decoded_text_from_ar(per_request)
        conditions = self._conditions(per_request) if self._conditions is not None else {}

        segments = {}
        ar_segment = build_ar_segment(per_request)
        if ar_segment is not None:
            segments["ar"] = ar_segment

        return assemble_tracks(
            req,
            segments_for_track=segments,
            decoded_for_track={"ar": decoded_text},
            conditions=conditions,
        )


class Hi3ImageOutputAdapter(DitOutputAdapter):
    """Two-track HI3 response: "ar" root + "image" child, DiT is Stage 1."""

    def __init__(self, modality: str) -> None:
        super().__init__(modality, stage_id=1)

    def conditions(self, diff_outputs: List[OmniRawResult]) -> Dict[str, Any]:
        return hi3_fused_conditions(diff_outputs, modality=self.modality)

    def build_decoded(
        self,
        pil_images: List[Any],
        frame_groups: List[List[Any]],
        per_request: List[List[OmniRawResult]],
    ) -> Dict[str, Any]:
        decoded = super().build_decoded(pil_images, frame_groups, per_request)
        # Surface the AR-generated text (best-effort; don't break rollout if
        # AR text extraction fails).
        try:
            decoded["ar"] = decoded_text_from_ar(per_request)
        except Exception:
            decoded["ar"] = None
        return decoded


class Hi3DitRecaptionOutputAdapter(DitOutputAdapter):
    """Single-"image"-track response of the standalone HI3 DiT (Stage 0)."""

    def conditions(self, diff_outputs: List[OmniRawResult]) -> Dict[str, Any]:
        return hi3_fused_conditions(diff_outputs, modality=self.modality)


# --------------------------------------------------------------------------- #
# Modality binders
# --------------------------------------------------------------------------- #


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

    def validate_request(self, req: RolloutReq) -> None:
        if req.primitives.get("image") is not None:
            raise ValueError(
                f"modality={self.modality!r} rejects image-bearing requests; use an image-conditioned modality instead."
            )

    def build_inputs(self, req: RolloutReq) -> List[GenerateCall]:
        return self.input_adapter.build(req)

    def build_response(self, req: RolloutReq, per_request: List[List[OmniRawResult]]) -> RolloutResp:
        return self.output_adapter.build(req, per_request)


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

    def validate_request(self, req: RolloutReq) -> None:
        if req.primitives.get("image") is None:
            raise ValueError(f"modality={self.modality!r} requires req.primitives['image'].")

    def build_inputs(self, req: RolloutReq) -> List[GenerateCall]:
        return self.input_adapter.build(req)

    def build_response(self, req: RolloutReq, per_request: List[List[OmniRawResult]]) -> RolloutResp:
        return self.output_adapter.build(req, per_request)


@register_adapter("hi3_i2t")
class Hi3I2tAdapter(ModelAdapter):
    """HI3 image+text → AR text (upstream comprehension YAML)."""

    stage_yaml = "hunyuan_image3_i2t.yaml"
    stage_yaml_source = "upstream"
    #: AR-only requests carry ``ARSamplingParams`` with no diffusion sub-block
    #: — ``ensure_req_sigmas`` would raise on them.
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

    def validate_request(self, req: RolloutReq) -> None:
        if req.primitives.get("image") is None:
            raise ValueError(f"modality={self.modality!r} requires req.primitives['image'].")

    def build_inputs(self, req: RolloutReq) -> List[GenerateCall]:
        return self.input_adapter.build(req)

    def build_response(self, req: RolloutReq, per_request: List[List[OmniRawResult]]) -> RolloutResp:
        return self.output_adapter.build(req, per_request)


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

    def validate_request(self, req: RolloutReq) -> None:
        if req.primitives.get("image") is not None:
            raise ValueError(
                f"modality={self.modality!r} rejects image-bearing requests; use modality='hi3_i2t' instead."
            )

    def build_inputs(self, req: RolloutReq) -> List[GenerateCall]:
        return self.input_adapter.build(req)

    def build_response(self, req: RolloutReq, per_request: List[List[OmniRawResult]]) -> RolloutResp:
        return self.output_adapter.build(req, per_request)


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
    #: HI3 two-engine stages are TP>1 — wake-time LoRA re-push must use the
    #: byte-copy transport (a zero-copy handle crashes ranks 2..N).
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
        self.output_adapter = Hi3TextOutputAdapter(self.modality, conditions=hi3_ar_fused_conditions)

    def build_inputs(self, req: RolloutReq) -> List[GenerateCall]:
        return self.input_adapter.build(req)

    def build_response(self, req: RolloutReq, per_request: List[List[OmniRawResult]]) -> RolloutResp:
        return self.output_adapter.build(req, per_request)


@register_adapter("hi3_dit_recaption")
class Hi3DitRecaptionAdapter(ModelAdapter):
    """Standalone HI3 DiT — the two-engine trainer's image half."""

    stage_yaml = "hunyuan_image3_dit_recaption_rl.yaml"
    omni_mode = "text-to-image"
    # v1 loads a driver tokenizer for dit_recaption even though this builder
    # never tokenizes — kept for parity (health semantics, warm cache).
    clear_cuda_visible = True
    #: HI3 two-engine stages are TP>1 — wake-time LoRA re-push must use the
    #: byte-copy transport.
    lora_copy_transport = True

    def __init__(self, config: Any, model_config: Any, *, strategy: Any = None, tokenize_fn: Any = None) -> None:
        super().__init__(config, model_config, strategy=strategy, tokenize_fn=tokenize_fn)
        self.input_adapter = Hi3DitRecaptionInputAdapter(self.modality)
        self.output_adapter = Hi3DitRecaptionOutputAdapter(self.modality)

    def build_inputs(self, req: RolloutReq) -> List[GenerateCall]:
        return self.input_adapter.build(req)

    def build_response(self, req: RolloutReq, per_request: List[List[OmniRawResult]]) -> RolloutResp:
        return self.output_adapter.build(req, per_request)


__all__ = [
    "Hi3ArRecaptionAdapter",
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
