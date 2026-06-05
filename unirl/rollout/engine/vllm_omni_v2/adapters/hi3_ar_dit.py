"""HI3 two-track shape: AR prelude (Stage 0) → DiT denoise (Stage 1).

``build_inputs`` mirrors the official vllm-omni end-to-end inference example
(``examples/offline_inference/hunyuan_image3/end2end.py``) — the canonical
reference for the per-prompt dict shape::

    {"prompt_token_ids": ids, "prompt": raw_user_text,
     "use_system_prompt": sys_type, "modalities": [...],
     # it2i: "multi_modal_data": {"image": pil}, "height": h, "width": w}

The HI3 chat-template knowledge lives here as class attributes
(``task_key`` / ``sys_type`` / ``output_modalities`` — one registry row per
adapter, mirroring upstream ``_TASK_PRESETS``) plus the ``resolve_task``
hook for the ``stage_config`` overrides; ``build_prompt_entries`` builds the
per-prompt dicts every AR-bearing HI3 modality shares.

``build_response`` produces the two-track ``RolloutResp``: ``"ar"``
(TextSegment root) + ``"image"`` (LatentSegment child, ``parent_track="ar"``,
``conditions["fused"]`` from the worker-side ``prepare_inputs_for_generation``
capture).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from unirl.rollout.engine.vllm_omni_v2.adapters.base import ModelAdapter, register_adapter
from unirl.rollout.engine.vllm_omni_v2.backends import (
    STAGE_KIND_AR,
    STAGE_KIND_DIFFUSION,
    GenerateCall,
    OmniRawResult,
    StageSampling,
)
from unirl.rollout.engine.vllm_omni_v2.utils import (
    assemble_tracks,
    build_ar_segment,
    build_fused_mm_condition,
    build_image_segment,
    collect_dit_outputs,
    decoded_text_from_ar,
    pil_images_from_req,
    pils_to_images,
    texts_from_req,
)
from unirl.types.primitives import Texts
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp
from unirl.types.sampling import get_ar_params, get_diffusion_params


def build_prompt_entries(
    texts: Texts,
    *,
    task: str,
    sys_type: str,
    modalities_field: List[str],
    tokenize_fn: Optional[Callable[..., List[int]]],
    decorate: Callable[[Dict[str, Any], int], None],
) -> List[Dict[str, Any]]:
    """Build the HI3 per-prompt dicts shared by the AR-bearing modalities.

    Each entry carries the official ``end2end.py`` base fields
    (``prompt_token_ids`` / ``prompt`` / ``use_system_prompt`` /
    ``modalities``); the adapter's ``decorate`` callback then attaches its
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


class Hi3ArDiTAdapter(ModelAdapter):
    """Shape base for the HI3 two-stage (AR + DiT) modalities."""

    omni_mode = "text-to-image"
    ar_lora_passthrough = True
    clear_cuda_visible = True
    #: it2i overrides: the request carries ``primitives['image']``.
    image_input = False

    # ---- HI3 chat-template row (upstream ``_TASK_PRESETS`` mirror) ----
    task_key = ""
    sys_type = "en_unified"
    output_modalities = ("image",)

    def resolve_task(self, stage_config: Dict[str, Any]) -> Tuple[str, str]:
        """Resolve ``(task_key, sys_type)`` with the ``stage_config`` overrides.

        ``stage_config["bot_task"]`` swaps the trigger tag used by upstream's
        chat template (``think`` / ``recaption``). ``stage_config["sys_type"]``
        overrides the system-prompt key (``en_unified`` / ``en_vanilla``).
        """
        sys_type = stage_config.get("sys_type") or self.sys_type
        bot_task = stage_config.get("bot_task")
        if bot_task in ("think", "recaption"):
            return f"{self.modality}_{bot_task}", sys_type
        return self.task_key, sys_type

    # ------------------------------------------------------------------ #
    # Request side
    # ------------------------------------------------------------------ #

    def build_inputs(self, req: RolloutReq) -> List[GenerateCall]:
        task, sys_type = self.resolve_task(req.stage_config or {})

        texts = texts_from_req(req)
        n = len(texts.texts)

        pil_images = pil_images_from_req(req, n) if self.image_input else []
        if self.image_input and not pil_images:
            raise ValueError(f"modality={self.modality!r} requires req.primitives['image']")
        if not self.image_input and req.primitives.get("image") is not None:
            raise ValueError(f"modality={self.modality!r} does not accept req.primitives['image']")

        diff_params = get_diffusion_params(req.sampling_params)
        ar_params = get_ar_params(req.sampling_params)

        prompts = build_prompt_entries(
            texts,
            task=task,
            sys_type=sys_type,
            modalities_field=self.output_modalities,
            tokenize_fn=self.tokenize_fn,
            decorate=lambda entry, i: self.decorate_prompt_entry(
                entry, i, pil_images=pil_images, diff_params=diff_params
            ),
        )

        return [
            GenerateCall(
                prompts=prompts,
                sampling=[
                    self.build_ar_sampling(ar_params),
                    self.build_dit_sampling(req, diff_params),
                ],
            )
        ]

    def decorate_prompt_entry(
        self, entry: Dict[str, Any], i: int, *, pil_images: List[Any], diff_params: Any
    ) -> None:
        """Modality-specific prompt-entry extras. Default: the t2i shape
        (request height/width on the entry); it2i overrides to attach the
        conditioning image + its own dimensions."""
        del i, pil_images
        entry["height"] = int(diff_params.height)
        entry["width"] = int(diff_params.width)

    def build_ar_sampling(self, ar_params: Any) -> StageSampling:
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

    def build_dit_sampling(self, req: RolloutReq, diff_params: Any) -> StageSampling:
        diff_kwargs = self.core_diff_kwargs(req, diff_params)
        seed = getattr(diff_params, "seed", None)
        if seed is not None:
            diff_kwargs["seed"] = int(seed)

        extra_args = self.sde_extra_args(diff_params)

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

    # ------------------------------------------------------------------ #
    # Response side
    # ------------------------------------------------------------------ #

    def build_response(self, req: RolloutReq, per_request: List[List[OmniRawResult]]) -> RolloutResp:
        if not per_request or not any(per_request):
            raise ValueError("build_response: empty per-request outputs (Omni.generate returned nothing surfaceable).")

        diff_outputs, _frames, pil_images = collect_dit_outputs(
            per_request, final_output_type="image", stage_id=1, modality=self.modality
        )
        decoded_image = pils_to_images(pil_images)
        segments = {"image": build_image_segment(diff_outputs, expected_sigmas=req.sigmas)}
        conditions = self.build_dit_condition(diff_outputs)

        # Surface the AR-generated text (best-effort; don't break rollout if
        # AR text extraction fails).
        try:
            decoded_text = decoded_text_from_ar(per_request)
        except Exception:
            decoded_text = None

        ar_segment = build_ar_segment(per_request)
        if ar_segment is not None:
            segments["ar"] = ar_segment

        return assemble_tracks(
            req,
            segments_for_track=segments,
            decoded_for_track={"image": decoded_image, "ar": decoded_text},
            conditions=conditions,
        )

    def build_dit_condition(self, diff_outputs: List[OmniRawResult]) -> Dict[str, Any]:
        """The DiT replay conditions — fused multimodal capture for HI3."""
        fused = build_fused_mm_condition(diff_outputs)
        if fused is None:
            raise RuntimeError(
                f"build_response: HI3 rollout (modality={self.modality!r}) "
                "returned no 'fused_mm_capture' on DiffusionOutput.custom_output. "
                "Check that RLHunyuanImage3Pipeline.prepare_inputs_for_generation "
                "hook ran in every DiT worker — the subclass swap may not have "
                "taken effect (verify custom_pipeline_args.pipeline_class in "
                "the stage YAML)."
            )
        return {"fused": fused}


@register_adapter("t2i")
class T2iAdapter(Hi3ArDiTAdapter):
    """HI3 text → AR think → DiT image."""

    stage_yaml = "hunyuan_image3_t2i_rl.yaml"
    task_key = "t2i_think"

    def resolve_task(self, stage_config: Dict[str, Any]) -> Tuple[str, str]:
        # ``vanilla`` swaps BOTH the task and the system prompt to the
        # no-think preset (upstream pairs t2i_vanilla with en_vanilla).
        if stage_config.get("bot_task") == "vanilla":
            return "t2i_vanilla", "en_vanilla"
        return super().resolve_task(stage_config)

    def validate_request(self, req: RolloutReq) -> None:
        if req.primitives.get("image") is not None:
            raise ValueError(
                "modality='t2i' rejects image-bearing requests; "
                "use an image-conditioned modality instead."
            )


@register_adapter("it2i")
class It2iAdapter(Hi3ArDiTAdapter):
    """HI3 image+text → AR recaption → DiT edited image."""

    stage_yaml = "hunyuan_image3_it2i_rl.yaml"
    task_key = "it2i_think"
    image_input = True

    def validate_request(self, req: RolloutReq) -> None:
        if req.primitives.get("image") is None:
            raise ValueError("modality='it2i' requires req.primitives['image'].")

    def decorate_prompt_entry(
        self, entry: Dict[str, Any], i: int, *, pil_images: List[Any], diff_params: Any
    ) -> None:
        # Upstream HI3 reads height/width off the prompt dict for the it2i
        # path (matches end2end.py:185-187) — the PIL dims, not the request's.
        del diff_params
        pil = pil_images[i]
        entry["multi_modal_data"] = {"image": pil}
        entry["height"] = pil.height
        entry["width"] = pil.width


__all__ = ["Hi3ArDiTAdapter", "It2iAdapter", "T2iAdapter", "build_prompt_entries"]
