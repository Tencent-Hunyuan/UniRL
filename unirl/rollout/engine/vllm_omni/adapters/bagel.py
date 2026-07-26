"""BAGEL-7B-MoT family adapters for direct T2I and think-then-image T2TI.

Single diffusion stage (the BAGEL single-stage topology, where the DiT worker
owns its own LLM/ViT/VAE/tokenizer), TP=1, no AR prelude. BAGEL forces two
deviations from the shared DiT skeleton — everything else is reused:

- **σ off-by-one.** BAGEL's ``generate_image`` builds its σ schedule internally
  from ``num_timesteps`` and loops ``num_timesteps - 1`` steps (``linspace(1, 0,
  num_timesteps)`` then drop the terminal). To run the trainside ``T`` steps the
  worker must receive ``num_inference_steps = T + 1``. The engine pins
  ``req.sigmas`` for ``T`` steps (``T + 1`` σ points) via this adapter's
  static-shift :meth:`schedule_policy`; BAGEL's internal schedule then equals it
  (BAGEL hardwires ``timestep_shift = 3.0`` — the trainside shift — and the σ
  formula is identical), and the response-side ``verify_engine_used_sigmas``
  asserts the match. NB: BAGEL ignores ``sampling_params.sigmas`` (it does NOT
  call ``set_timesteps(sigmas=...)`` like SD3/Qwen), so the schedule is steered
  purely through ``num_inference_steps`` + the fixed shift.

- **CFG via ``extra_args``, NOT ``guidance_scale``.** Upstream ``forward`` reads
  ``cfg_text_scale`` / ``cfg_img_scale`` / ``cfg_interval`` / ``cfg_renorm_*`` off
  ``extra_args`` and **defaults them to 4.0 / 1.5 / (0.4,1.0) — CFG ON — when
  absent**. The trainside recipes run cfg=1 (single-forward), so the adapter
  ALWAYS sends the BagelDiffusionParams CFG knobs explicitly; a missing key would
  silently arm CFG@4.0 and diverge from the trainside oracle.

- **Conditions = PROMPTS, not embeds.** BAGEL conditioning is opaque KV-cache
  contexts built through the (frozen) und/text path — not a dense tensor, and not
  transportable across the worker→driver IPC boundary. So the output adapter
  ships the prompts (+ per-sample image shape) as a deferred
  :class:`~unirl.models.bagel.conditions.BagelDiffusionConditions`; the trainer
  rebuilds the KV contexts on its own bundle at replay time (the und path being
  frozen, the rebuilt contexts are identical regardless of the gen-LoRA state).
  This is the load-bearing difference from SD3 / Qwen-Image, which ship dense
  text embeds captured by an ``encode_prompt`` tap.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Tuple

import torch

from unirl.models.bagel.conditions import (
    BAGEL_T2TI_REPLAY_CUSTOM_OUTPUT,
    BagelARConditions,
    BagelDiffusionConditions,
    BagelT2TIDiffusionConditions,
    BagelThinkKVReplaySpec,
)
from unirl.models.bagel.diffusion import BagelDiffusionParams
from unirl.rollout.engine.sigma_verify import verify_engine_used_sigmas
from unirl.rollout.engine.vllm_omni.adapters.base import ModelAdapter, register_adapter
from unirl.rollout.engine.vllm_omni.adapters.dit import DitInputAdapter, DitOutputAdapter
from unirl.rollout.engine.vllm_omni.backends import (
    STAGE_KIND_AR,
    STAGE_KIND_DIFFUSION,
    GenerateCall,
    OmniRawResult,
    StageSampling,
)
from unirl.rollout.engine.vllm_omni.utils import (
    build_ar_segment,
    build_image_segment,
    collect_dit_outputs,
    decoded_text_from_ar,
    grouped_texts_from_req,
    pils_to_images,
    seed_from_sample_id,
    texts_from_req,
)
from unirl.rollout.engine.vllm_omni.utils.noise import pack_initial_noise_extra_args
from unirl.rollout.engine.vllm_omni.utils.sigmas import sigmas_list_from_req
from unirl.sde.runtime import FlowMatchSchedulePolicy
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp, RolloutTrack
from unirl.types.sampling import ARSamplingParams

# Keep this byte-identical to vllm-omni's bagel stage-input processor and the
# vendored trainside inferencer. The native prompt below deliberately contains
# no chat roles or extra newlines: BAGEL's prepare_prompts wraps each logical
# text split with these special tokens directly.
GEN_THINK_SYSTEM_PROMPT = (
    "You should first think about the planning process in the mind "
    "and then generate the image. \n"
    "The planning process is enclosed within <think> </think> tags, "
    "i.e. <think> planning process here </think> image here"
)


def _bagel_think_prompt(user_prompt: str) -> str:
    """Render trainside ``[BOS sys EOS][BOS user EOS]`` plus AR start BOS."""
    return f"<|im_start|>{GEN_THINK_SYSTEM_PROMPT}<|im_end|><|im_start|>{user_prompt}<|im_end|><|im_start|>"


def _t2ti_request_keys(
    req: RolloutReq,
    *,
    n_prompts: int,
    n_ar: int,
    share_initial_noise: bool,
) -> Tuple[List[str], List[str]]:
    """Resolve independent RNG identities and initial-noise group identities."""
    total = n_prompts * n_ar
    supplied = [str(x) for x in (req.init_noise_group_ids or [])]
    if len(supplied) == total:
        noise_keys = supplied
    elif len(supplied) == n_prompts:
        noise_keys = [root if share_initial_noise else f"{root}/a{j}/i0" for root in supplied for j in range(n_ar)]
    elif supplied:
        raise RuntimeError(
            "bagel_t2ti: init_noise_group_ids must contain either one id per "
            f"source prompt ({n_prompts}) or one per AR/image pair ({total}); got {len(supplied)}."
        )
    else:
        rollout_id = (req.task_config or {}).get("rollout_id")
        prefix = f"rollout:{rollout_id}:" if rollout_id is not None else "sample:"
        roots = [f"{prefix}{sample_id}" for sample_id in req.sample_ids]
        noise_keys = [root if share_initial_noise else f"{root}/a{j}/i0" for root in roots for j in range(n_ar)]

    rollout_id = (req.task_config or {}).get("rollout_id")
    prefix = f"rollout:{rollout_id}:" if rollout_id is not None else "sample:"
    rng_keys = [f"{prefix}{sample_id}/a{j}/i0" for sample_id in req.sample_ids for j in range(n_ar)]
    if len(rng_keys) != total or len(noise_keys) != total:
        raise RuntimeError(
            f"bagel_t2ti: resolved rng={len(rng_keys)} and noise={len(noise_keys)} keys for {total} pairs."
        )
    if len(set(rng_keys)) != total:
        raise ValueError("bagel_t2ti: sample_ids must be unique so every AR/image draw has distinct RNG state.")
    return rng_keys, noise_keys


def _allocate_seed(namespace: str, key: str, used: set[int]) -> int:
    """Allocate a stable 31-bit seed, deterministically resolving rare hashes."""
    nonce = 0
    while True:
        suffix = "" if nonce == 0 else f":collision:{nonce}"
        candidate = seed_from_sample_id(f"{namespace}:{key}{suffix}")
        if candidate not in used:
            used.add(candidate)
            return candidate
        nonce += 1


def _t2ti_params(req: RolloutReq) -> Tuple[ARSamplingParams, BagelDiffusionParams]:
    ar_params = req.sampling_params.get("ar")
    diff_params = req.sampling_params.get("diffusion")
    if not isinstance(ar_params, ARSamplingParams) or not isinstance(diff_params, BagelDiffusionParams):
        raise TypeError(
            "bagel_t2ti requires sampling_params['ar'] (ARSamplingParams) and "
            "sampling_params['diffusion'] (BagelDiffusionParams); got "
            f"ar={type(ar_params).__name__}, diffusion={type(diff_params).__name__}."
        )
    return ar_params, diff_params


class BagelInputAdapter(DitInputAdapter):
    """Request side: prompt dicts + the BAGEL diffusion-stage sampling intent."""

    def _spp(self, req: RolloutReq) -> int:
        """``samples_per_prompt`` — the GRPO group size; 1 disables packing."""
        return req.sampling_params.get("diffusion").samples_per_prompt

    def _is_packable_t2i(self, req: RolloutReq) -> bool:
        """Collapse spp samples into one ``num_outputs_per_prompt=spp`` request.

        Mirrors ``RLBagelPipeline._is_batchable_t2i``: packed ``generate_image``
        is cfg=1 t2i only. CFG>1 keeps the sample-level layout.
        """
        if self._spp(req) <= 1:
            return False
        diff_params = req.sampling_params.get("diffusion")
        return diff_params.cfg_text_scale <= 1.0 and diff_params.cfg_img_scale <= 1.0

    def build_prompts(self, req: RolloutReq) -> List[Any]:
        """Plain ``{"prompt": text}`` dicts (no ``modalities`` → image path).

        BAGEL's ``forward`` routes to text-only output only when
        ``modalities`` contains ``"text"``; an absent/empty ``modalities`` runs
        the text2img diffusion path we want. No ``negative_prompt`` key is added
        — the trainside oracle runs cfg=1 (the negative text branch is unused at
        cfg_text_scale=1.0), and the CFG scales ride ``extra_args`` instead.

        When packable, each prompt's spp samples collapse to ONE request
        (``num_outputs_per_prompt=spp``). Otherwise keep one request per sample.
        """
        if req.primitives.get("image") is not None:
            raise ValueError(f"modality={self.modality!r} does not accept req.primitives['image']")
        if not self._is_packable_t2i(req):
            texts = texts_from_req(req)
            return [{"prompt": text} for text in texts.texts]
        grouped_texts, _ = grouped_texts_from_req(
            req,
            samples_per_prompt=self._spp(req),
            caller=f"{self.modality}.build_prompts",
        )
        return [{"prompt": text} for text in grouped_texts]

    def build_sampling(self, req: RolloutReq) -> List[StageSampling]:
        """One diffusion-stage intent with the BAGEL-specific kwargs.

        ``num_inference_steps`` is sent as ``T + 1`` (BAGEL loops
        ``num_timesteps - 1``); CFG knobs + SDE step set + trajectory precision
        ride ``extra_args``; the driver-authoritative x_T recipe is packed in.
        """
        texts = texts_from_req(req)
        diff_params = req.sampling_params.get("diffusion")
        pack = self._is_packable_t2i(req)
        spp = self._spp(req)

        T = int(diff_params.num_inference_steps)
        diff_kwargs: Dict[str, Any] = dict(
            height=int(diff_params.height),
            width=int(diff_params.width),
            # +1: BAGEL builds linspace(1, 0, num_timesteps) and loops T = num_timesteps-1.
            num_inference_steps=T + 1,
            eta=float(diff_params.eta),
            return_trajectory_latents=True,
            return_trajectory_decoded=False,
            # Packable: one packed generate_image. Else: one image per request.
            num_outputs_per_prompt=spp if pack else 1,
        )
        seed = getattr(diff_params, "seed", None)
        if seed is not None:
            diff_kwargs["seed"] = int(seed)

        # σ contract self-check: req.sigmas (pinned by the engine for T steps) must
        # have T+1 points. We don't SEND sigmas (BAGEL ignores them), but assert the
        # engine resolved the schedule for the same T the worker will loop.
        _ = sigmas_list_from_req(req, T)

        # BAGEL CFG knobs — ALWAYS explicit (upstream defaults them to CFG-ON).
        extra_args: Dict[str, Any] = {
            "cfg_text_scale": float(getattr(diff_params, "cfg_text_scale", 1.0)),
            "cfg_img_scale": float(getattr(diff_params, "cfg_img_scale", 1.0)),
            "cfg_interval": tuple(getattr(diff_params, "cfg_interval", (0.0, 1.0))),
            "cfg_renorm_min": float(getattr(diff_params, "cfg_renorm_min", 0.0)),
            "cfg_renorm_type": str(getattr(diff_params, "cfg_renorm_type", "global")),
        }
        sde_indices = getattr(diff_params, "sde_indices", None)
        if sde_indices is not None:
            extra_args["sde_indices"] = sorted({int(i) for i in sde_indices})
        # σ_max for the SDE std_dev_t clamp. The trainside BagelDiffusionStage uses
        # ``schedule[1]`` (the second σ point) as sigma_max — the value that
        # replaces σ==1 in ``sqrt(σ/(1-σ))`` on the FIRST step (σ_0 == 1.0, which
        # would divide by zero). The worker MUST use the SAME value or the first
        # SDE step's std_dev_t / log-prob diverges and the GRPO ratio drifts off 1
        # (observed ratio ≈ 0.8 with the hardcoded 0.99 default). req.sigmas is the
        # engine-pinned T+1-point schedule, identical to the trainside schedule.
        if req.sigmas is not None and int(req.sigmas.shape[0]) > 1:
            extra_args["sigma_max"] = float(req.sigmas[1].item())
        # Tell the worker scheduler the trajectory storage dtype so its SDE
        # log-prob round-trip matches the trainside trajectory_precision.
        traj_prec = getattr(diff_params, "trajectory_precision", None)
        if traj_prec is not None:
            extra_args["trajectory_precision"] = str(traj_prec)

        pack_initial_noise_extra_args(extra_args, req, diff_params, n_samples=len(texts.texts), caller=self.modality)
        diff_kwargs["extra_args"] = extra_args

        return [StageSampling(kind=STAGE_KIND_DIFFUSION, kwargs=diff_kwargs)]


class BagelOutputAdapter(DitOutputAdapter):
    """Response side: one ``"image"`` track with prompt-carrying conditions."""

    def build_segments(self, req: RolloutReq, per_request: List[List[OmniRawResult]]) -> Dict[str, Any]:
        """The DiT trajectory segment (asserts the σ echo). No AR sweep (BAGEL
        single-stage has no Stage-0 completions)."""
        diff_outputs, _, _ = collect_dit_outputs(
            per_request, final_output_type=self.final_output_type, stage_id=self.stage_id, modality=self.modality
        )
        return {self.track_name: build_image_segment(diff_outputs, expected_sigmas=req.sigmas)}

    def build_decoded(self, req: RolloutReq, per_request: List[List[OmniRawResult]]) -> Dict[str, Any]:
        del req
        _, _, pil_images = collect_dit_outputs(
            per_request, final_output_type=self.final_output_type, stage_id=self.stage_id, modality=self.modality
        )
        return {self.track_name: pils_to_images(pil_images)}

    def build_conditions(self, req: RolloutReq, per_request: List[List[OmniRawResult]]) -> Dict[str, Any]:
        """Ship the PROMPTS (deferred conditions) for trainer-side KV rebuild.

        BAGEL KV contexts can't cross the IPC boundary, so instead of capturing
        embeds we carry the prompt text + per-sample image shape. The trainer's
        :class:`BagelDiffusionStage` rebuilds the three KV contexts on its own
        bundle at replay (the und/text path is frozen → identical contexts).
        """
        del per_request
        texts = texts_from_req(req)
        diff_params = req.sampling_params.get("diffusion")
        image_shape = (int(diff_params.height), int(diff_params.width))
        prompts = list(texts.texts)
        conditions = BagelDiffusionConditions(
            prompts=prompts,
            image_shapes=[image_shape] * len(prompts),
        )
        return conditions.to_dict()


class BagelT2TIInputAdapter:
    """Request side for native BAGEL thinking followed by image diffusion.

    vLLM-Omni shares one sampling object per stage across a generate call, so
    each AR chain is its own one-prompt call. This is required for distinct AR
    and image seeds and for selecting exactly one driver-authored x_T recipe.
    """

    def __init__(self, modality: str) -> None:
        self.modality = modality

    def build(self, req: RolloutReq) -> List[GenerateCall]:
        ar_params, diff_params = _t2ti_params(req)
        texts = texts_from_req(req)
        n_prompts = len(texts.texts)
        n_ar = int(ar_params.samples_per_prompt)
        rollout_keys, noise_keys = _t2ti_request_keys(
            req,
            n_prompts=n_prompts,
            n_ar=n_ar,
            share_initial_noise=bool(diff_params.init_same_noise),
        )

        T = int(diff_params.num_inference_steps)
        _ = sigmas_list_from_req(req, T)
        image_shape = (int(diff_params.height), int(diff_params.width))

        initial_latent_cond = (req.request_conditions or {}).get("initial_latents")
        initial_latents = getattr(initial_latent_cond, "latents", None)
        expected = n_prompts * n_ar
        if initial_latents is not None and int(initial_latents.shape[0]) != expected:
            raise RuntimeError(
                "bagel_t2ti: initial_latents must have one row per AR/image pair; "
                f"got {int(initial_latents.shape[0])}, expected {expected}."
            )
        if req.init_noise_group_ids and not req.init_noise_latent_shape and initial_latents is None:
            raise ValueError("bagel_t2ti: init_noise_group_ids require init_noise_latent_shape.")

        used_seeds: set[int] = set()
        calls: List[GenerateCall] = []
        flat_idx = 0
        ar_base_seed = int(ar_params.seed) if ar_params.seed is not None else 0
        image_base_seed = int(diff_params.seed) if diff_params.seed is not None else 0
        for user_prompt in texts.texts:
            for _ in range(n_ar):
                rollout_key = rollout_keys[flat_idx]
                ar_seed = _allocate_seed(f"bagel-t2ti-ar:{ar_base_seed}", rollout_key, used_seeds)
                image_seed = _allocate_seed(f"bagel-t2ti-image:{image_base_seed}", rollout_key, used_seeds)
                sde_seed = _allocate_seed(f"bagel-t2ti-sde:{image_base_seed}", rollout_key, used_seeds)

                prompt: Dict[str, Any] = {
                    "prompt": _bagel_think_prompt(user_prompt),
                    "modalities": ["image"],
                    "mm_processor_kwargs": {
                        "target_h": image_shape[0],
                        "target_w": image_shape[1],
                        "modalities": ["image"],
                    },
                }

                ar_kwargs: Dict[str, Any] = {
                    "n": 1,
                    "temperature": float(ar_params.temperature),
                    "top_p": float(ar_params.top_p),
                    "top_k": int(ar_params.top_k),
                    "max_tokens": int(ar_params.max_new_tokens),
                    "seed": ar_seed,
                    "logprobs": 1,
                    "detokenize": True,
                }
                if ar_params.stop_token_id is not None:
                    ar_kwargs["stop_token_ids"] = [int(ar_params.stop_token_id)]

                diff_kwargs: Dict[str, Any] = {
                    "height": image_shape[0],
                    "width": image_shape[1],
                    # BAGEL loops num_timesteps - 1 denoising transitions.
                    "num_inference_steps": T + 1,
                    "eta": float(diff_params.eta),
                    "seed": image_seed,
                    "return_trajectory_latents": True,
                    "return_trajectory_decoded": False,
                    "num_outputs_per_prompt": 1,
                }
                extra_args: Dict[str, Any] = {
                    # Explicit because upstream BAGEL defaults missing CFG to on.
                    "cfg_text_scale": float(diff_params.cfg_text_scale),
                    "cfg_img_scale": float(diff_params.cfg_img_scale),
                    "cfg_interval": tuple(diff_params.cfg_interval),
                    "cfg_renorm_min": float(diff_params.cfg_renorm_min),
                    "cfg_renorm_type": str(diff_params.cfg_renorm_type),
                    # Keep SDE exploration deterministic without sharing it
                    # across N branches that intentionally share only x_T.
                    "sde_seed": sde_seed,
                }
                if diff_params.sde_indices is not None:
                    extra_args["sde_indices"] = sorted({int(i) for i in diff_params.sde_indices})
                if req.sigmas is not None and int(req.sigmas.shape[0]) > 1:
                    extra_args["sigma_max"] = float(req.sigmas[1].item())
                if diff_params.trajectory_precision is not None:
                    extra_args["trajectory_precision"] = str(diff_params.trajectory_precision)

                if initial_latents is not None:
                    extra_args["initial_noise_batch"] = initial_latents[flat_idx : flat_idx + 1]
                elif req.init_noise_latent_shape:
                    extra_args["init_noise_group_ids"] = [noise_keys[flat_idx]]
                    extra_args["init_noise_latent_shape"] = [int(x) for x in req.init_noise_latent_shape]
                    extra_args["init_noise_seed"] = int(diff_params.seed) if diff_params.seed is not None else 0
                diff_kwargs["extra_args"] = extra_args

                calls.append(
                    GenerateCall(
                        prompts=[prompt],
                        sampling=[
                            StageSampling(kind=STAGE_KIND_AR, kwargs=ar_kwargs),
                            StageSampling(kind=STAGE_KIND_DIFFUSION, kwargs=diff_kwargs),
                        ],
                        group_by_request_id=False,
                    )
                )
                flat_idx += 1
        return calls


def _require_t2ti_stage_pair(
    outputs: Sequence[OmniRawResult], *, request_index: int
) -> Tuple[OmniRawResult, OmniRawResult]:
    ar_outputs = [
        out
        for out in outputs
        if getattr(out, "stage_id", None) == 0 and getattr(out, "final_output_type", None) == "text"
    ]
    image_outputs = [
        out
        for out in outputs
        if getattr(out, "stage_id", None) == 1 and getattr(out, "final_output_type", None) == "image"
    ]
    if len(ar_outputs) != 1 or len(image_outputs) != 1:
        raise RuntimeError(
            "bagel_t2ti: every call must surface exactly one Stage-0 text and one Stage-1 image output; "
            f"request {request_index} returned {len(ar_outputs)} text and {len(image_outputs)} image outputs."
        )
    completions = getattr(getattr(ar_outputs[0], "request_output", None), "outputs", None) or []
    if len(completions) != 1:
        raise RuntimeError(
            f"bagel_t2ti: Stage-0 request {request_index} returned {len(completions)} completions; expected one."
        )
    images = getattr(image_outputs[0], "images", None) or []
    if len(images) != 1:
        raise RuntimeError(f"bagel_t2ti: Stage-1 request {request_index} returned {len(images)} images; expected one.")
    return ar_outputs[0], image_outputs[0]


def _ar_replay_prompt_ids(
    ar_output: OmniRawResult,
    spec: BagelThinkKVReplaySpec,
    *,
    request_index: int,
    excluded_tail_input_ids: Sequence[int] = (),
) -> List[int]:
    """Validate native cache provenance and remove the single AR start BOS."""
    raw_prompt_ids = getattr(ar_output, "prompt_token_ids", None)
    if raw_prompt_ids is None:
        raise RuntimeError(f"bagel_t2ti: Stage-0 request {request_index} omitted prompt_token_ids.")
    prompt_ids = [int(x) for x in raw_prompt_ids]
    if len(prompt_ids) < 2:
        raise RuntimeError(
            f"bagel_t2ti: Stage-0 request {request_index} prompt is too short to contain prompt content plus AR start."
        )
    if prompt_ids[-1] != prompt_ids[0]:
        raise RuntimeError(
            f"bagel_t2ti: Stage-0 request {request_index} does not end with the same "
            "<|im_start|> token that begins the exact rendered prompt."
        )

    trace = tuple(int(x) for x in spec.cache_input_ids)
    prompt_tuple = tuple(prompt_ids)
    if len(trace) < len(prompt_tuple) or trace[: len(prompt_tuple)] != prompt_tuple:
        raise RuntimeError(
            f"bagel_t2ti: Stage-0 request {request_index} prompt_token_ids are not the prefix of the "
            "Stage-1 native KV replay trace."
        )

    completion = getattr(ar_output.request_output, "outputs")[0]
    sampled_ids = tuple(int(x) for x in (getattr(completion, "token_ids", None) or []))
    traced_completion = trace[len(prompt_tuple) :]
    expected_completion = sampled_ids[:-1]
    if traced_completion != expected_completion:
        raise RuntimeError(
            f"bagel_t2ti: Stage-0 request {request_index} transferred completion tokens "
            "must equal sampled token_ids[:-1] at the final computed-token boundary."
        )
    excluded_tail = tuple(int(token) for token in excluded_tail_input_ids)
    if excluded_tail and excluded_tail != sampled_ids[-1:]:
        raise RuntimeError(
            f"bagel_t2ti: Stage-0 request {request_index} excluded async tail does not equal "
            "the final emitted token_id."
        )

    # The exact prompt renderer appends one <|im_start|>. BagelARStage.replay
    # supplies that start token itself, so retaining it here would score a
    # duplicated BOS. The image replay spec intentionally keeps the full trace.
    return prompt_ids[:-1]


def _validate_t2ti_ar_logprobs(ar_output: OmniRawResult, *, request_index: int) -> None:
    """Require one finite raw log-prob for every emitted AR token."""
    completion = getattr(ar_output.request_output, "outputs")[0]
    sampled_ids = [int(x) for x in (getattr(completion, "token_ids", None) or [])]
    raw_logprobs = getattr(completion, "logprobs", None)
    if not sampled_ids:
        raise RuntimeError(f"bagel_t2ti: Stage-0 request {request_index} emitted no AR tokens.")
    if not isinstance(raw_logprobs, Sequence) or isinstance(raw_logprobs, (str, bytes)):
        raise RuntimeError(f"bagel_t2ti: Stage-0 request {request_index} omitted per-token AR logprobs.")
    if len(raw_logprobs) != len(sampled_ids):
        raise RuntimeError(
            f"bagel_t2ti: Stage-0 request {request_index} returned {len(raw_logprobs)} logprob steps "
            f"for {len(sampled_ids)} emitted tokens."
        )
    for step_idx, (token_id, step) in enumerate(zip(sampled_ids, raw_logprobs)):
        if isinstance(step, Mapping):
            entry = step.get(token_id, step.get(str(token_id)))
        else:
            entry = step if hasattr(step, "logprob") else None
        if entry is None:
            raise RuntimeError(
                f"bagel_t2ti: Stage-0 request {request_index} logprobs omitted emitted token "
                f"{token_id} at step {step_idx}."
            )
        value = float(getattr(entry, "logprob", entry))
        if not math.isfinite(value):
            raise RuntimeError(
                f"bagel_t2ti: Stage-0 request {request_index} has non-finite AR logprob at step {step_idx}: {value}."
            )


def _validate_t2ti_image_trajectories(
    image_outputs: Sequence[OmniRawResult],
    *,
    expected_sigmas: torch.Tensor | None,
    expected_sde_indices: Sequence[int],
) -> None:
    """Require one aligned Stage-1 trajectory and image for every thought."""
    reference: tuple[torch.Size, torch.Size, torch.Tensor, tuple[int, ...]] | None = None
    expected_sde = tuple(int(index) for index in expected_sde_indices)
    for request_index, output in enumerate(image_outputs):
        images = getattr(output, "images", None) or []
        if len(images) != 1:
            raise RuntimeError(
                f"bagel_t2ti: Stage-1 request {request_index} returned {len(images)} images; expected exactly one."
            )

        latents = getattr(output, "trajectory_latents", None)
        log_probs = getattr(output, "trajectory_log_probs", None)
        sigmas = getattr(output, "trajectory_timesteps", None)
        if not torch.is_tensor(latents) or latents.ndim < 3 or int(latents.shape[0]) != 1:
            shape = None if not torch.is_tensor(latents) else tuple(latents.shape)
            raise RuntimeError(
                f"bagel_t2ti: Stage-1 request {request_index} must return trajectory_latents "
                f"with batch size 1; got {shape}."
            )
        if not torch.is_tensor(log_probs) or log_probs.ndim != 2 or int(log_probs.shape[0]) != 1:
            shape = None if not torch.is_tensor(log_probs) else tuple(log_probs.shape)
            raise RuntimeError(
                f"bagel_t2ti: Stage-1 request {request_index} must return trajectory_log_probs "
                f"with shape [1, K], including [1, 0] for an ODE-only rollout; got {shape}."
            )
        if not torch.is_tensor(sigmas) or sigmas.ndim != 1:
            shape = None if not torch.is_tensor(sigmas) else tuple(sigmas.shape)
            raise RuntimeError(
                f"bagel_t2ti: Stage-1 request {request_index} must return a 1-D trajectory_timesteps; got {shape}."
            )
        if int(latents.shape[1]) != int(sigmas.numel()):
            raise RuntimeError(
                f"bagel_t2ti: Stage-1 request {request_index} returned {int(latents.shape[1])} latent positions "
                f"for {int(sigmas.numel())} sigma positions."
            )
        if not torch.isfinite(latents).all() or not torch.isfinite(log_probs).all() or not torch.isfinite(sigmas).all():
            raise RuntimeError(f"bagel_t2ti: Stage-1 request {request_index} returned a non-finite trajectory value.")

        verify_engine_used_sigmas(
            sigmas, expected=expected_sigmas, engine_name=f"vllm-omni bagel_t2ti[{request_index}]"
        )
        custom_output = getattr(output, "custom_output", None)
        if not isinstance(custom_output, Mapping) or "sde_step_indices" not in custom_output:
            raise RuntimeError(
                f"bagel_t2ti: Stage-1 request {request_index} omitted custom_output['sde_step_indices']."
            )
        sde_indices = tuple(int(index) for index in custom_output["sde_step_indices"])
        if sde_indices != expected_sde:
            raise RuntimeError(
                f"bagel_t2ti: Stage-1 request {request_index} used SDE indices {sde_indices}; expected {expected_sde}."
            )
        if int(log_probs.shape[1]) != len(sde_indices):
            raise RuntimeError(
                f"bagel_t2ti: Stage-1 request {request_index} returned {int(log_probs.shape[1])} log-probs "
                f"for {len(sde_indices)} SDE indices."
            )

        current = (latents.shape, log_probs.shape, sigmas.detach().cpu(), sde_indices)
        if reference is None:
            reference = current
            continue
        ref_latent_shape, ref_logp_shape, ref_sigmas, ref_sde_indices = reference
        if current[0] != ref_latent_shape or current[1] != ref_logp_shape:
            raise RuntimeError(
                f"bagel_t2ti: Stage-1 request {request_index} trajectory shapes "
                f"{tuple(current[0])}/{tuple(current[1])} differ from request 0 "
                f"{tuple(ref_latent_shape)}/{tuple(ref_logp_shape)}."
            )
        if current[3] != ref_sde_indices or not torch.equal(current[2], ref_sigmas):
            raise RuntimeError(
                f"bagel_t2ti: Stage-1 request {request_index} trajectory schedule differs from request 0."
            )


class BagelT2TIOutputAdapter:
    """Response side: independent AR and image tracks joined 1:1 by lineage."""

    def __init__(self, modality: str) -> None:
        self.modality = modality

    def build(self, req: RolloutReq, per_request: List[List[OmniRawResult]]) -> RolloutResp:
        ar_params, diff_params = _t2ti_params(req)
        expected = len(req.sample_ids) * int(ar_params.samples_per_prompt)
        if len(per_request) != expected:
            raise RuntimeError(f"bagel_t2ti: Omni returned {len(per_request)} request groups; expected P*N={expected}.")

        stage_pairs = [_require_t2ti_stage_pair(outputs, request_index=i) for i, outputs in enumerate(per_request)]
        image_outputs = [pair[1] for pair in stage_pairs]
        rollout_id = int((req.task_config or {}).get("rollout_id", 0))
        _validate_t2ti_image_trajectories(
            image_outputs,
            expected_sigmas=req.sigmas,
            expected_sde_indices=diff_params.resolve_sde_indices(rollout_id),
        )

        image_shape = (int(diff_params.height), int(diff_params.width))
        replay_specs: List[BagelThinkKVReplaySpec] = []
        prompt_splits: List[List[Dict[str, Any]]] = []
        for i, (ar_output, image_output) in enumerate(stage_pairs):
            _validate_t2ti_ar_logprobs(ar_output, request_index=i)
            custom_output = getattr(image_output, "custom_output", None)
            if not isinstance(custom_output, Mapping) or BAGEL_T2TI_REPLAY_CUSTOM_OUTPUT not in custom_output:
                raise RuntimeError(
                    "bagel_t2ti: Stage-1 output is missing required native KV replay metadata "
                    f"custom_output[{BAGEL_T2TI_REPLAY_CUSTOM_OUTPUT!r}] for request {i}."
                )
            spec = BagelThinkKVReplaySpec.from_custom_output(custom_output)
            replay_payload = custom_output[BAGEL_T2TI_REPLAY_CUSTOM_OUTPUT]
            excluded_tail_input_ids = replay_payload.get("excluded_tail_input_ids", ())
            if spec.image_shape != image_shape:
                raise RuntimeError(
                    f"bagel_t2ti: replay image_shape={spec.image_shape} for request {i} does not match "
                    f"the requested shape {image_shape}."
                )
            replay_specs.append(spec)
            prompt_ids = _ar_replay_prompt_ids(
                ar_output,
                spec,
                request_index=i,
                excluded_tail_input_ids=excluded_tail_input_ids,
            )
            prompt_splits.append([{"kind": "text", "ids": torch.tensor(prompt_ids, dtype=torch.long)}])

        ar_segment = build_ar_segment(per_request)
        if ar_segment is None:
            raise RuntimeError("bagel_t2ti: Stage 0 returned no sampled AR tokens.")
        image_segment = build_image_segment(image_outputs, expected_sigmas=req.sigmas)
        ar_decoded = decoded_text_from_ar(per_request)
        _, _, pil_images = collect_dit_outputs(
            per_request,
            final_output_type="image",
            stage_id=1,
            modality=self.modality,
        )
        image_decoded = pils_to_images(pil_images)
        if image_segment.latents is None or int(image_segment.latents.shape[0]) != expected:
            actual = None if image_segment.latents is None else int(image_segment.latents.shape[0])
            raise RuntimeError(f"bagel_t2ti: image trajectory batch has {actual} rows; expected P*N={expected}.")
        if int(image_decoded.pixels.shape[0]) != expected:
            raise RuntimeError(
                f"bagel_t2ti: decoded image batch has {int(image_decoded.pixels.shape[0])} rows; "
                f"expected P*N={expected}."
            )

        ar_conditions = BagelARConditions(prompt_splits=prompt_splits)
        image_conditions = BagelT2TIDiffusionConditions(replay_specs=replay_specs)

        ar_shell = req.make_root_track(track_name="ar", branch=int(ar_params.samples_per_prompt))
        ar_track = RolloutTrack(
            sample_ids=list(ar_shell.sample_ids),
            parent_ids=list(ar_shell.parent_ids) if ar_shell.parent_ids is not None else None,
            parent_track=ar_shell.parent_track,
            conditions=ar_conditions.to_dict(),
            segment=ar_segment,
            decoded=ar_decoded,
        )
        image_shell = ar_track.fork_track(parent_name="ar", child_name="image", branch=1)
        image_track = RolloutTrack(
            sample_ids=list(image_shell.sample_ids),
            parent_ids=list(image_shell.parent_ids) if image_shell.parent_ids is not None else None,
            parent_track=image_shell.parent_track,
            conditions=image_conditions.to_dict(),
            segment=image_segment,
            decoded=image_decoded,
        )
        return RolloutResp(tracks={"ar": ar_track, "image": image_track})


@register_adapter("bagel_t2i")
class BagelT2iAdapter(ModelAdapter):
    """BAGEL-7B-MoT text → image (single diffusion stage, TP=1)."""

    stage_yaml = "bagel_t2i_rl.yaml"
    omni_mode = "text-to-image"
    # The BAGEL single-stage DiT worker owns its tokenizer; the driver loads none.
    needs_driver_tokenizer = False

    def __init__(self, config: Any, model_config: Any, *, strategy: Any = None, tokenize_fn: Any = None) -> None:
        super().__init__(config, model_config, strategy=strategy, tokenize_fn=tokenize_fn)
        self.input_adapter = BagelInputAdapter(self.modality)
        self.output_adapter = BagelOutputAdapter(self.modality)

    def schedule_policy(self) -> FlowMatchSchedulePolicy:
        """Static-shift FlowMatch σ policy (BAGEL uses no dynamic shifting).

        Mirrors the trainside ``BagelPipeline.build_schedule_policy`` — a plain
        static-shift policy from ``model_config.shift`` — rather than the base
        ``from_pretrained`` path (the BAGEL checkpoint ships no
        ``scheduler_config.json``). The shift MUST equal BAGEL's hardwired
        ``timestep_shift`` (3.0) for the worker schedule to echo back equal.
        """
        shift = float(getattr(self.model_config, "shift", 3.0))
        return FlowMatchSchedulePolicy.static_only(shift)

    def validate_request(self, req: RolloutReq) -> None:
        if req.primitives.get("image") is not None:
            raise ValueError(
                f"modality={self.modality!r} rejects image-bearing requests; use an image-conditioned modality instead."
            )

    def build_inputs(self, req: RolloutReq) -> List[GenerateCall]:
        return self.input_adapter.build(req)

    def build_response(self, req: RolloutReq, per_request: List[List[OmniRawResult]]) -> RolloutResp:
        return self.output_adapter.build(req, per_request)


@register_adapter("bagel_t2ti")
class BagelT2TIAdapter(ModelAdapter):
    """BAGEL native planning text -> image with strict UniGRPO M=1."""

    stage_yaml = "bagel_t2ti_rl.yaml"
    omni_mode = "text-to-image"
    needs_driver_tokenizer = False
    # This initial scope is full-weight-only; do not expose a dormant LoRA path.
    ar_lora_passthrough = False

    def __init__(self, config: Any, model_config: Any, *, strategy: Any = None, tokenize_fn: Any = None) -> None:
        super().__init__(config, model_config, strategy=strategy, tokenize_fn=tokenize_fn)
        self.input_adapter = BagelT2TIInputAdapter(self.modality)
        self.output_adapter = BagelT2TIOutputAdapter(self.modality)

    def validate(self) -> None:
        super().validate()
        if bool(getattr(self.model_config, "use_lora", False)):
            raise ValueError(
                "bagel_t2ti currently supports full-weight training only; model_config.use_lora must be false."
            )

    def schedule_policy(self) -> FlowMatchSchedulePolicy:
        return FlowMatchSchedulePolicy.static_only(float(getattr(self.model_config, "shift", 3.0)))

    def validate_request(self, req: RolloutReq) -> None:
        if req.primitives.get("image") is not None:
            raise ValueError("bagel_t2ti accepts text prompts only; image-conditioned thinking is not supported.")
        ar_params, diff_params = _t2ti_params(req)
        n_ar = int(ar_params.samples_per_prompt)
        n_images = int(diff_params.samples_per_prompt)
        if n_ar < 2:
            raise ValueError(
                f"bagel_t2ti training requires ar.samples_per_prompt >= 2 so prompt-group advantages are nonzero; "
                f"got {n_ar}."
            )
        if n_images != 1:
            raise ValueError(
                f"bagel_t2ti currently requires diffusion.samples_per_prompt == 1 (UniGRPO strict M=1); got {n_images}."
            )
        if float(ar_params.temperature) != 1.0:
            raise ValueError(
                "bagel_t2ti requires ar.temperature == 1.0 because vLLM emits raw, "
                "untempered log-probs while trainer replay scores the sampled-token "
                f"distribution; got {ar_params.temperature}."
            )
        cfg_values = {
            "guidance_scale": float(diff_params.guidance_scale),
            "cfg_text_scale": float(diff_params.cfg_text_scale),
            "cfg_img_scale": float(diff_params.cfg_img_scale),
        }
        invalid_cfg = {name: value for name, value in cfg_values.items() if value != 1.0}
        if invalid_cfg:
            raise ValueError(
                f"bagel_t2ti requires all CFG scales == 1 for single-forward rollout/replay parity; got {invalid_cfg}."
            )

    def build_inputs(self, req: RolloutReq) -> List[GenerateCall]:
        return self.input_adapter.build(req)

    def build_response(self, req: RolloutReq, per_request: List[List[OmniRawResult]]) -> RolloutResp:
        return self.output_adapter.build(req, per_request)


__all__ = [
    "GEN_THINK_SYSTEM_PROMPT",
    "BagelInputAdapter",
    "BagelOutputAdapter",
    "BagelT2iAdapter",
    "BagelT2TIAdapter",
    "BagelT2TIInputAdapter",
    "BagelT2TIOutputAdapter",
]
