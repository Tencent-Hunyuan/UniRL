"""BAGEL-7B-MoT family: input/output sub-adapters + the ``bagel_t2i`` modality class.

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

from typing import Any, Dict, List

from unirl.models.bagel.conditions import BagelDiffusionConditions
from unirl.rollout.engine.vllm_omni.adapters.base import ModelAdapter, register_adapter
from unirl.rollout.engine.vllm_omni.adapters.dit import DitInputAdapter, DitOutputAdapter
from unirl.rollout.engine.vllm_omni.backends import (
    STAGE_KIND_DIFFUSION,
    GenerateCall,
    OmniRawResult,
    StageSampling,
)
from unirl.rollout.engine.vllm_omni.utils import (
    build_image_segment,
    collect_dit_outputs,
    pils_to_images,
    texts_from_req,
)
from unirl.rollout.engine.vllm_omni.utils.noise import pack_initial_noise_extra_args
from unirl.rollout.engine.vllm_omni.utils.sigmas import sigmas_list_from_req
from unirl.sde.runtime import FlowMatchSchedulePolicy
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp
from unirl.types.sampling import get_diffusion_params


class BagelInputAdapter(DitInputAdapter):
    """Request side: prompt dicts + the BAGEL diffusion-stage sampling intent."""

    def build_prompts(self, req: RolloutReq) -> List[Any]:
        """Plain ``{"prompt": text}`` dicts (no ``modalities`` → image path).

        BAGEL's ``forward`` routes to text-only output only when
        ``modalities`` contains ``"text"``; an absent/empty ``modalities`` runs
        the text2img diffusion path we want. No ``negative_prompt`` key is added
        — the trainside oracle runs cfg=1 (the negative text branch is unused at
        cfg_text_scale=1.0), and the CFG scales ride ``extra_args`` instead.
        """
        if req.primitives.get("image") is not None:
            raise ValueError(f"modality={self.modality!r} does not accept req.primitives['image']")
        texts = texts_from_req(req)
        return [{"prompt": text} for text in texts.texts]

    def build_sampling(self, req: RolloutReq) -> List[StageSampling]:
        """One diffusion-stage intent with the BAGEL-specific kwargs.

        ``num_inference_steps`` is sent as ``T + 1`` (BAGEL loops
        ``num_timesteps - 1``); CFG knobs + SDE step set + trajectory precision
        ride ``extra_args``; the driver-authoritative x_T recipe is packed in.
        """
        texts = texts_from_req(req)
        diff_params = get_diffusion_params(req.sampling_params)

        T = int(diff_params.num_inference_steps)
        diff_kwargs: Dict[str, Any] = dict(
            height=int(diff_params.height),
            width=int(diff_params.width),
            # +1: BAGEL builds linspace(1, 0, num_timesteps) and loops T = num_timesteps-1.
            num_inference_steps=T + 1,
            eta=float(diff_params.eta),
            return_trajectory_latents=True,
            return_trajectory_decoded=False,
            num_outputs_per_prompt=1,
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

        pack_initial_noise_extra_args(extra_args, req, diff_params, n_prompts=len(texts.texts), caller=self.modality)
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
        diff_params = get_diffusion_params(req.sampling_params)
        image_shape = (int(diff_params.height), int(diff_params.width))
        prompts = list(texts.texts)
        conditions = BagelDiffusionConditions(
            prompts=prompts,
            image_shapes=[image_shape] * len(prompts),
        )
        return conditions.to_dict()


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


__all__ = ["BagelInputAdapter", "BagelOutputAdapter", "BagelT2iAdapter"]
