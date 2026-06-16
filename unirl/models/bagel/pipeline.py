"""BagelPipeline — RolloutReq → RolloutResp end-to-end for BAGEL-7B-MoT (T2I).

Four-tier flow, per-sample (navit ``bs=1``)::

    Texts ─build 3 KV contexts─▶ BagelDiffusionConditions ─diffuse─▶ LatentSegment ─vae_decode─▶ Images

Per prompt the pipeline builds the three KV-cache contexts the sampler needs
(mirroring ``InterleaveInferencer.interleave_inference`` for plain T2I, ``think=False``,
no input image):

- ``gen``      = init + text(prompt)          (conditional)
- ``cfg_text`` = init snapshot before the text (unconditional / text-CFG)
- ``cfg_img``  = init + text(prompt)          (== gen for pure T2I)

then runs ``diffusion.diffuse`` once per sample, accumulates the per-sample latents
into one batched ``LatentSegment``, decodes them, and packs one ``"image"`` track.

Central-runtime contract (same as SD3 — NOT a flow_grpo port):

- **σ schedule**: the hosting engine pins ``req.sigmas`` via
  :func:`unirl.sde.runtime.ensure_req_sigmas` (built from :meth:`build_schedule_policy`)
  BEFORE ``generate``; this pipeline reads it verbatim and passes it to ``diffuse``.
- **initial noise x_T**: driver-authored via :class:`NoiseRecipe` (per-sample,
  ``r{rollout_id}:{sample_id}``-keyed, byte-identical across engines). The pipeline
  resolves it from the request and hands each sample its slice; :meth:`latent_shape`
  declares the packed ``(seq, C)`` geometry so the driver can author the recipe.
- **SDE steps**: ``params.sde_indices`` (driver-resolved via
  ``resolve_sde_indices`` → ``AllSDEScheduler``); shared per rollout across the group.

``BagelBundle`` is imported lazily (it pulls the vendored modeling + flash_attn);
this keeps ``BagelPipeline`` importable on CPU for fake-stage tests.
"""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import torch

from unirl.models.types.pipeline import Pipeline
from unirl.sde.kernels import FlowSDEStrategy, StepStrategy
from unirl.sde.runtime import FlowMatchSchedulePolicy
from unirl.types.noise_recipe import NoiseRecipe
from unirl.types.primitives import Texts
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp, RolloutTrack
from unirl.types.sampling import get_diffusion_params
from unirl.types.segments.latent import LatentSegment

from .conditions import BagelDiffusionConditions
from .diffusion import BagelDiffusionParams, BagelDiffusionStage
from .vae import BagelVAEDecodeStage, bagel_latent_shape

if TYPE_CHECKING:
    from .bundle import BagelBundle


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    """Read ``key`` from a DictConfig / dict / dataclass, falling back to ``default``."""
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        try:
            val = cfg.get(key, default)
            return default if val is None else val
        except Exception:
            return default
    return getattr(cfg, key, default)


class BagelPipeline(Pipeline):
    """BAGEL-7B-MoT T2I generate pipeline (trainside A1)."""

    def __init__(
        self,
        *,
        bundle: "BagelBundle",
        diffusion: Optional[BagelDiffusionStage] = None,
        vae_decode: Optional[BagelVAEDecodeStage] = None,
        strategy: Optional[StepStrategy] = None,
        autocast_precision: str = "bf16",
        trajectory_precision: str = "fp32",
        logprob_precision: str = "fp32",
        shift: float = 3.0,
    ) -> None:
        super().__init__()
        self.bundle = bundle
        if diffusion is None:
            diffusion = BagelDiffusionStage(
                model=bundle,
                strategy=strategy if strategy is not None else FlowSDEStrategy(),
                autocast_precision=autocast_precision,
                trajectory_precision=trajectory_precision,
                logprob_precision=logprob_precision,
            )
        self.diffusion = diffusion
        self.vae_decode = vae_decode if vae_decode is not None else BagelVAEDecodeStage(bundle)
        self.autocast_precision = autocast_precision
        # FlowMatch time-shift for the σ schedule policy (read by the hosting engine
        # via build_schedule_policy → ensure_req_sigmas). Bagel uses static shift.
        self.shift = shift

    @classmethod
    def latent_shape(cls, *, model_config: Any, sampling_spec: Any) -> Tuple[int, ...]:
        """Packed per-sample x_T shape ``(seq, p²·z)`` for the driver NoiseRecipe.

        Bagel's x_T is packed navit ``[h·w, p²·z]`` (the ``packed_init_noises`` shape),
        NOT spatial ``[C, H, W]``; ``seq = (H // (vae_downsample·patch))²`` for a square
        image. Returning a concrete shape (rather than raising) opts Bagel into the
        driver-authored, cross-engine-reproducible x_T recipe (same as SD3).
        """
        cfg = _cfg_get(model_config, "config", model_config)
        patch = int(_cfg_get(cfg, "latent_patch_size", 2))
        vae_ds = int(_cfg_get(cfg, "vae_downsample", 8))
        z = int(_cfg_get(cfg, "latent_channels", 16))
        H, W = int(sampling_spec.height), int(sampling_spec.width)
        return bagel_latent_shape((H, W), latent_downsample=vae_ds * patch, latent_patch_size=patch, latent_channels=z)

    def build_schedule_policy(self) -> FlowMatchSchedulePolicy:
        """Static-shift FlowMatch σ policy (BAGEL uses no dynamic shifting).

        The hosting engine calls this once and pins ``req.sigmas`` via
        ``ensure_req_sigmas``; ``get_sigma_schedule(num_inference_steps, shift)`` then
        produces the byte-identical schedule the (former) ``bagel_timesteps`` did.
        """
        return FlowMatchSchedulePolicy.static_only(float(self.shift))

    @classmethod
    def from_config(cls, config: Any, *, strategy: Optional[StepStrategy] = None) -> "BagelPipeline":
        """Build the full pipeline from a :class:`BagelPipelineConfig`."""
        from .bundle import BagelBundle

        bundle = BagelBundle.from_config(config)
        return cls(
            bundle=bundle,
            strategy=strategy,
            autocast_precision=config.autocast_precision,
            trajectory_precision=config.trajectory_precision,
            logprob_precision=config.logprob_precision,
            shift=float(config.shift),
        )

    def _autocast_ctx(self):
        if torch.cuda.is_available() and self.autocast_precision in ("bf16", "fp16"):
            dtype = torch.bfloat16 if self.autocast_precision == "bf16" else torch.float16
            return torch.autocast("cuda", dtype)
        return nullcontext()

    def _build_contexts(self, prompt: str) -> Tuple[Any, Any, Any]:
        """Build (gen, cfg_text, cfg_img) KV contexts for a plain-T2I prompt.

        Mirrors ``InterleaveInferencer.interleave_inference`` (think=False, no
        image): ``cfg_text`` is the init snapshot taken *before* the prompt text
        (unconditional); ``gen`` and ``cfg_img`` both ingest the prompt.
        """
        inf = self.bundle.inferencer
        gen = inf.init_gen_context()
        cfg_img = deepcopy(gen)
        with torch.no_grad(), self._autocast_ctx():
            cfg_text = deepcopy(gen)  # snapshot before the prompt text → unconditional
            gen = inf.update_context_text(prompt, gen)
            cfg_img = inf.update_context_text(prompt, cfg_img)
        return gen, cfg_text, cfg_img

    def generate(self, req: RolloutReq) -> RolloutResp:
        """Run BAGEL T2I per-sample and pack one ``"image"`` track."""
        if req.sigmas is None:
            raise ValueError(
                "BagelPipeline.generate: req.sigmas is None. The hosting engine must call "
                "unirl.sde.runtime.ensure_req_sigmas(req, policy) before generate "
                "(policy = pipeline.build_schedule_policy())."
            )
        texts = req.primitives.get("text")
        if not isinstance(texts, Texts):
            raise TypeError(
                f"BagelPipeline.generate: req.primitives['text'] must be Texts, "
                f"got {type(texts).__name__ if texts is not None else 'None'}"
            )
        params = get_diffusion_params(req.sampling_params)
        if not isinstance(params, BagelDiffusionParams):
            raise TypeError(
                f"BagelPipeline.generate: sampling_params must be BagelDiffusionParams, got {type(params).__name__}"
            )

        prompts = list(texts.texts)
        n = len(prompts)
        sample_ids = list(req.sample_ids) if req.sample_ids else [f"s{i}" for i in range(n)]
        image_shape = (int(params.height), int(params.width))
        device = torch.device(self.bundle.device)
        schedule = req.sigmas.to(device)

        # Driver-authoritative per-sample x_T (NoiseRecipe), [n, seq, C] or None
        # (engine draws its own — tests / no driver recipe). Per-sample-unique +
        # per-rollout-fresh via the driver's r{rollout_id}:{sample_id} group ids.
        initial = NoiseRecipe.from_rollout_req(req).resolve(device=device, dtype=torch.float32)

        gen_list: List[Any] = []
        cfg_text_list: List[Any] = []
        cfg_img_list: List[Any] = []
        shapes: List[Tuple[int, int]] = []
        segments: List[LatentSegment] = []
        for i, prompt in enumerate(prompts):
            gen_ctx, cfg_text_ctx, cfg_img_ctx = self._build_contexts(prompt)
            cond_i = BagelDiffusionConditions.for_sample(
                gen_context=gen_ctx,
                cfg_text_context=cfg_text_ctx,
                cfg_img_context=cfg_img_ctx,
                image_shape=image_shape,
            )
            x0_i = initial[i] if initial is not None else None
            seg_i = self.diffusion.diffuse(cond_i, schedule=schedule, params=params, initial_latents=x0_i)
            segments.append(seg_i)
            gen_list.append(gen_ctx)
            cfg_text_list.append(cfg_text_ctx)
            cfg_img_list.append(cfg_img_ctx)
            shapes.append(image_shape)

        segment = self._batch_segments(segments)
        conditions = BagelDiffusionConditions(
            gen_contexts=gen_list,
            cfg_text_contexts=cfg_text_list,
            cfg_img_contexts=cfg_img_list,
            image_shapes=shapes,
        )
        images = self.vae_decode.decode(segment, image_shape=image_shape)

        return RolloutResp(
            tracks={
                "image": RolloutTrack(
                    sample_ids=sample_ids,
                    parent_ids=list(req.group_ids) if req.group_ids else None,
                    conditions=conditions.to_dict(),
                    segment=segment,
                    decoded=images,
                ),
            }
        )

    @staticmethod
    def _batch_segments(segments: List[LatentSegment]) -> LatentSegment:
        """Stack per-sample 1-row segments into one ``[N, ...]`` segment.

        With the per-rollout SDE window (``resolve_sde_indices(rollout_id)`` is shared
        across the group), every sample's ``sde_indices`` / ``indices`` / ``sigmas`` is
        identical, so the shared fields are taken from ``segments[0]`` and only the
        per-sample ``latents`` / ``sde_logp`` stack along the batch axis.
        """
        if len(segments) == 1:
            return segments[0]
        latents = torch.cat([s.latents for s in segments], dim=0)  # [N, K, seq, C]
        sde_logp = (
            torch.cat([s.sde_logp for s in segments], dim=0) if segments[0].sde_logp is not None else None
        )  # [N, S]
        return LatentSegment(
            latents=latents,
            sigmas=segments[0].sigmas,
            indices=segments[0].indices,
            sde_logp=sde_logp,
            sde_indices=segments[0].sde_indices,
        )


__all__ = ["BagelPipeline"]
