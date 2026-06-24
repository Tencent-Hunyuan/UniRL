"""BagelPipeline — RolloutReq → RolloutResp end-to-end for BAGEL-7B-MoT (T2I / it2i).

Four-tier flow, per-sample (navit ``bs=1``)::

    Texts [+ Images] ─build 3 KV contexts─▶ BagelDiffusionConditions ─diffuse─▶ LatentSegment ─vae_decode─▶ Images

Also serves the text-out modes (t2t / i2t / it2t, via :class:`BagelARStage`) and
the composed **t2ti** (native think-then-generate: the AR und path plans a
``<think>`` caption, then diffusion generates conditioned on it — one bundle, two
linked tracks).

Task routing (``_resolve_task``): explicit ``req.stage_config["task"]`` wins;
else both ``ar`` + ``diffusion`` sampling entries ⇒ ``t2ti``; ``ar`` only ⇒
text-out; an ``Images`` input ⇒ ``it2i`` (editing), else ``t2i``.

Per prompt the pipeline builds the three KV-cache contexts the sampler needs
(mirroring ``InterleaveInferencer.interleave_inference``, ``think=False``;
editing input order ``[image, text]``, vendor/inferencer.py:242-253):

- t2i:  ``gen`` = init + text; ``cfg_text`` = init snapshot before the text
  (unconditional); ``cfg_img`` = init + text (== gen — no image branch).
- it2i: ``gen`` = init + image(VAE+ViT) + text; ``cfg_text`` = init + image
  (drop-text branch); ``cfg_img`` = init + text (drop-image branch).

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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch

from unirl.models.types.pipeline import Pipeline
from unirl.sde.kernels import FlowSDEStrategy, StepStrategy
from unirl.sde.runtime import FlowMatchSchedulePolicy
from unirl.types.noise_recipe import NoiseRecipe
from unirl.types.primitives import Images, Texts
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp, RolloutTrack
from unirl.types.segments.latent import LatentSegment
from unirl.types.segments.text import TextSegment

from .ar import BagelARStage
from .conditions import BagelARConditions, BagelDiffusionConditions
from .diffusion import BagelDiffusionParams, BagelDiffusionStage
from .rl_ops import _to_device
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
        replay_mode: str = "train",
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
        # AR (text-out) stage for t2t / i2t / it2t; resolvable via the trainside
        # engine's ``stage_attrs=["ar"]``. Shares the bundle (same MoT root).
        self.ar = BagelARStage(
            model=bundle,
            autocast_precision=autocast_precision,
            logprob_precision=logprob_precision,
            replay_mode=replay_mode,
        )
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

    def _resize_input_image(self, image: Any) -> Any:
        """Canonical input-image preproc (inferencer.py:249): rgb → aspect-preserving
        stride-8 resize. Every image-input task funnels through this before the
        VAE/ViT branches so the chain has one home."""
        from .vendor.data.data_utils import pil_img2rgb

        return self.bundle.vae_transform.resize_transform(pil_img2rgb(image))

    def _extract_input_images(self, req: RolloutReq, task: str, *, n_prompts: Optional[int]) -> List[Any]:
        """Validated per-sample input PILs for image-input tasks (it2i / i2t / it2t).

        Requires an ``Images`` primitive, a ViT-loaded bundle (``enable_vit``),
        and — when prompts are present — a matching per-sample count.
        """
        images_prim = req.primitives.get("image")
        if not isinstance(images_prim, Images):
            raise TypeError(
                f"BagelPipeline.generate ({task}): req.primitives['image'] must be Images, "
                f"got {type(images_prim).__name__ if images_prim is not None else 'None'}"
            )
        if getattr(self.bundle.model, "vit_model", None) is None:
            raise ValueError(
                f"BagelPipeline.generate ({task}): the bundle was built without the und ViT; "
                "set BagelPipelineConfig.enable_vit=true for image-input tasks."
            )
        pil_images = [img.to_pil() for img in images_prim.to_list()]
        if n_prompts is not None and len(pil_images) != n_prompts:
            raise ValueError(
                f"BagelPipeline.generate ({task}): image count {len(pil_images)} != prompt count {n_prompts}"
            )
        return pil_images

    def _update_context_image(self, image: Any, gen_context: Any, *, vae: bool, vit: bool) -> Any:
        """Prefill one input image into a KV context (VAE and/or ViT branch).

        Mirrors the vendored ``InterleaveInferencer.update_context_image``
        (vendor/inferencer.py:62-96) with explicit device pinning: the vendored
        ``prepare_vae_images`` / ``prepare_vit_images`` build their tensors on
        CPU and the bundle's VAE carries no accelerate hooks, so ``padded_images``
        (and the packed index tensors) must be moved before the cache update.
        ``image`` is already ``resize_transform``-ed. Caller owns no_grad+autocast.
        """
        bagel = self.bundle.model
        device = torch.device(self.bundle.device)
        ctx = gen_context
        if vae:
            gi, kv_lens, ropes = bagel.prepare_vae_images(
                curr_kvlens=ctx["kv_lens"],
                curr_rope=ctx["ropes"],
                images=[image],
                transforms=self.bundle.vae_transform,
                new_token_ids=self.bundle.new_token_ids,
            )
            gi = _to_device(gi, device)
            gi["padded_images"] = gi["padded_images"].to(dtype=self.bundle.vae_dtype)
            past = bagel.forward_cache_update_vae(self.bundle.vae, ctx["past_key_values"], **gi)
            ctx = {"kv_lens": kv_lens, "ropes": ropes, "past_key_values": past}
        if vit:
            gi, kv_lens, ropes = bagel.prepare_vit_images(
                curr_kvlens=ctx["kv_lens"],
                curr_rope=ctx["ropes"],
                images=[image],
                transforms=self.bundle.vit_transform,
                new_token_ids=self.bundle.new_token_ids,
            )
            gi = _to_device(gi, device)
            past = bagel.forward_cache_update_vit(ctx["past_key_values"], **gi)
            ctx = {"kv_lens": kv_lens, "ropes": ropes, "past_key_values": past}
        return ctx

    def _build_contexts(self, prompt: str, image: Optional[Any] = None) -> Tuple[Any, Any, Any]:
        """Build (gen, cfg_text, cfg_img) KV contexts for T2I or editing (it2i).

        Mirrors ``InterleaveInferencer.interleave_inference`` exactly
        (vendor/inferencer.py:242-253; editing input order ``[image, text]``)::

            t2i  (image=None): gen      = init + text
                               cfg_text = init                  (unconditional)
                               cfg_img  = init + text           (no image branch)
            it2i (image set):  gen      = init + image(VAE+ViT) + text
                               cfg_text = init + image          (drop-text branch)
                               cfg_img  = init + text           (drop-image branch)

        ``image`` is a raw PIL image; preprocessing matches inferencer.py:249
        (``pil_img2rgb`` + ``vae_transform.resize_transform``).
        """
        inf = self.bundle.inferencer
        gen = inf.init_gen_context()
        cfg_img = deepcopy(gen)
        with torch.no_grad(), self._autocast_ctx():
            if image is not None:
                gen = self._update_context_image(self._resize_input_image(image), gen, vae=True, vit=True)
            cfg_text = deepcopy(gen)  # snapshot before the prompt text → drop-text branch
            gen = inf.update_context_text(prompt, gen)
            cfg_img = inf.update_context_text(prompt, cfg_img)
        return gen, cfg_text, cfg_img

    @staticmethod
    def _resolve_task(req: RolloutReq) -> str:
        """Resolve the task mode: explicit ``stage_config["task"]`` wins, else infer.

        Inference from the modality-keyed ``sampling_params`` dict: both
        ``"ar"`` + ``"diffusion"`` ⇒ ``t2ti`` (think-then-generate); ``"ar"``
        only ⇒ text-out (``it2t`` with an ``Images`` input, else ``t2t``; pure
        ``i2t`` — image, no prompt — must be explicit); ``"diffusion"`` only ⇒
        image-out (``it2i`` with an ``Images`` input, else ``t2i``).
        """
        task = req.stage_config.get("task")
        if task is not None:
            return str(task)
        sp = req.sampling_params
        if "ar" in sp and "diffusion" in sp:
            return "t2ti"
        has_image = isinstance(req.primitives.get("image"), Images)
        if "ar" in sp:
            return "it2t" if has_image else "t2t"
        return "it2i" if has_image else "t2i"

    def generate(self, req: RolloutReq) -> RolloutResp:
        """Dispatch on the resolved task and pack one track per request."""
        task = self._resolve_task(req)
        if task in ("t2i", "it2i"):
            return self._generate_image(req, task)
        if task in ("t2t", "i2t", "it2t"):
            return self._generate_text(req, task)
        if task == "t2ti":
            return self._generate_t2ti(req)
        raise ValueError(
            f"BagelPipeline.generate: unsupported task {task!r}; "
            "expected one of 't2i', 'it2i', 't2t', 'i2t', 'it2t', 't2ti'."
        )

    def _generate_image(self, req: RolloutReq, task: str) -> RolloutResp:
        """Run BAGEL image-out (t2i / it2i) per-sample and pack one ``"image"`` track."""
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
        params = req.sampling_params.get("diffusion")
        if not isinstance(params, BagelDiffusionParams):
            raise TypeError(
                f"BagelPipeline.generate: sampling_params must be BagelDiffusionParams, got {type(params).__name__}"
            )

        prompts = list(texts.texts)
        n = len(prompts)

        # it2i: per-sample input images — editing prefills the input image
        # through BOTH the VAE branch (pixel fidelity) and the ViT branch
        # (semantics) in _build_contexts.
        pil_images = self._extract_input_images(req, task, n_prompts=n) if task == "it2i" else None

        sample_ids = list(req.sample_ids) if req.sample_ids else [f"s{i}" for i in range(n)]
        image_shape = (int(params.height), int(params.width))

        contexts = [
            self._build_contexts(prompt, image=pil_images[i] if pil_images is not None else None)
            for i, prompt in enumerate(prompts)
        ]
        segment, conditions, images = self._diffuse_and_decode(
            contexts, prompts=prompts, params=params, req=req, image_shape=image_shape
        )

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

    def _diffuse_and_decode(
        self,
        contexts: List[Tuple[Any, Any, Any]],
        *,
        prompts: List[str],
        params: BagelDiffusionParams,
        req: RolloutReq,
        image_shape: Tuple[int, int],
    ) -> Tuple[LatentSegment, BagelDiffusionConditions, Images]:
        """Diffuse per-sample over prebuilt ``(gen, cfg_text, cfg_img)`` contexts,
        batch the latents, build the ``BagelDiffusionConditions``, and VAE-decode.

        Shared by image-out (t2i / it2i) and think-then-generate (t2ti). x_T is
        driver-authored per-sample via :class:`NoiseRecipe` (keyed on the request's
        sample ids — engine draws its own when no driver recipe is present); the
        three CFG contexts ride verbatim into the stored conditions for
        frozen-context replay.
        """
        device = torch.device(self.bundle.device)
        schedule = req.sigmas.to(device)
        initial = NoiseRecipe.from_rollout_req(req).resolve(device=device, dtype=torch.float32)

        gen_list: List[Any] = []
        cfg_text_list: List[Any] = []
        cfg_img_list: List[Any] = []
        shapes: List[Tuple[int, int]] = []
        segments: List[LatentSegment] = []
        for i, (gen_ctx, cfg_text_ctx, cfg_img_ctx) in enumerate(contexts):
            cond_i = BagelDiffusionConditions.for_sample(
                gen_context=gen_ctx,
                cfg_text_context=cfg_text_ctx,
                cfg_img_context=cfg_img_ctx,
                image_shape=image_shape,
                prompt=prompts[i],
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
            prompts=list(prompts),
            image_shapes=shapes,
        )
        images = self.vae_decode.decode(segment, image_shape=image_shape)
        return segment, conditions, images

    @staticmethod
    def _batch_segments(segments: List[LatentSegment]) -> LatentSegment:
        """Stack per-sample 1-row segments into one ``[N, ...]`` segment.

        With the per-rollout SDE window (``resolve_sde_indices(rollout_id)`` is shared
        across the group), every sample's ``sde_indices`` / ``indices`` / ``sigmas`` is
        identical, so the shared fields are taken from ``segments[0]`` and only the
        per-sample ``latents`` / ``sde_logp`` stack along the batch axis. The
        framework manages the row→sample mapping at concat time (matching the
        ``LatentSegment`` field set used by every other diffusion stage).
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

    # ------------------------------------------------------------------
    # Text-out (t2t / i2t / it2t)
    # ------------------------------------------------------------------

    def _generate_text(self, req: RolloutReq, task: str) -> RolloutResp:
        """Run BAGEL text-out per-sample and pack one ``"ar"`` track.

        Builds per-sample RAW prompt splits (see :class:`BagelARConditions`)
        mirroring the vendored understanding flow (inferencer.py:242-260,
        ``understanding_output=True``): image ingested ViT-only, then the
        prompt text; ``BagelARStage`` owns prefill + decode + replay.
        """
        ar_params = req.sampling_params.get("ar")
        if ar_params is None:
            raise TypeError(
                f"BagelPipeline.generate ({task}): sampling_params must carry ARSamplingParams, "
                f"got {type(req.sampling_params).__name__}"
            )

        texts = req.primitives.get("text")
        prompts: Optional[List[str]] = list(texts.texts) if isinstance(texts, Texts) else None
        if task in ("t2t", "it2t") and prompts is None:
            raise TypeError(
                f"BagelPipeline.generate ({task}): req.primitives['text'] must be Texts, "
                f"got {type(texts).__name__ if texts is not None else 'None'}"
            )

        pil_images: Optional[List[Any]] = None
        if task in ("i2t", "it2t"):
            pil_images = self._extract_input_images(req, task, n_prompts=len(prompts) if prompts is not None else None)

        n = len(prompts) if prompts is not None else len(pil_images)
        sample_ids = list(req.sample_ids) if req.sample_ids else [f"s{i}" for i in range(n)]
        ntk = self.bundle.new_token_ids
        tokenizer = self.bundle.tokenizer

        splits_per_sample: List[List[Dict[str, Any]]] = []
        for i in range(n):
            splits: List[Dict[str, Any]] = []
            if pil_images is not None:
                # Understanding preproc chain (inferencer.py:249-250, vae=False):
                # rgb → vae resize → vit_transform; store the FINAL pixels so
                # rollout and replay consume byte-identical inputs.
                img = self._resize_input_image(pil_images[i])
                splits.append({"kind": "vit", "image": self.bundle.vit_transform(img)})
            if prompts is not None:
                # bos/eos (<|im_start|>/<|im_end|>) wrap, as prepare_prompts does
                # (vendor bagel.py:246) — stored wrapped so replay is tokenizer-free.
                ids = [ntk["bos_token_id"]] + tokenizer.encode(prompts[i]) + [ntk["eos_token_id"]]
                splits.append({"kind": "text", "ids": torch.tensor(ids, dtype=torch.long)})
            splits_per_sample.append(splits)

        conditions = BagelARConditions(prompt_splits=splits_per_sample)
        segment = self.ar.autoregress(conditions, sampling_params=ar_params)
        decoded = self._detokenize(segment)

        return RolloutResp(
            tracks={
                "ar": RolloutTrack(
                    sample_ids=sample_ids,
                    parent_ids=list(req.group_ids) if req.group_ids else None,
                    conditions=conditions.to_dict(),
                    segment=segment,
                    decoded=decoded,
                ),
            }
        )

    def _detokenize(self, segment: TextSegment) -> Texts:
        """Decode packed response tokens to strings, stripped at ``<|im_end|>``.

        The segment holds SAMPLED tokens only (no leading bos — unlike the
        vendored ``gen_text``, which also strips a leading ``<|im_start|>``).
        """
        cu = [int(c) for c in segment.cu_seqlens.tolist()]
        out: List[str] = []
        for i in range(len(cu) - 1):
            toks = segment.tokens[cu[i] : cu[i + 1]].tolist()
            out.append(self.bundle.tokenizer.decode(toks).split("<|im_end|>")[0])
        return Texts(texts=out)

    # ------------------------------------------------------------------
    # Composed think-then-generate (t2ti)
    # ------------------------------------------------------------------

    def _build_think_contexts(self, system_prompt: str, prompt: str, think_text: str) -> Tuple[Any, Any, Any]:
        """Build (gen, cfg_text, cfg_img) KV contexts for native think-gen (t2ti).

        Mirrors ``interleave_inference(think=True, understanding_output=False)``
        (vendor/inferencer.py:234-272)::

            gen      = init + system + prompt + think_text
            cfg_text = init + system                  (drop prompt + think)
            cfg_img  = init + system + prompt         (drop think only — so
                       cfg_img_scale weights the planning text's influence)

        ``think_text`` is the AR-decoded planning string (stripped at
        ``<|im_end|>``), re-encoded into the gen context exactly as the vendor's
        ``update_context_text(gen_text, ...)`` step. The ``[system, prompt]``
        prefix is byte-identical to the AR stage's prefill (both wrap as
        ``[bos] + encode(text) + [eos]``), so the planning text the image
        conditions on is the one the AR actually generated.
        """
        inf = self.bundle.inferencer
        gen = inf.init_gen_context()
        cfg_img = deepcopy(gen)
        with torch.no_grad(), self._autocast_ctx():
            gen = inf.update_context_text(system_prompt, gen)
            cfg_img = inf.update_context_text(system_prompt, cfg_img)
            cfg_text = deepcopy(gen)  # init + system → drop-prompt-and-think branch
            gen = inf.update_context_text(prompt, gen)
            cfg_img = inf.update_context_text(prompt, cfg_img)
            gen = inf.update_context_text(think_text, gen)
        return gen, cfg_text, cfg_img

    def _generate_t2ti(self, req: RolloutReq) -> RolloutResp:
        """BAGEL native think-then-generate (t2ti).

        The AR und path plans a ``<think>`` caption from ``[system + prompt]``,
        then diffusion generates the image conditioned on ``[system + prompt +
        think]`` — one bundle, two stages. Emits two linked tracks: ``"ar"`` (the
        planning text, grouped by prompt) and ``"image"`` (``parent_track="ar"``
        → grouped by the rewrite, mirroring the PE composition's lineage). The
        request carries both ``ar`` + ``diffusion`` sampling entries.
        """
        if req.sigmas is None:
            raise ValueError(
                "BagelPipeline.generate (t2ti): req.sigmas is None — the hosting engine must call "
                "ensure_req_sigmas(req, pipeline.build_schedule_policy()) before generate."
            )
        ar_params = req.sampling_params.get("ar")
        diff_params = req.sampling_params.get("diffusion")
        if ar_params is None or not isinstance(diff_params, BagelDiffusionParams):
            raise TypeError(
                "BagelPipeline.generate (t2ti): sampling_params must carry both an 'ar' (ARSamplingParams) "
                f"and a 'diffusion' (BagelDiffusionParams) entry; got keys {sorted(req.sampling_params)}."
            )
        texts = req.primitives.get("text")
        if not isinstance(texts, Texts):
            raise TypeError(
                f"BagelPipeline.generate (t2ti): req.primitives['text'] must be Texts, "
                f"got {type(texts).__name__ if texts is not None else 'None'}"
            )

        # Lazy: the vendor module hard-imports flash_attn; the bundle is loaded by now.
        from .vendor.inferencer import GEN_THINK_SYSTEM_PROMPT

        prompts = list(texts.texts)
        n = len(prompts)
        sample_ids = list(req.sample_ids) if req.sample_ids else [f"s{i}" for i in range(n)]
        ar_sample_ids = [f"{sid}#think" for sid in sample_ids]
        image_shape = (int(diff_params.height), int(diff_params.width))
        ntk = self.bundle.new_token_ids
        tokenizer = self.bundle.tokenizer

        # Stage 1 — AR plans the think text from [system, prompt] (bos/eos-wrapped
        # text splits, as the vendor's two update_context_text calls do).
        ar_splits: List[List[Dict[str, Any]]] = []
        for prompt in prompts:
            sys_ids = [ntk["bos_token_id"]] + tokenizer.encode(GEN_THINK_SYSTEM_PROMPT) + [ntk["eos_token_id"]]
            pr_ids = [ntk["bos_token_id"]] + tokenizer.encode(prompt) + [ntk["eos_token_id"]]
            ar_splits.append(
                [
                    {"kind": "text", "ids": torch.tensor(sys_ids, dtype=torch.long)},
                    {"kind": "text", "ids": torch.tensor(pr_ids, dtype=torch.long)},
                ]
            )
        ar_conditions = BagelARConditions(prompt_splits=ar_splits)
        ar_segment = self.ar.autoregress(ar_conditions, sampling_params=ar_params)
        think_texts = self._detokenize(ar_segment)

        # Stage 2 — diffusion conditioned on [system + prompt + think].
        contexts = [
            self._build_think_contexts(GEN_THINK_SYSTEM_PROMPT, prompts[i], think_texts.texts[i]) for i in range(n)
        ]
        segment, conditions, images = self._diffuse_and_decode(
            contexts, prompts=prompts, params=diff_params, req=req, image_shape=image_shape
        )

        return RolloutResp(
            tracks={
                "ar": RolloutTrack(
                    sample_ids=ar_sample_ids,
                    parent_ids=list(req.group_ids) if req.group_ids else None,
                    conditions=ar_conditions.to_dict(),
                    segment=ar_segment,
                    decoded=think_texts,
                ),
                "image": RolloutTrack(
                    sample_ids=sample_ids,
                    parent_track="ar",
                    parent_ids=ar_sample_ids,
                    conditions=conditions.to_dict(),
                    segment=segment,
                    decoded=images,
                ),
            }
        )


__all__ = ["BagelPipeline"]
