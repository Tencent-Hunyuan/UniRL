"""BagelPipeline — ``Sample → Sample`` end-to-end for BAGEL-7B-MoT (T2I / it2i)."""

from __future__ import annotations

import logging
from collections import OrderedDict
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch

from unirl.models.types.pipeline import Pipeline
from unirl.sde.kernels import FlowSDEStrategy, StepStrategy
from unirl.sde.runtime import FlowMatchSchedulePolicy
from unirl.types.noise_recipe import NoiseRecipe
from unirl.types.primitives import Images, Texts
from unirl.types.sample import Sample
from unirl.types.sampling import ARSamplingParams, DiffusionSamplingParams
from unirl.types.segments.latent import LatentSegment
from unirl.types.segments.text import TextSegment

from .ar import BagelARStage
from .chat_template import BagelChatTemplateStage
from .conditions import BagelARConditions, BagelDiffusionConditions
from .diffusion import BagelDiffusionParams, BagelDiffusionStage
from .rl_ops import _to_device
from .vae import BagelVAEDecodeStage, BagelVAEEncodeStage, bagel_latent_shape

if TYPE_CHECKING:
    from .bundle import BagelBundle

logger = logging.getLogger(__name__)


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
        max_prompt_length: int = 8192,
        cache_t2i_contexts: Optional[bool] = None,
        context_cache_size: Optional[int] = None,
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
        self.vae_encode = BagelVAEEncodeStage(bundle)
        self.chat_template = BagelChatTemplateStage(bundle, max_prompt_length=max_prompt_length)
        self.ar = BagelARStage(
            model=bundle,
            autocast_precision=autocast_precision,
            logprob_precision=logprob_precision,
            replay_mode=replay_mode,
        )
        self.autocast_precision = autocast_precision
        self.shift = shift
        cache_config = getattr(bundle, "config", None)
        cache_t2i_contexts = (
            _cfg_get(cache_config, "cache_t2i_contexts", True) if cache_t2i_contexts is None else cache_t2i_contexts
        )
        context_cache_size = (
            _cfg_get(cache_config, "context_cache_size", 32) if context_cache_size is None else context_cache_size
        )
        self._cache_t2i_contexts = bool(cache_t2i_contexts)
        self._context_cache_size = max(1, int(context_cache_size))
        self._t2i_context_cache: "OrderedDict[str, Tuple[Any, Any, Any]]" = OrderedDict()
        self._und_frozen: Optional[bool] = None

    @classmethod
    def latent_shape(cls, *, model_config: Any, sampling_spec: Any) -> Tuple[int, ...]:
        """Packed per-sample x_T shape ``(seq, p²·z)`` for the driver NoiseRecipe."""
        cfg = _cfg_get(model_config, "config", model_config)
        patch = int(_cfg_get(cfg, "latent_patch_size", 2))
        vae_ds = int(_cfg_get(cfg, "vae_downsample", 8))
        z = int(_cfg_get(cfg, "latent_channels", 16))
        H, W = int(sampling_spec.height), int(sampling_spec.width)
        return bagel_latent_shape((H, W), latent_downsample=vae_ds * patch, latent_patch_size=patch, latent_channels=z)

    def build_schedule_policy(self) -> FlowMatchSchedulePolicy:
        """Static-shift FlowMatch σ policy (BAGEL uses no dynamic shifting)."""
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
        """Canonical input-image preproc (inferencer.py:249): rgb → aspect-preserving"""
        from .vendor.data.data_utils import pil_img2rgb

        return self.bundle.vae_transform.resize_transform(pil_img2rgb(image))

    def _extract_input_images(self, conditioning: List[Any], task: str, *, n_prompts: Optional[int]) -> List[Any]:
        """Validated per-sample input PILs for image-input tasks (it2i / i2t / it2t)."""
        images_prim = next((c for c in conditioning if isinstance(c, Images)), None)
        if not isinstance(images_prim, Images):
            raise TypeError(
                f"BagelPipeline.generate ({task}): expected an Images input in sample.conditioning(), found none"
            )
        if getattr(self.bundle.model, "vit_model", None) is None:
            raise ValueError(
                f"BagelPipeline.generate ({task}): the bundle was built without the und ViT; "
                "set BagelPipelineConfig.enable_vit=true for image-input tasks."
            )
        pil_images = images_prim.to_pils()
        if n_prompts is not None and len(pil_images) != n_prompts:
            raise ValueError(
                f"BagelPipeline.generate ({task}): image count {len(pil_images)} != prompt count {n_prompts}"
            )
        return pil_images

    def _update_context_image(self, image: Any, gen_context: Any, *, vae: bool, vit: bool) -> Any:
        """Prefill one input image into a KV context (VAE and/or ViT branch)."""
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
            vae_mod, proj = self.bundle.vae, bagel.vae2llm

            def _vae_encode(x: torch.Tensor) -> torch.Tensor:
                return vae_mod.encode(x.to(dtype=next(vae_mod.parameters()).dtype)).to(
                    dtype=next(proj.parameters()).dtype
                )

            past = bagel.forward_cache_update_vae(SimpleNamespace(encode=_vae_encode), ctx["past_key_values"], **gi)
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
        """Build (gen, cfg_text, cfg_img) KV contexts for T2I or editing (it2i)."""
        inf = self.bundle.inferencer
        gen = inf.init_gen_context()
        cfg_img = deepcopy(gen)
        mot = self.bundle.transformer
        was_training = mot.training
        mot.eval()
        try:
            with torch.no_grad(), self._autocast_ctx():
                if image is not None:
                    gen = self._update_context_image(self._resize_input_image(image), gen, vae=True, vit=True)
                cfg_text = deepcopy(gen)
                gen = inf.update_context_text(prompt, gen)
                cfg_img = inf.update_context_text(prompt, cfg_img)
        finally:
            mot.train(was_training)
        return gen, cfg_text, cfg_img

    def _t2i_cache_enabled(self) -> bool:
        """Whether the T2I context cache is safe to use right now."""
        if not self._cache_t2i_contexts:
            return False
        if self._und_frozen is None:
            try:
                und_trainable = [
                    n for n, p in self.bundle.transformer.named_parameters() if p.requires_grad and "moe_gen" not in n
                ]
            except Exception:  # pragma: no cover - be conservative if not introspectable
                und_trainable = ["<introspection-failed>"]
            self._und_frozen = not und_trainable
            if not self._und_frozen:
                logger.warning(
                    "BagelPipeline: T2I context cache DISABLED — the prompt-prefill "
                    "(und/shared) path has %d trainable param(s) (e.g. %s), so cached "
                    "contexts could go stale. Caching is only safe with gen-only "
                    "(*_moe_gen) LoRA; set cache_t2i_contexts=false to silence.",
                    len(und_trainable),
                    und_trainable[:3],
                )
            else:
                logger.info("BagelPipeline: T2I context cache enabled (prompt-prefill path is frozen).")
        return self._und_frozen

    def _build_contexts_cached(self, prompt: str) -> Tuple[Any, Any, Any]:
        """Memoized :meth:`_build_contexts` for the T2I path (image-free)."""
        cache = self._t2i_context_cache
        hit = cache.get(prompt)
        if hit is not None:
            cache.move_to_end(prompt)
            return hit
        ctx = self._build_contexts(prompt, image=None)
        cache[prompt] = ctx
        while len(cache) > self._context_cache_size:
            cache.popitem(last=False)
        return ctx

    def clear_context_cache(self) -> None:
        """Drop the cached T2I prompt contexts (frees their prompt KV caches)."""
        self._t2i_context_cache.clear()

    def build_conditions(
        self,
        texts: Texts,
        *,
        negatives: Optional[Texts] = None,
        guidance_scale: float = 1.0,
        image_shape: Tuple[int, int] = (512, 512),
    ) -> BagelDiffusionConditions:
        """Encode prompts into T2I ``BagelDiffusionConditions`` (no diffusion run)."""
        del guidance_scale
        if negatives is not None:
            raise ValueError(
                "BagelPipeline.build_conditions: Bagel CFG uses prefilled cfg contexts, not "
                "negative prompt embeddings — pass negatives=None and set cfg_*_scale on the params."
            )
        contexts = [self._build_contexts(prompt, image=None) for prompt in texts.texts]
        shape = image_shape
        return BagelDiffusionConditions(
            gen_contexts=[c[0] for c in contexts],
            cfg_text_contexts=[c[1] for c in contexts],
            cfg_img_contexts=[c[2] for c in contexts],
            prompts=list(texts.texts),
            image_shapes=[shape] * len(texts.texts),
        )

    @staticmethod
    def _resolve_task(sample: Sample) -> str:
        """Resolve the task mode: explicit ``parts[0].control["task"]`` wins, else infer."""
        task = (sample.parts[0].control or {}).get("task")
        if task is not None:
            return str(task)
        frontier_params = sample.parts[-1].sampling_params
        has_ar = isinstance(frontier_params, ARSamplingParams)
        has_diff = isinstance(frontier_params, DiffusionSamplingParams)
        has_current_ar_prelude = (
            has_diff and len(sample.parts) >= 2 and isinstance(sample.parts[-2].sampling_params, ARSamplingParams)
        )
        has_image = any("image" in p.primitives for p in sample.parts[:-1])
        if has_current_ar_prelude:
            return "t2ti"
        if has_ar:
            return "it2t" if has_image else "t2t"
        return "it2i" if has_image else "t2i"

    def generate(self, sample: Sample) -> Sample:
        """Dispatch on the resolved task and fill the gen Part(s)."""
        task = self._resolve_task(sample)
        if task in ("t2i", "it2i"):
            return self._generate_image(sample, task)
        if task in ("t2t", "i2t", "it2t"):
            return self._generate_text(sample, task)
        if task == "t2ti":
            return self._generate_t2ti(sample)
        raise ValueError(
            f"BagelPipeline.generate: unsupported task {task!r}; "
            "expected one of 't2i', 'it2i', 't2t', 'i2t', 'it2t', 't2ti'."
        )

    def _generate_image(self, sample: Sample, task: str) -> Sample:
        """Run BAGEL image-out (t2i / it2i) per-sample and fill the diffusion gen Part."""
        frontier = sample.frontier_gen_part(DiffusionSamplingParams)
        params = frontier.sampling_params
        if not isinstance(params, BagelDiffusionParams):
            raise TypeError(
                f"BagelPipeline.generate: gen Part sampling_params must be BagelDiffusionParams, "
                f"got {type(params).__name__}"
            )
        if params.sigmas is None:
            raise ValueError(
                "BagelPipeline.generate: gen part sampling_params.sigmas is None. The hosting engine must "
                "pin σ (policy = pipeline.build_schedule_policy()) before generate."
            )
        conditioning = sample.conditioning()
        texts = conditioning[0] if conditioning else None
        if not isinstance(texts, Texts):
            raise TypeError(
                f"BagelPipeline.generate: expected a Texts prompt from sample.conditioning()[0], "
                f"got {type(texts).__name__ if texts is not None else 'None'}"
            )

        prompts = list(texts.texts)
        n = len(prompts)

        pil_images = self._extract_input_images(conditioning, task, n_prompts=n) if task == "it2i" else None

        image_shape = (int(params.height), int(params.width))

        if task == "t2i" and pil_images is None and self._t2i_cache_enabled():
            contexts = [self._build_contexts_cached(prompt) for prompt in prompts]
        else:
            contexts = [
                self._build_contexts(prompt, image=pil_images[i] if pil_images is not None else None)
                for i, prompt in enumerate(prompts)
            ]
        segment, conditions, images = self._diffuse_and_decode(
            contexts,
            prompts=prompts,
            params=params,
            sample=sample,
            image_shape=image_shape,
        )

        filled = frontier.fill(segment=segment, primitives={"image": images}, conditions=conditions.to_dict())
        return sample.replace_frontier(filled)

    def _diffuse_and_decode(
        self,
        contexts: List[Tuple[Any, Any, Any]],
        *,
        prompts: List[str],
        params: BagelDiffusionParams,
        sample: Sample,
        image_shape: Tuple[int, int],
    ) -> Tuple[LatentSegment, BagelDiffusionConditions, Images]:
        """Diffuse per-sample over prebuilt ``(gen, cfg_text, cfg_img)`` contexts,"""
        device = torch.device(self.bundle.device)
        schedule = params.sigmas.to(device)
        initial = NoiseRecipe.from_sample(sample).resolve(device=device, dtype=torch.float32)

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
        """Stack per-sample 1-row segments into one ``[N, ...]`` segment."""
        if len(segments) == 1:
            return segments[0]
        latents = torch.cat([s.latents for s in segments], dim=0)
        sde_logp = torch.cat([s.sde_logp for s in segments], dim=0) if segments[0].sde_logp is not None else None
        sde_means = torch.cat([s.sde_means for s in segments], dim=0) if segments[0].sde_means is not None else None
        return LatentSegment(
            latents=latents,
            sigmas=segments[0].sigmas,
            indices=segments[0].indices,
            sde_logp=sde_logp,
            sde_means=sde_means,
            sde_indices=segments[0].sde_indices,
        )

    def _generate_text(self, sample: Sample, task: str) -> Sample:
        """Run BAGEL text-out per-sample and fill the AR gen Part."""
        frontier = sample.frontier_gen_part(ARSamplingParams)
        ar_params = frontier.sampling_params
        if ar_params is None:
            raise TypeError(f"BagelPipeline.generate ({task}): gen Part must carry ARSamplingParams")

        conditioning = sample.conditioning()
        texts = conditioning[0] if conditioning else None
        prompts: Optional[List[str]] = list(texts.texts) if isinstance(texts, Texts) else None
        if task in ("t2t", "it2t") and prompts is None:
            raise TypeError(
                f"BagelPipeline.generate ({task}): expected a Texts prompt from sample.conditioning()[0], "
                f"got {type(texts).__name__ if texts is not None else 'None'}"
            )

        pil_images: Optional[List[Any]] = None
        if task in ("i2t", "it2t"):
            pil_images = self._extract_input_images(
                conditioning, task, n_prompts=len(prompts) if prompts is not None else None
            )

        n = len(prompts) if prompts is not None else len(pil_images)
        ntk = self.bundle.new_token_ids
        tokenizer = self.bundle.tokenizer

        splits_per_sample: List[List[Dict[str, Any]]] = []
        for i in range(n):
            splits: List[Dict[str, Any]] = []
            if pil_images is not None:
                img = self._resize_input_image(pil_images[i])
                splits.append({"kind": "vit", "image": self.bundle.vit_transform(img)})
            if prompts is not None:
                ids = [ntk["bos_token_id"]] + tokenizer.encode(prompts[i]) + [ntk["eos_token_id"]]
                splits.append({"kind": "text", "ids": torch.tensor(ids, dtype=torch.long)})
            splits_per_sample.append(splits)

        conditions = BagelARConditions(prompt_splits=splits_per_sample)
        segment = self.ar.autoregress(conditions, sampling_params=ar_params)
        decoded = self._detokenize(segment)

        filled = frontier.fill(segment=segment, primitives={"text": decoded}, conditions=conditions.to_dict())
        return sample.replace_frontier(filled)

    def _detokenize(self, segment: TextSegment) -> Texts:
        """Decode packed response tokens to strings, stripped at ``<|im_end|>``."""
        cu = [int(c) for c in segment.cu_seqlens.tolist()]
        out: List[str] = []
        for i in range(len(cu) - 1):
            toks = segment.tokens[cu[i] : cu[i + 1]].tolist()
            out.append(self.bundle.tokenizer.decode(toks).split("<|im_end|>")[0])
        return Texts(texts=out)

    def _build_think_contexts(self, system_prompt: str, prompt: str, think_text: str) -> Tuple[Any, Any, Any]:
        """Build (gen, cfg_text, cfg_img) KV contexts for native think-gen (t2ti)."""
        inf = self.bundle.inferencer
        gen = inf.init_gen_context()
        cfg_img = deepcopy(gen)
        with torch.no_grad(), self._autocast_ctx():
            gen = inf.update_context_text(system_prompt, gen)
            cfg_img = inf.update_context_text(system_prompt, cfg_img)
            cfg_text = deepcopy(gen)
            gen = inf.update_context_text(prompt, gen)
            cfg_img = inf.update_context_text(prompt, cfg_img)
            gen = inf.update_context_text(think_text, gen)
        return gen, cfg_text, cfg_img

    def _generate_t2ti(self, sample: Sample) -> Sample:
        """Run the pre-forked P → P*N → P*N*M think-then-generate lineage."""
        if len(sample.parts) < 2:
            raise ValueError("BagelPipeline.generate (t2ti): expected trailing [AR, diffusion] generation Parts")
        ar_idx = len(sample.parts) - 2
        image_idx = len(sample.parts) - 1
        ar_part = sample.parts[ar_idx]
        image_part = sample.parts[image_idx]
        if not isinstance(ar_part.sampling_params, ARSamplingParams) or not isinstance(
            image_part.sampling_params, DiffusionSamplingParams
        ):
            raise ValueError(
                "BagelPipeline.generate (t2ti): expected the current trailing Parts to be "
                f"[ARSamplingParams, DiffusionSamplingParams], got "
                f"[{type(ar_part.sampling_params).__name__}, {type(image_part.sampling_params).__name__}]"
            )
        ar_params = ar_part.sampling_params
        diff_params = image_part.sampling_params
        if not isinstance(diff_params, BagelDiffusionParams):
            raise TypeError(
                "BagelPipeline.generate (t2ti): diffusion gen Part must carry "
                f"BagelDiffusionParams, got {type(diff_params).__name__}"
            )
        if diff_params.sigmas is None:
            raise ValueError(
                "BagelPipeline.generate (t2ti): diffusion gen part sigmas is None; "
                "the hosting engine must pin the schedule before generate."
            )

        ar_texts = [value for value in sample.conditioning_at(ar_idx) if isinstance(value, Texts)]
        if len(ar_texts) != 1:
            raise TypeError(
                "BagelPipeline.generate (t2ti): expected exactly one Texts input "
                f"for the AR frontier, got {len(ar_texts)}"
            )
        prompts = list(ar_texts[0].texts)
        if len(prompts) != len(ar_part.sample_ids):
            raise ValueError(
                f"BagelPipeline.generate (t2ti): AR-aligned prompt count {len(prompts)} "
                f"!= AR sample count {len(ar_part.sample_ids)}"
            )

        from .vendor.inferencer import GEN_THINK_SYSTEM_PROMPT

        ntk = self.bundle.new_token_ids
        tokenizer = self.bundle.tokenizer
        ar_splits: List[List[Dict[str, Any]]] = []
        for prompt in prompts:
            sys_ids = [ntk["bos_token_id"]] + tokenizer.encode(GEN_THINK_SYSTEM_PROMPT) + [ntk["eos_token_id"]]
            prompt_ids = [ntk["bos_token_id"]] + tokenizer.encode(prompt) + [ntk["eos_token_id"]]
            ar_splits.append(
                [
                    {"kind": "text", "ids": torch.tensor(sys_ids, dtype=torch.long)},
                    {"kind": "text", "ids": torch.tensor(prompt_ids, dtype=torch.long)},
                ]
            )

        ar_conditions = BagelARConditions(prompt_splits=ar_splits)
        ar_segment = self.ar.autoregress(ar_conditions, sampling_params=ar_params)
        thinking = self._detokenize(ar_segment)
        if len(thinking.texts) != len(ar_part.sample_ids):
            raise RuntimeError(
                f"BagelPipeline.generate (t2ti): AR produced {len(thinking.texts)} "
                f"thinking texts for {len(ar_part.sample_ids)} AR samples"
            )

        new_parts = list(sample.parts)
        new_parts[ar_idx] = ar_part.fill(
            segment=ar_segment,
            primitives={"text": thinking},
            conditions=ar_conditions.to_dict(),
        )
        partially_filled = sample.with_parts(new_parts)

        image_texts = [value for value in partially_filled.conditioning() if isinstance(value, Texts)]
        if len(image_texts) < 2:
            raise RuntimeError(
                "BagelPipeline.generate (t2ti): image frontier did not surface "
                "both the original prompt and generated thinking text"
            )
        image_prompts = list(image_texts[0].texts)
        image_thinking = list(image_texts[-1].texts)
        expected_images = len(image_part.sample_ids)
        if len(image_prompts) != expected_images or len(image_thinking) != expected_images:
            raise ValueError(
                "BagelPipeline.generate (t2ti): image-aligned conditioning counts "
                f"prompt={len(image_prompts)}, thinking={len(image_thinking)}, "
                f"expected={expected_images}"
            )

        image_shape = (int(diff_params.height), int(diff_params.width))
        contexts = [
            self._build_think_contexts(
                GEN_THINK_SYSTEM_PROMPT,
                image_prompts[i],
                image_thinking[i],
            )
            for i in range(expected_images)
        ]
        segment, diff_conditions, images = self._diffuse_and_decode(
            contexts,
            prompts=image_prompts,
            params=diff_params,
            sample=partially_filled,
            image_shape=image_shape,
        )
        new_parts[image_idx] = image_part.fill(
            segment=segment,
            primitives={"image": images},
            conditions=diff_conditions.to_dict(),
        )
        return sample.with_parts(new_parts)


class BagelUniPipeline(BagelPipeline):
    """Configuration-compatible name for the Sample-native unified BAGEL path."""

    pass


__all__ = ["BagelPipeline", "BagelUniPipeline"]
