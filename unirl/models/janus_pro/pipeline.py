from __future__ import annotations

from typing import Any, Dict, Optional

from unirl.models.types.ar import ARSamplingParams
from unirl.models.types.pipeline import Pipeline
from unirl.types.primitives import Images, Texts
from unirl.types.sample import Sample

from .ar import JanusProARParams, JanusProARStage
from .bundle import JanusProBundle
from .chat_template import JanusProChatTemplateStage
from .conditions import JanusProARConditions, JanusProImageARConditions
from .config import JanusProPipelineConfig
from .image_ar import JanusProImageARSamplingParams, JanusProImageARStage
from .image_prompt import JanusProImagePromptStage


class JanusProPipeline(Pipeline):
    """Janus-Pro generate pipeline: ``Sample -> Sample``.

    Two tasks share one bundle, picked by ``parts[0].control["task"]`` or
    inferred from whether an ancestor Part carries an ``Images`` primitive:

    - ``i2t`` — Text+Image -> Text through the understanding tower. Reads one
      ``Texts`` and one ``Images`` turn off the trajectory and fills the
      frontier Part with a ``TextSegment`` plus ``primitives["text"]``.
    - ``t2i`` — Text -> Image as autoregressive image tokens (Janus-Pro's image
      path is AR, not diffusion). Fills the frontier Part with the image-token
      ``TextSegment`` plus the decoded ``primitives["image"]``.

    Either way ``Part.conditions`` carries the encoded prompt, and trainer-side
    replay teacher-forces over those stored ids, so this encode is the single
    source of truth for the importance ratio.
    """

    def __init__(
        self,
        *,
        bundle: JanusProBundle,
        chat_template: JanusProChatTemplateStage,
        ar: JanusProARStage,
        image_prompt: Optional[JanusProImagePromptStage] = None,
        image_ar: Optional[JanusProImageARStage] = None,
    ) -> None:
        super().__init__()
        self.bundle = bundle
        self.chat_template = chat_template
        self.ar = ar
        self.image_prompt = image_prompt
        self.image_ar = image_ar

    @classmethod
    def from_bundle(
        cls,
        bundle: JanusProBundle,
        *,
        config: JanusProPipelineConfig | Dict[str, Any] | None = None,
        max_prompt_length: int | None = None,
    ) -> "JanusProPipeline":
        if isinstance(config, dict):
            config = JanusProPipelineConfig(**{k: v for k, v in config.items() if k != "_target_"})

        if config is None:
            chat_template = JanusProChatTemplateStage(
                bundle,
                max_prompt_length=4096 if max_prompt_length is None else int(max_prompt_length),
            )
            ar = JanusProARStage(model=bundle)
            image_prompt = JanusProImagePromptStage(
                bundle,
                max_prompt_length=4096 if max_prompt_length is None else int(max_prompt_length),
            )
            image_ar = JanusProImageARStage(model=bundle)
            return cls(
                bundle=bundle,
                chat_template=chat_template,
                ar=ar,
                image_prompt=image_prompt,
                image_ar=image_ar,
            )

        chat_template = JanusProChatTemplateStage(
            bundle,
            user_role=config.user_role,
            assistant_role=config.assistant_role,
            image_placeholder=config.image_placeholder,
            max_prompt_length=config.max_prompt_length if max_prompt_length is None else int(max_prompt_length),
        )
        ar = JanusProARStage(
            model=bundle,
            autocast_precision=config.autocast_precision,
            logprob_precision=config.logprob_precision,
        )
        image_prompt = JanusProImagePromptStage(
            bundle,
            user_role=config.user_role,
            assistant_role=config.assistant_role,
            max_prompt_length=config.max_prompt_length if max_prompt_length is None else int(max_prompt_length),
        )
        image_ar = JanusProImageARStage(
            model=bundle,
            autocast_precision=config.autocast_precision,
            logprob_precision=config.logprob_precision,
        )
        return cls(
            bundle=bundle,
            chat_template=chat_template,
            ar=ar,
            image_prompt=image_prompt,
            image_ar=image_ar,
        )

    @classmethod
    def from_config(cls, config) -> "JanusProPipeline":
        if isinstance(config, dict):
            config = JanusProPipelineConfig(**{k: v for k, v in config.items() if k != "_target_"})
        bundle = JanusProBundle.from_config(config)
        return cls.from_bundle(bundle, config=config)

    @staticmethod
    def _resolve_task(sample: Sample) -> str:
        """Explicit ``parts[0].control["task"]`` wins, else infer from the inputs.

        Janus-Pro's two paths are told apart by the presence of an input image:
        understanding consumes one, AR image generation does not.
        """
        task = (sample.parts[0].control or {}).get("task")
        if task is None:
            return "i2t" if sample.has_image_input() else "t2i"
        return str(task).strip().lower()

    def generate(self, sample: Sample) -> Sample:
        task = self._resolve_task(sample)
        if task in {"i2t", "it2t", "understanding", "text"}:
            return self._generate_i2t(sample)
        if task in {"t2i", "text2image", "image", "generation"}:
            return self._generate_t2i(sample)
        raise ValueError(f"JanusProPipeline.generate: unsupported task={task!r}")

    @staticmethod
    def _single(turns, kind, task: str):
        """The one primitive of ``kind`` on the trajectory.

        Janus-Pro's chat template renders exactly one user turn, so a multi-turn
        trajectory has no faithful encoding here — fail rather than silently
        dropping turns.
        """
        found = [t.content for t in turns if isinstance(t.content, kind)]
        if len(found) != 1:
            raise ValueError(
                f"JanusProPipeline.{task}: expected exactly one {kind.__name__} turn, got {len(found)}. "
                "The Janus-Pro chat template renders a single user turn and cannot encode a "
                "multi-turn trajectory."
            )
        return found[0]

    def _generate_i2t(self, sample: Sample) -> Sample:
        frontier = sample.frontier_gen_part(ARSamplingParams)

        # Fails loud on zero images or a non-text/image modality.
        turns, _images = sample.vision_conditioning()
        texts = self._single(turns, Texts, "i2t")
        images_prim = self._single(turns, Images, "i2t")

        chat_overrides: Dict[str, Any] = dict((sample.parts[0].control or {}).get("chat") or {})
        if "system_instruction" in chat_overrides:
            chat_stage = JanusProChatTemplateStage(
                self.bundle,
                user_role=self.chat_template.user_role,
                assistant_role=self.chat_template.assistant_role,
                image_placeholder=self.chat_template.image_placeholder,
                system_instruction=chat_overrides["system_instruction"],
                max_prompt_length=self.chat_template.max_prompt_length,
            )
        else:
            chat_stage = self.chat_template

        conds: JanusProARConditions = chat_stage.embed(texts, images=images_prim.to_pils())

        # Normalize the gen shell's params through JanusProARParams (stop_token_id
        # reset, types coerced), mirroring qwen_vl.
        ar = frontier.sampling_params
        params = JanusProARParams(
            max_tokens=ar.max_new_tokens,
            temperature=ar.temperature,
            top_p=ar.top_p,
            top_k=ar.top_k,
        )
        sampling_params = ARSamplingParams(
            max_new_tokens=int(params.max_tokens),
            temperature=float(params.temperature),
            top_p=float(params.top_p),
            top_k=int(params.top_k),
            stop_token_id=None,
        )

        segment = self.ar.autoregress(conds, sampling_params=sampling_params, params=params)
        decoded = self._detokenize(segment)
        return sample.with_filled_frontier(
            segment=segment,
            primitives={"text": decoded},
            conditions=conds.to_dict(),
        )

    def _generate_t2i(self, sample: Sample) -> Sample:
        if self.image_prompt is None or self.image_ar is None:
            raise RuntimeError("JanusProPipeline Text -> Image requires image_prompt and image_ar stages.")

        # The image grid, CFG weight, and token count all ride on the gen shell,
        # so the params type is part of the contract rather than a soft default.
        frontier = sample.frontier_gen_part(JanusProImageARSamplingParams)
        sampling_params: JanusProImageARSamplingParams = frontier.sampling_params

        texts = self._single(sample.turns(), Texts, "t2i")
        conds: JanusProImageARConditions = self.image_prompt.embed(
            texts,
            cfg_weight=float(sampling_params.cfg_weight),
        )
        segment = self.image_ar.autoregress(conds, sampling_params=sampling_params)
        decoded = self.image_ar.decode(segment, sampling_params=sampling_params)
        return sample.with_filled_frontier(
            segment=segment,
            primitives={"image": decoded},
            conditions=conds.to_dict(),
        )

    def _detokenize(self, segment) -> Texts:
        if segment.tokens is None or segment.cu_seqlens is None:
            return Texts(texts=[])
        cu = [int(c) for c in segment.cu_seqlens.tolist()]
        out = []
        for i in range(len(cu) - 1):
            chunk = segment.tokens[cu[i] : cu[i + 1]]
            ids = chunk.tolist() if chunk.numel() > 0 else []
            out.append(_clean_decoded_text(self.bundle.tokenizer.decode(ids, skip_special_tokens=True)))
        return Texts(texts=out)


def _clean_decoded_text(text: str) -> str:
    return text.replace("Ġ", " ").replace("▁", " ").strip()


__all__ = ["JanusProPipeline"]
