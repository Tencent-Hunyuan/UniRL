from __future__ import annotations

from typing import Any, Dict, Optional

from unirl.models.types.ar import ARSamplingParams
from unirl.models.types.pipeline import Pipeline
from unirl.types.primitives import Images, Texts
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp, RolloutTrack

from .ar import JanusProARParams, JanusProARStage
from .bundle import JanusProBundle
from .chat_template import JanusProChatTemplateStage
from .conditions import JanusProARConditions, JanusProImageARConditions
from .config import JanusProPipelineConfig
from .image_ar import JanusProImageARSamplingParams, JanusProImageARStage
from .image_prompt import JanusProImagePromptStage


class JanusProPipeline(Pipeline):
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

    def generate(self, req: RolloutReq) -> RolloutResp:
        task = str(req.task_config.get("task") or "").strip().lower()
        if not task:
            task = "i2t" if isinstance(req.primitives.get("image"), Images) else "t2i"
        if task in {"i2t", "it2t", "understanding", "text"}:
            return self._generate_i2t(req)
        if task in {"t2i", "text2image", "image", "generation"}:
            return self._generate_t2i(req)
        raise ValueError(f"JanusProPipeline.generate: unsupported task={task!r}")

    def _generate_i2t(self, req: RolloutReq) -> RolloutResp:
        texts = req.primitives.get("text")
        if not isinstance(texts, Texts):
            raise TypeError(
                f"JanusProPipeline.generate: req.primitives['text'] must be Texts, "
                f"got {type(texts).__name__ if texts is not None else 'None'}"
            )

        images_prim = req.primitives.get("image")
        if not isinstance(images_prim, Images):
            raise TypeError(
                "JanusProPipeline.generate implements Text+Image -> Text and requires "
                f"req.primitives['image'] to be Images, got "
                f"{type(images_prim).__name__ if images_prim is not None else 'None'}"
            )

        chat_overrides: Dict[str, Any] = dict(req.task_config.get("chat") or {})
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

        ar = req.sampling_params.get("ar")
        if ar is not None:
            params = JanusProARParams(
                max_tokens=ar.max_new_tokens,
                temperature=ar.temperature,
                top_p=ar.top_p,
                top_k=ar.top_k,
            )
        else:
            params = JanusProARParams()

        sampling_params = ARSamplingParams(
            max_new_tokens=int(params.max_tokens),
            temperature=float(params.temperature),
            top_p=float(params.top_p),
            top_k=int(params.top_k),
            stop_token_id=None,
        )

        segment = self.ar.autoregress(conds, sampling_params=sampling_params, params=params)
        decoded = self._detokenize(segment)

        return RolloutResp(
            tracks={
                "ar": RolloutTrack(
                    sample_ids=list(req.sample_ids),
                    parent_ids=list(req.group_ids),
                    conditions=conds.to_dict(),
                    segment=segment,
                    decoded=decoded,
                ),
            }
        )

    def _generate_t2i(self, req: RolloutReq) -> RolloutResp:
        if self.image_prompt is None or self.image_ar is None:
            raise RuntimeError("JanusProPipeline Text -> Image requires image_prompt and image_ar stages.")

        texts = req.primitives.get("text")
        if not isinstance(texts, Texts):
            raise TypeError(
                f"JanusProPipeline.generate: req.primitives['text'] must be Texts, "
                f"got {type(texts).__name__ if texts is not None else 'None'}"
            )

        ar = req.sampling_params.get("ar")
        sampling_params = ar if ar is not None else JanusProImageARSamplingParams()
        cfg_weight = float(getattr(sampling_params, "cfg_weight", 5.0))

        conds: JanusProImageARConditions = self.image_prompt.embed(texts, cfg_weight=cfg_weight)
        segment = self.image_ar.autoregress(conds, sampling_params=sampling_params)
        decoded = self.image_ar.decode(segment, sampling_params=sampling_params)

        return RolloutResp(
            tracks={
                "image": RolloutTrack(
                    sample_ids=list(req.sample_ids),
                    parent_ids=list(req.group_ids),
                    conditions=conds.to_dict(),
                    segment=segment,
                    decoded=decoded,
                ),
            }
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
