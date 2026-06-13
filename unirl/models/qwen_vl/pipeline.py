from __future__ import annotations

from typing import Any, Dict

from unirl.models.types.ar import ARSamplingParams
from unirl.models.types.pipeline import Pipeline
from unirl.types.primitives import Images, Texts
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp, RolloutTrack
from unirl.types.sampling import get_ar_params

from .ar import QwenVLARParams, QwenVLARStage
from .bundle import QwenVLBundle
from .chat_template import QwenVLChatTemplateStage
from .conditions import QwenVLARConditions
from .config import QwenVLPipelineConfig


class QwenVLPipeline(Pipeline):
    def __init__(
        self,
        *,
        bundle: QwenVLBundle,
        chat_template: QwenVLChatTemplateStage,
        ar: QwenVLARStage,
    ) -> None:
        super().__init__()
        self.bundle = bundle
        self.chat_template = chat_template
        self.ar = ar

    @classmethod
    def from_bundle(
        cls,
        bundle: QwenVLBundle,
        *,
        max_prompt_length: int = 4096,
        pad_to_max_length: bool = False,
    ) -> "QwenVLPipeline":
        """Wire chat-template + AR stages around an already-loaded bundle.

        The v2 trainer loads the bundle once and injects it
        (``remote_hydra(pipeline_cfg, bundle=...)``); routing pipeline
        construction through ``from_config`` instead would load a second copy
        of the model. This factory shares the single bundle.

        ``pad_to_max_length`` fixes the prompt sequence length to
        ``max_prompt_length`` so DP rollout shards stay concat-compatible at
        merge time (see :class:`QwenVLChatTemplateStage`).
        """
        chat_template = QwenVLChatTemplateStage(
            bundle,
            max_prompt_length=max_prompt_length,
            pad_to_max_length=pad_to_max_length,
        )
        ar = QwenVLARStage(model=bundle)
        return cls(bundle=bundle, chat_template=chat_template, ar=ar)

    @classmethod
    def from_config(cls, config) -> "QwenVLPipeline":
        if isinstance(config, dict):
            config = QwenVLPipelineConfig(**{k: v for k, v in config.items() if k != "_target_"})
        bundle = QwenVLBundle.from_config(config)
        return cls.from_bundle(bundle, max_prompt_length=config.max_prompt_length)

    def generate(self, req: RolloutReq) -> RolloutResp:
        texts = req.primitives.get("text")
        if not isinstance(texts, Texts):
            raise TypeError(
                f"QwenVLPipeline.generate: req.primitives['text'] must be Texts, "
                f"got {type(texts).__name__ if texts is not None else 'None'}"
            )

        pil_images = None
        images_prim = req.primitives.get("image")
        if images_prim is not None and isinstance(images_prim, Images):
            pil_images = images_prim.to_pils()

        chat_overrides: Dict[str, Any] = dict(req.stage_config.get("chat") or {})
        if "system_instruction" in chat_overrides:
            chat_stage = QwenVLChatTemplateStage(
                self.bundle,
                system_instruction=chat_overrides["system_instruction"],
                max_prompt_length=self.chat_template.max_prompt_length,
            )
        else:
            chat_stage = self.chat_template

        conds: QwenVLARConditions = chat_stage.embed(texts, images=pil_images)

        ar = get_ar_params(req.sampling_params)
        if ar is not None:
            params = QwenVLARParams(
                max_tokens=ar.max_new_tokens,
                temperature=ar.temperature,
                top_p=ar.top_p,
                top_k=ar.top_k,
            )
        else:
            params = QwenVLARParams()

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

    def _detokenize(self, segment) -> Texts:
        if segment.tokens is None or segment.cu_seqlens is None:
            return Texts(texts=[])
        cu = [int(c) for c in segment.cu_seqlens.tolist()]
        tokenizer = self.bundle.tokenizer
        out: list = []
        n = len(cu) - 1
        for i in range(n):
            chunk = segment.tokens[cu[i] : cu[i + 1]]
            ids = chunk.tolist() if chunk.numel() > 0 else []
            out.append(tokenizer.decode(ids, skip_special_tokens=True))
        return Texts(texts=out)


__all__ = ["QwenVLPipeline"]
