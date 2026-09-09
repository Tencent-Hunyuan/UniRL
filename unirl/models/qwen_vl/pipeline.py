from __future__ import annotations

from typing import Any, Dict, List, Optional

from unirl.models.types.ar import ARSamplingParams
from unirl.models.types.pipeline import Pipeline
from unirl.types.primitives import Texts
from unirl.types.sample import Sample, Turn

from .ar import QwenVLARParams, QwenVLARStage
from .bundle import QwenVLBundle
from .chat_template import QwenVLChatTemplateStage
from .conditions import QwenVLARConditions
from .config import QwenVLPipelineConfig


class QwenVLPipeline(Pipeline):
    """Qwen-VL AR (understanding) generate pipeline: ``Sample → Sample``."""

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
        """Wire chat-template + AR stages around an already-loaded bundle."""
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

    def _conditions_for(
        self,
        turns: List[Turn],
        control: Optional[Dict[str, Any]] = None,
    ) -> QwenVLARConditions:
        """Chat-template + tokenize the trajectory ``turns`` (text + image turns) → :class:`QwenVLARConditions`."""
        chat_overrides: Dict[str, Any] = dict((control or {}).get("chat") or {})
        if "system_instruction" in chat_overrides:
            chat_stage = QwenVLChatTemplateStage(
                self.bundle,
                system_instruction=chat_overrides["system_instruction"],
                max_prompt_length=self.chat_template.max_prompt_length,
            )
        else:
            chat_stage = self.chat_template
        return chat_stage.embed(turns)

    def generate(self, sample: Sample) -> Sample:
        """Run Qwen-VL AR generation end-to-end, filling the frontier (pre-forked) gen Part."""
        frontier = sample.parts[-1]
        ar = frontier.sampling_params
        if not isinstance(ar, ARSamplingParams):
            raise TypeError(
                f"QwenVLPipeline.generate: frontier gen Part must carry ARSamplingParams, "
                f"got {type(ar).__name__ if ar is not None else 'None'}"
            )

        turns, _images = sample.vision_conditioning()
        conds = self._conditions_for(turns, sample.parts[0].control)

        params = QwenVLARParams(
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

        filled = frontier.fill(segment=segment, primitives={"text": decoded}, conditions=conds.to_dict())
        return Sample(parts=[*sample.parts[:-1], filled], reward_compute_s=sample.reward_compute_s)

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
