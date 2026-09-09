"""Qwen3Pipeline — ``Sample → Sample`` end-to-end for Qwen3."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from unirl.models.types.ar import ARSamplingParams
from unirl.models.types.pipeline import Pipeline
from unirl.types.primitives import Texts
from unirl.types.sample import Sample, Turn

from .ar import Qwen3ARParams, Qwen3ARStage
from .bundle import Qwen3Bundle
from .chat_template import Qwen3ChatTemplateStage
from .conditions import Qwen3ARConditions
from .config import Qwen3PipelineConfig


class Qwen3Pipeline(Pipeline):
    """Qwen3 AR generate pipeline: ``Sample → Sample``."""

    def __init__(
        self,
        *,
        bundle: Qwen3Bundle,
        chat_template: Optional[Qwen3ChatTemplateStage] = None,
        ar: Optional[Qwen3ARStage] = None,
        autocast_precision: str = "bf16",
        logprob_precision: str = "fp32",
    ) -> None:
        super().__init__()
        self.bundle = bundle
        self.chat_template = chat_template if chat_template is not None else Qwen3ChatTemplateStage(bundle)
        self.ar = (
            ar
            if ar is not None
            else Qwen3ARStage(model=bundle, autocast_precision=autocast_precision, logprob_precision=logprob_precision)
        )

    @classmethod
    def from_bundle(
        cls,
        bundle: Qwen3Bundle,
        *,
        system_instruction: Optional[str] = None,
        autocast_precision: str = "bf16",
        logprob_precision: str = "fp32",
        enable_thinking: bool = False,
        max_prompt_length: int = 4096,
    ) -> "Qwen3Pipeline":
        """Wire chat-template + AR stages around an already-loaded bundle."""
        chat_template = Qwen3ChatTemplateStage(
            bundle,
            system_instruction=system_instruction,
            enable_thinking=enable_thinking,
            max_prompt_length=max_prompt_length,
        )
        ar = Qwen3ARStage(
            model=bundle,
            autocast_precision=autocast_precision,
            logprob_precision=logprob_precision,
        )
        return cls(
            bundle=bundle,
            chat_template=chat_template,
            ar=ar,
            autocast_precision=autocast_precision,
            logprob_precision=logprob_precision,
        )

    @classmethod
    def from_config(cls, config: Qwen3PipelineConfig) -> "Qwen3Pipeline":
        """Build the full pipeline from a config."""
        bundle = Qwen3Bundle.from_config(config)
        chat_template = Qwen3ChatTemplateStage(
            bundle,
            system_instruction=config.system_instruction,
            enable_thinking=config.enable_thinking,
            max_prompt_length=config.max_prompt_length,
        )
        ar = Qwen3ARStage(
            model=bundle,
            autocast_precision=config.autocast_precision,
            logprob_precision=config.logprob_precision,
        )
        return cls(bundle=bundle, chat_template=chat_template, ar=ar)

    def _conditions_for(self, turns: List[Turn], control: Optional[Dict[str, Any]] = None) -> Qwen3ARConditions:
        """Chat-template + tokenize the trajectory ``turns`` → :class:`Qwen3ARConditions`."""
        chat_overrides: Dict[str, Any] = dict((control or {}).get("chat") or {})
        if "system_instruction" in chat_overrides:
            chat_stage = Qwen3ChatTemplateStage(
                self.bundle,
                system_instruction=chat_overrides["system_instruction"],
                max_prompt_length=self.chat_template.max_prompt_length,
                enable_thinking=self.chat_template.enable_thinking,
            )
        else:
            chat_stage = self.chat_template
        return chat_stage.embed(turns)

    def generate(self, sample: Sample) -> Sample:
        """Run Qwen3 AR generation end-to-end, filling the frontier (pre-forked) gen Part."""
        frontier = sample.parts[-1]
        ar = frontier.sampling_params
        if not isinstance(ar, ARSamplingParams):
            raise TypeError(
                f"Qwen3Pipeline.generate: frontier gen Part must carry ARSamplingParams, "
                f"got {type(ar).__name__ if ar is not None else 'None'}"
            )

        turns = sample.text_conditioning()
        conds = self._conditions_for(turns, sample.parts[0].control)

        params = Qwen3ARParams(
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
        """Decode each per-sample varlen token chunk via the bundle tokenizer."""
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


__all__ = ["Qwen3Pipeline"]
