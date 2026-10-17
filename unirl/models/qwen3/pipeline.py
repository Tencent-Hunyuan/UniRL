"""Qwen3Pipeline — ``Sample → Sample`` end-to-end for Qwen3."""

from __future__ import annotations

from typing import Any, Dict, Optional

from unirl.models.types.ar import ARSamplingParams, decode_text_segment
from unirl.models.types.pipeline import Pipeline
from unirl.types.sample import Sample

from .ar import Qwen3ARStage
from .bundle import Qwen3Bundle
from .chat_template import Qwen3ChatTemplateStage
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
        replay_kernel_options: Optional[Dict[str, Any]] = None,
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
            replay_kernel_options=replay_kernel_options,
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
        chat_overrides: Dict[str, Any] = dict(sample.parts[0].control.get("chat") or {})
        if "system_instruction" in chat_overrides:
            chat_stage = Qwen3ChatTemplateStage(
                self.bundle,
                system_instruction=chat_overrides["system_instruction"],
                max_prompt_length=self.chat_template.max_prompt_length,
                enable_thinking=self.chat_template.enable_thinking,
            )
        else:
            chat_stage = self.chat_template
        conds = chat_stage.embed(turns)

        segment = self.ar.autoregress(conds, sampling_params=ar)
        decoded = decode_text_segment(segment, self.bundle.tokenizer)

        filled = frontier.fill(segment=segment, primitives={"text": decoded}, conditions=conds.to_dict())
        return Sample(parts=[*sample.parts[:-1], filled], reward_compute_s=sample.reward_compute_s)


__all__ = ["Qwen3Pipeline"]
