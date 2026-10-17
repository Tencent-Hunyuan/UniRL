"""Qwen3.5 AR pipeline: text (+ optional image) in, text out."""

from __future__ import annotations

from typing import Any, Dict, Optional

from unirl.models.types.ar import ARSamplingParams, decode_text_segment
from unirl.models.types.pipeline import Pipeline
from unirl.types.sample import Sample

from .ar import Qwen3_5ARStage
from .bundle import Qwen3_5Bundle
from .chat_template import Qwen3_5ChatTemplateStage
from .config import Qwen3_5PipelineConfig


class Qwen3_5Pipeline(Pipeline):
    """Qwen3.5 AR pipeline for dense/MoE text and single-image inputs."""

    def __init__(
        self,
        *,
        bundle: Qwen3_5Bundle,
        chat_template: Optional[Qwen3_5ChatTemplateStage] = None,
        ar: Optional[Qwen3_5ARStage] = None,
        autocast_precision: str = "bf16",
        logprob_precision: str = "fp32",
    ) -> None:
        super().__init__()
        self.bundle = bundle
        self.chat_template = chat_template if chat_template is not None else Qwen3_5ChatTemplateStage(bundle)
        self.ar = (
            ar
            if ar is not None
            else Qwen3_5ARStage(
                model=bundle,
                autocast_precision=autocast_precision,
                logprob_precision=logprob_precision,
            )
        )

    @classmethod
    def from_bundle(
        cls,
        bundle: Qwen3_5Bundle,
        *,
        system_instruction: Optional[str] = None,
        max_prompt_length: int = 4096,
        enable_thinking: bool = False,
        pad_to_max_length: bool = False,
        autocast_precision: str = "bf16",
        logprob_precision: str = "fp32",
    ) -> "Qwen3_5Pipeline":
        """Wire chat-template + AR stages around an already-loaded bundle."""
        chat_template = Qwen3_5ChatTemplateStage(
            bundle,
            system_instruction=system_instruction,
            max_prompt_length=max_prompt_length,
            enable_thinking=enable_thinking,
            pad_to_max_length=pad_to_max_length,
        )
        ar = Qwen3_5ARStage(
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
    def from_config(cls, config: Qwen3_5PipelineConfig) -> "Qwen3_5Pipeline":
        """Build the full pipeline from a config."""
        if isinstance(config, dict):
            config = Qwen3_5PipelineConfig(**{k: v for k, v in config.items() if k != "_target_"})
        bundle = Qwen3_5Bundle.from_config(config)
        chat_template = Qwen3_5ChatTemplateStage(
            bundle,
            system_instruction=config.system_instruction,
            max_prompt_length=config.max_prompt_length,
            enable_thinking=config.enable_thinking,
        )
        ar = Qwen3_5ARStage(
            model=bundle,
            autocast_precision=config.autocast_precision,
            logprob_precision=config.logprob_precision,
        )
        return cls(
            bundle=bundle,
            chat_template=chat_template,
            ar=ar,
            autocast_precision=config.autocast_precision,
            logprob_precision=config.logprob_precision,
        )

    def generate(self, sample: Sample) -> Sample:
        """Run Qwen3.5 generation and fill the frontier generation Part."""
        frontier = sample.frontier_gen_part(ARSamplingParams)
        ar = frontier.sampling_params
        assert isinstance(ar, ARSamplingParams)

        if sample.has_image_input():
            turns, _images = sample.vision_conditioning()
        else:
            turns = sample.text_conditioning()
        chat_overrides: Dict[str, Any] = dict(sample.parts[0].control.get("chat") or {})
        if "system_instruction" in chat_overrides:
            chat_stage = Qwen3_5ChatTemplateStage(
                self.bundle,
                system_instruction=chat_overrides["system_instruction"],
                max_prompt_length=self.chat_template.max_prompt_length,
                enable_thinking=self.chat_template.enable_thinking,
                pad_to_max_length=self.chat_template.pad_to_max_length,
            )
        else:
            chat_stage = self.chat_template
        conds = chat_stage.embed(turns)

        segment = self.ar.autoregress(conds, sampling_params=ar)
        decoded = decode_text_segment(segment, self.bundle.tokenizer)

        return sample.with_filled_frontier(
            segment=segment,
            primitives={"text": decoded},
            conditions=conds.to_dict(),
        )


__all__ = ["Qwen3_5Pipeline"]
