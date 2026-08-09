"""Qwen3.5 AR pipeline: text (+ optional image) in, text out.

Combines :class:`Qwen3_5ChatTemplateStage` (chat template + image preprocessing)
with :class:`Qwen3_5ARStage` (per-token decode + chunked-logp replay). Mirrors
:class:`unirl.models.qwen_vl.QwenVLPipeline`'s VL data path and
:class:`unirl.models.qwen3.Qwen3Pipeline`'s ``system_instruction`` /
``enable_thinking`` / precision knobs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from unirl.models.types.ar import ARSamplingParams
from unirl.models.types.pipeline import Pipeline
from unirl.types.primitives import Texts
from unirl.types.sample import Sample, Turn

from .ar import Qwen3_5ARParams, Qwen3_5ARStage
from .bundle import Qwen3_5Bundle
from .chat_template import Qwen3_5ChatTemplateStage
from .conditions import Qwen3_5ARConditions
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
        """Wire chat-template + AR stages around an already-loaded bundle.

        The v2 trainer loads the bundle once and injects it
        (``remote_hydra(pipeline_cfg, bundle=...)``); ``from_config`` would
        load a second copy. ``system_instruction`` and ``enable_thinking`` are
        applied to the chat template here so they survive the bundle-injected path.
        """
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

    def _conditions_for(
        self,
        turns: List[Turn],
        control: Optional[Dict[str, Any]] = None,
    ) -> Qwen3_5ARConditions:
        """Render one role-aware text/image trajectory into replay conditions."""
        chat_overrides: Dict[str, Any] = dict((control or {}).get("chat") or {})
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
        return chat_stage.embed(turns)

    def generate(self, sample: Sample) -> Sample:
        """Run Qwen3.5 generation and fill the frontier generation Part."""
        frontier = sample.frontier_gen_part(ARSamplingParams)
        ar = frontier.sampling_params
        assert isinstance(ar, ARSamplingParams)

        if sample.has_image_input():
            turns, _images = sample.vision_conditioning()
        else:
            turns = sample.text_conditioning()
        conds = self._conditions_for(turns, sample.parts[0].control)

        params = Qwen3_5ARParams(
            max_tokens=ar.max_new_tokens,
            temperature=ar.temperature,
            top_p=ar.top_p,
            top_k=ar.top_k,
            stop_token_ids=([int(ar.stop_token_id)] if ar.stop_token_id is not None else []),
        )

        sampling_params = ARSamplingParams(
            samples_per_prompt=int(ar.samples_per_prompt),
            max_new_tokens=int(params.max_tokens),
            temperature=float(params.temperature),
            top_p=float(params.top_p),
            top_k=int(params.top_k),
            stop_token_id=ar.stop_token_id,
        )

        segment = self.ar.autoregress(conds, sampling_params=sampling_params, params=params)
        decoded = self._detokenize(segment)

        return sample.with_filled_frontier(
            segment=segment,
            primitives={"text": decoded},
            conditions=conds.to_dict(),
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


__all__ = ["Qwen3_5Pipeline"]
