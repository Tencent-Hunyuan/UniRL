"""Typed rollout pipeline for the Qwen3-Omni thinker."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from unirl.models.types.pipeline import Pipeline
from unirl.types.primitives import Texts
from unirl.types.sample import Sample, Turn
from unirl.types.sampling import ARSamplingParams

from .ar import Qwen3OmniARParams, Qwen3OmniARStage
from .bundle import Qwen3OmniBundle
from .chat_template import Qwen3OmniChatTemplateStage
from .conditions import Qwen3OmniARConditions
from .config import Qwen3OmniPipelineConfig
from .media import build_omni_messages


class Qwen3OmniPipeline(Pipeline):
    """Qwen3-Omni thinker generation pipeline: ``Sample → Sample``."""

    # URI-backed MediaRefs are an Omni prompt-input channel; the AR trainers
    # read this declaration (unirl.trainer.ar.ar_preflight) instead of
    # special-casing the model.
    extra_input_primitives = ("media",)

    def __init__(
        self,
        *,
        bundle: Qwen3OmniBundle,
        chat_template: Optional[Qwen3OmniChatTemplateStage] = None,
        ar: Optional[Qwen3OmniARStage] = None,
        autocast_precision: str = "bf16",
        logprob_precision: str = "fp32",
    ) -> None:
        super().__init__()
        self.bundle = bundle
        self.chat_template = chat_template if chat_template is not None else Qwen3OmniChatTemplateStage(bundle)
        self.ar = (
            ar
            if ar is not None
            else Qwen3OmniARStage(
                model=bundle, autocast_precision=autocast_precision, logprob_precision=logprob_precision
            )
        )

    @classmethod
    def from_bundle(
        cls,
        bundle: Qwen3OmniBundle,
        *,
        system_instruction: Optional[str] = None,
        max_prompt_length: int = 4096,
        image_max_pixels: Optional[int] = None,
        video_fps: float = 1.0,
        video_max_frames: Optional[int] = None,
        video_max_pixels: Optional[int] = None,
        use_audio_in_video: bool = False,
        autocast_precision: str = "bf16",
        logprob_precision: str = "fp32",
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "Qwen3OmniPipeline":
        """Build stages around a shared, potentially FSDP-wrapped bundle."""
        chat_template = Qwen3OmniChatTemplateStage(
            bundle,
            system_instruction=system_instruction,
            max_prompt_length=max_prompt_length,
            image_max_pixels=image_max_pixels,
            video_fps=video_fps,
            video_max_frames=video_max_frames,
            video_max_pixels=video_max_pixels,
            use_audio_in_video=use_audio_in_video,
            chat_template_kwargs=chat_template_kwargs,
        )
        ar = Qwen3OmniARStage(model=bundle, autocast_precision=autocast_precision, logprob_precision=logprob_precision)
        return cls(
            bundle=bundle,
            chat_template=chat_template,
            ar=ar,
            autocast_precision=autocast_precision,
            logprob_precision=logprob_precision,
        )

    @classmethod
    def from_config(cls, config: Qwen3OmniPipelineConfig) -> "Qwen3OmniPipeline":
        bundle = Qwen3OmniBundle.from_config(config)
        chat_template = Qwen3OmniChatTemplateStage(
            bundle,
            system_instruction=config.system_instruction,
            max_prompt_length=config.max_prompt_length,
            image_max_pixels=config.image_max_pixels,
            video_fps=config.video_fps,
            video_max_frames=config.video_max_frames,
            video_max_pixels=config.video_max_pixels,
            use_audio_in_video=config.use_audio_in_video,
            chat_template_kwargs=config.chat_template_kwargs,
        )
        ar = Qwen3OmniARStage(
            model=bundle,
            autocast_precision=config.autocast_precision,
            logprob_precision=config.logprob_precision,
        )
        return cls(bundle=bundle, chat_template=chat_template, ar=ar)

    def _conditions_for(
        self,
        turns: List[Turn],
        control: Optional[Dict[str, Any]] = None,
    ) -> Qwen3OmniARConditions:
        """Render the trajectory using config plus root-Part chat overrides."""
        chat_overrides: Dict[str, Any] = dict((control or {}).get("chat") or {})
        system_instruction = chat_overrides.get(
            "system_instruction",
            self.chat_template.system_instruction,
        )
        template_kwargs = dict(self.chat_template.chat_template_kwargs)
        template_kwargs.update(dict(chat_overrides.get("template_kwargs") or {}))
        if (
            system_instruction != self.chat_template.system_instruction
            or template_kwargs != self.chat_template.chat_template_kwargs
        ):
            chat_stage = Qwen3OmniChatTemplateStage(
                self.bundle,
                system_instruction=system_instruction,
                max_prompt_length=self.chat_template.max_prompt_length,
                pad_to_max_length=self.chat_template.pad_to_max_length,
                image_max_pixels=self.chat_template.image_max_pixels,
                video_fps=self.chat_template.video_fps,
                video_max_frames=self.chat_template.video_max_frames,
                video_max_pixels=self.chat_template.video_max_pixels,
                use_audio_in_video=self.chat_template.use_audio_in_video,
                chat_template_kwargs=template_kwargs,
            )
        else:
            chat_stage = self.chat_template
        conversations = build_omni_messages(
            turns,
            system_instruction,
        )
        return chat_stage.embed_messages(conversations)

    def generate(self, sample: Sample) -> Sample:
        """Generate one Qwen3-Omni assistant turn and fill the AR frontier."""
        frontier = sample.frontier_gen_part(ARSamplingParams)
        ar = frontier.sampling_params
        assert isinstance(ar, ARSamplingParams)

        turns = sample.turns()
        conds = self._conditions_for(turns, sample.parts[0].control)

        params = Qwen3OmniARParams(
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
        return sample.replace_frontier(
            frontier.fill(
                segment=segment,
                primitives={"text": decoded},
                conditions=conds.to_dict(),
            )
        )

    def _detokenize(self, segment) -> Texts:
        if segment.tokens is None or segment.cu_seqlens is None:
            return Texts(texts=[])
        cu = [int(c) for c in segment.cu_seqlens.tolist()]
        tokenizer = self.bundle.tokenizer
        out: list = []
        for i in range(len(cu) - 1):
            chunk = segment.tokens[cu[i] : cu[i + 1]]
            ids = chunk.tolist() if chunk.numel() > 0 else []
            out.append(tokenizer.decode(ids, skip_special_tokens=True))
        return Texts(texts=out)


__all__ = ["Qwen3OmniPipeline"]
