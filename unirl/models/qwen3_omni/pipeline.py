"""Typed rollout pipeline for the Qwen3-Omni thinker."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from unirl.models.types.ar import ARSamplingParams
from unirl.models.types.pipeline import Pipeline
from unirl.types.primitives import Texts, Videos
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp, RolloutTrack

from .ar import Qwen3OmniARParams, Qwen3OmniARStage
from .bundle import Qwen3OmniBundle
from .chat_template import Qwen3OmniChatTemplateStage
from .conditions import Qwen3OmniARConditions
from .config import Qwen3OmniPipelineConfig


class Qwen3OmniPipeline(Pipeline):
    """Generate the ``ar`` rollout track from text and optional video inputs."""

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
        video_fps: float = 1.0,
        video_max_frames: Optional[int] = None,
        video_max_pixels: Optional[int] = None,
        autocast_precision: str = "bf16",
        logprob_precision: str = "fp32",
    ) -> "Qwen3OmniPipeline":
        """Build stages around a shared, potentially FSDP-wrapped bundle."""
        chat_template = Qwen3OmniChatTemplateStage(
            bundle,
            system_instruction=system_instruction,
            max_prompt_length=max_prompt_length,
            video_fps=video_fps,
            video_max_frames=video_max_frames,
            video_max_pixels=video_max_pixels,
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
            video_fps=config.video_fps,
            video_max_frames=config.video_max_frames,
            video_max_pixels=config.video_max_pixels,
        )
        ar = Qwen3OmniARStage(
            model=bundle,
            autocast_precision=config.autocast_precision,
            logprob_precision=config.logprob_precision,
        )
        return cls(bundle=bundle, chat_template=chat_template, ar=ar)

    def generate(self, req: RolloutReq) -> RolloutResp:
        texts = req.primitives.get("text")
        if not isinstance(texts, Texts):
            raise TypeError(
                f"Qwen3OmniPipeline.generate: req.primitives['text'] must be Texts, "
                f"got {type(texts).__name__ if texts is not None else 'None'}"
            )

        # Optional per-sample videos.
        videos_prim = req.primitives.get("video")
        per_sample_videos: Optional[List[Any]] = None
        if isinstance(videos_prim, Videos):
            per_sample_videos = self._videos_to_list(videos_prim)

        # Optional per-request system instruction.
        chat_overrides: Dict[str, Any] = dict(req.task_config.get("chat") or {})
        if "system_instruction" in chat_overrides:
            chat_stage = Qwen3OmniChatTemplateStage(
                self.bundle,
                system_instruction=chat_overrides["system_instruction"],
                max_prompt_length=self.chat_template.max_prompt_length,
                video_fps=self.chat_template.video_fps,
                video_max_frames=self.chat_template.video_max_frames,
                video_max_pixels=self.chat_template.video_max_pixels,
            )
        else:
            chat_stage = self.chat_template

        conds: Qwen3OmniARConditions = chat_stage.embed(texts, videos=per_sample_videos)

        ar = req.sampling_params.get("ar")
        if ar is not None:
            params = Qwen3OmniARParams(
                max_tokens=ar.max_new_tokens,
                temperature=ar.temperature,
                top_p=ar.top_p,
                top_k=ar.top_k,
            )
        else:
            params = Qwen3OmniARParams()

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

    @staticmethod
    def _videos_to_list(videos: Videos) -> List[Any]:
        """Return per-sample paths or decoded frames for TMRoPE preprocessing."""
        uris = getattr(videos, "uris", None)
        if uris:
            return list(uris)
        if videos.frames is not None and videos.cu_frames is not None:
            return [video.frames for video in videos.to_list()]
        for attr in ("to_list", "frames", "videos"):
            v = getattr(videos, attr, None)
            if callable(v):
                return list(v())
            if v is not None:
                return list(v)
        raise TypeError("Qwen3OmniPipeline: could not extract per-sample videos from the Videos primitive")

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
