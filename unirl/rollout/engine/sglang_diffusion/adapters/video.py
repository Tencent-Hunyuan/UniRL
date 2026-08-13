"""Video-family adapters — 6-D ``[B, T+1, C, F, H, W]`` trajectory decoded to ragged ``[total_T, C, H, W]``."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from unirl.config.require import require
from unirl.rollout.engine.sglang_diffusion import utils
from unirl.rollout.engine.sglang_diffusion.adapters.base import register_adapter
from unirl.rollout.engine.sglang_diffusion.adapters.image import ImageAdapter
from unirl.rollout.engine.sglang_diffusion.backends import RawResult
from unirl.types.conditions.text import TextEmbedCondition
from unirl.types.sample import Sample
from unirl.types.sampling import DiffusionSamplingParams
from unirl.types.segments.latent import make_video_segment


class VideoAdapter(ImageAdapter):
    """Base for true video families — ``Modality.VIDEO`` and a 6-D ``[B, T+1, C, F, H, W]`` trajectory."""

    segment_factory = staticmethod(make_video_segment)

    def build_segment(
        self,
        sample: Sample,
        results: List[RawResult],
        *,
        num_steps: int,
        sde_indices: Optional[List[int]],
        emit_native_logprob: bool,
    ):
        """Video-form trajectory: collect, gate the 6-D shape, assemble."""
        traj = utils.collect_trajectory_latents(results)
        if traj.ndim != 6:
            raise ValueError(
                f"{self.model_family}: expected a 6-D video-form trajectory "
                f"[B, T+1, C, F, H, W]; got rank {traj.ndim}, shape {tuple(traj.shape)}."
            )
        return utils.build_latent_segment(
            traj,
            results=results,
            expected_sigmas=sample.frontier_gen_part(DiffusionSamplingParams).sampling_params.sigmas,
            num_steps=num_steps,
            sde_indices=sde_indices,
            emit_native_logprob=emit_native_logprob,
            segment_factory=self.segment_factory,
        )

    def build_decoded(self, sample: Sample, results: List[RawResult]):
        del sample
        return utils.stack_decoded_videos(results)


@register_adapter("mochi")
class MochiAdapter(ImageAdapter):
    """Mochi — image-path parity (see module note); migrate to VideoAdapter when it has a video reward baseline."""

    squeeze_single_frame_4d = False


@register_adapter("hunyuan_video")
class HunyuanVideoAdapter(VideoAdapter):
    """HunyuanVideo-1.0 T2V with video output and dual text conditions."""

    def build_condition(self, results: List[RawResult]) -> Dict[str, Any]:
        """Keep HunyuanVideo's LLaMA and pooled-CLIP streams separate."""
        require(bool(results), "HunyuanVideo: cannot build conditions from empty results")

        llama_conditions: List[TextEmbedCondition] = []
        clip_list: List[torch.Tensor] = []
        mask_presence: List[bool] = []
        for result in results:
            prompt_embeds = result.prompt_embeds
            require(
                isinstance(prompt_embeds, (list, tuple)) and len(prompt_embeds) >= 2,
                "HunyuanVideo: expected prompt_embeds=[LLaMA, CLIP-pooled]; got "
                f"{type(prompt_embeds).__name__} with "
                f"{len(prompt_embeds) if isinstance(prompt_embeds, (list, tuple)) else 'n/a'} entries",
            )
            llama, clip = prompt_embeds[:2]
            require(
                torch.is_tensor(llama) and llama.ndim == 3,
                "HunyuanVideo: LLaMA prompt embed must be [B, seq, hidden]",
            )
            require(
                torch.is_tensor(clip) and clip.ndim in (2, 3),
                "HunyuanVideo: pooled CLIP embed must be [B, hidden] or [B, 1, hidden]",
            )
            require(
                int(llama.shape[0]) == int(clip.shape[0]),
                "HunyuanVideo: LLaMA and CLIP prompt embed batch sizes must match",
            )

            attention_mask = None
            encoder_masks = result.encoder_attention_mask
            if isinstance(encoder_masks, (list, tuple)) and encoder_masks and encoder_masks[0] is not None:
                attention_mask = encoder_masks[0]
                require(
                    torch.is_tensor(attention_mask)
                    and attention_mask.ndim == 2
                    and tuple(attention_mask.shape) == tuple(llama.shape[:2]),
                    "HunyuanVideo: LLaMA attention mask must match [B, seq]",
                )

            mask_presence.append(attention_mask is not None)
            llama_conditions.append(
                TextEmbedCondition(
                    embeds=llama.detach().cpu(),
                    attn_mask=attention_mask.detach().cpu() if attention_mask is not None else None,
                )
            )
            clip_list.append(clip.detach().cpu().reshape(int(clip.shape[0]), -1))

        require(
            all(mask_presence) or not any(mask_presence),
            "HunyuanVideo: LLaMA attention masks must be present for every result or none",
        )
        return {
            "text_llama": TextEmbedCondition.concat(llama_conditions),
            "pooled_clip": TextEmbedCondition(embeds=torch.cat(clip_list, dim=0)),
        }


@register_adapter("wan22")
class Wan22T2VAdapter(VideoAdapter):
    """WAN 2.2-A14B T2V — DUAL-EXPERT (high-noise / low-noise) MoE."""

    def build_sampling(self, sample: Sample, *, diffusion: Any) -> Dict[str, Any]:
        kwargs = super().build_sampling(sample, diffusion=diffusion)
        g2 = getattr(diffusion, "guidance_scale_2", None)
        if g2 is not None:
            kwargs["guidance_scale_2"] = float(g2)
        return kwargs


@register_adapter("wan21")
class Wan21T2VAdapter(VideoAdapter):
    """WAN 2.1 T2V — proper video output consumed by ``video_pickscore``."""

    pass


@register_adapter("ltx2")
class Ltx2T2VAdapter(VideoAdapter):
    """LTX-2 T2V — ~19B AV DiT with packed video and audio trajectories."""

    def schedule_policy(self):
        from unirl.models.ltx2.schedule import build_ltx2_schedule_policy

        return build_ltx2_schedule_policy(float(self.model_config.shift))

    def build_sampling(self, sample: Sample, *, diffusion: Any) -> Dict[str, Any]:
        kwargs = super().build_sampling(sample, diffusion=diffusion)
        kwargs["max_sequence_length"] = int(self.model_config.max_sequence_length)

        from unirl.models.ltx2.diffusion import audio_latent_shape
        from unirl.types.noise_recipe import NoiseRecipe

        audio_noise = NoiseRecipe.from_sample(sample).resolve(
            salt="audio",
            latent_shape=audio_latent_shape(diffusion),
        )
        if audio_noise is not None:
            kwargs["initial_audio_noise"] = audio_noise
        return kwargs

    @staticmethod
    def _fuse_audio_condition(results: List[RawResult], field: str) -> Optional[TextEmbedCondition]:
        tensors = []
        for result in results:
            value = utils.fuse_encoder_outputs(getattr(result, field, None))
            if value is not None:
                tensors.append(value.detach().cpu())
        if not tensors:
            return None
        require(
            len(tensors) == len(results),
            f"LTX-2: {field} must be present for every result or none",
        )
        return TextEmbedCondition(embeds=torch.cat(tensors, dim=0))

    def build_condition(self, results: List[RawResult]) -> Dict[str, Any]:
        out = super().build_condition(results)
        text = out.get("text")
        negative_text = out.get("negative_text")

        if text is not None:
            text = TextEmbedCondition(embeds=text.embeds, pooled=text.pooled)
            out["text"] = text
        if negative_text is not None:
            negative_text = TextEmbedCondition(embeds=negative_text.embeds, pooled=negative_text.pooled)
            out["negative_text"] = negative_text

        audio_text = self._fuse_audio_condition(results, "audio_prompt_embeds")
        negative_audio_text = self._fuse_audio_condition(results, "negative_audio_prompt_embeds")
        if audio_text is not None:
            out["audio_text"] = audio_text
        if negative_audio_text is not None:
            out["negative_audio_text"] = negative_audio_text
        return out

    def build_segment(
        self,
        sample: Sample,
        results: List[RawResult],
        *,
        num_steps: int,
        sde_indices: Optional[List[int]],
        emit_native_logprob: bool,
    ):
        """LTX-2 latents are PACKED token sequences, not a spatial video grid."""
        traj = utils.collect_trajectory_latents(results)
        if traj.ndim < 3:
            raise ValueError(
                f"ltx2: expected a packed trajectory [B, T+1, ...]; got rank {traj.ndim}, shape {tuple(traj.shape)}."
            )
        aux_traj = utils.collect_aux_trajectory_latents(results)
        return utils.build_latent_segment(
            traj,
            results=results,
            expected_sigmas=sample.frontier_gen_part(DiffusionSamplingParams).sampling_params.sigmas,
            num_steps=num_steps,
            sde_indices=sde_indices,
            emit_native_logprob=emit_native_logprob,
            segment_factory=self.segment_factory,
            aux_trajectory=aux_traj,
        )


__all__ = [
    "VideoAdapter",
    "MochiAdapter",
    "HunyuanVideoAdapter",
    "Wan21T2VAdapter",
    "Wan22T2VAdapter",
    "Ltx2T2VAdapter",
]
