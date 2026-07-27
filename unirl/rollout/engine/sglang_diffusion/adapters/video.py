"""Video-family adapters.

Two output shapes live here:

* ``VideoAdapter`` — proper video output. The latent trajectory is video-form
  6-D ``[B, T+1, C, F, H, W]`` (an extra latent-frame axis vs the image path's
  5-D ``[B, T+1, C, H, W]``) and the decoded media is packed into a ragged
  :class:`~unirl.types.primitives.Videos` (``[total_T, C, H, W]``) instead of
  being dropped. HunyuanVideo and WAN ride this base so their rollout output can
  be consumed by the ``video_pickscore`` reward.

* ``MochiAdapter`` — kept on the legacy image path (see note below) for
  behavioral parity with the old ``sglang`` engine. Migrate it onto
  ``VideoAdapter`` once it has a verified video reward baseline.

PARITY NOTE (image-path video families): the legacy ``sglang`` engine treated
every family — including the video ones — through the image path: it built an
image-form ``LatentSegment`` (``make_image_segment``) and *dropped* 4-D decoded
video with a warning (there was no video reward consumer yet). ``MochiAdapter``
reproduces that exactly; families with a real video consumer move to
``VideoAdapter``.
"""

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
    """Base for true video-output families (6-D latent trajectory → ``Videos``).

    Reuses ``ImageAdapter``'s request side verbatim — ``build_sampling`` already
    forwards ``num_frames`` and the SDE/rollout pins are modality-agnostic — and
    overrides only the response-shape variation points: the segment is stamped
    ``Modality.VIDEO`` and carries the 6-D ``[B, T+1, C, F, H, W]`` trajectory,
    and the decoded media is packed as ``Videos`` rather than dropped.
    """

    #: Modality stamp for the latent segment.
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
        """Video-form trajectory: collect, gate the 6-D shape, assemble.

        Video latents keep the extra frame axis throughout, so the trajectory is
        rank 6 ``[B, T+1, C, F, H, W]`` (vs the image path's rank 5). The downstream
        ``build_latent_segment`` is shape-agnostic past the T+1 invariant, so the
        only difference from the image path is the rank gate + the video segment
        factory.
        """
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

    # Legacy image-path video family: drop 4-D decoded samples (incl. single-frame)
    # rather than squeezing them into images.
    squeeze_single_frame_4d = False


@register_adapter("hunyuan_video")
class HunyuanVideoAdapter(VideoAdapter):
    """HunyuanVideo-1.0 T2V with video output and dual text conditions."""

    def build_condition(self, results: List[RawResult]) -> Dict[str, Any]:
        """Keep HunyuanVideo's LLaMA and pooled-CLIP streams separate.

        SGLang returns ``prompt_embeds`` as
        ``[LLaMA [B, seq, 4096], CLIP [B, 1, 768]]``. The generic image adapter
        fuses multi-encoder outputs along the token axis, which cannot combine
        these different hidden sizes and does not match
        :class:`HunyuanVideoConditions`. Pack the two streams under the exact
        keys consumed by trainer-side replay instead.
        """
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
    """WAN 2.2-A14B T2V — DUAL-EXPERT (high-noise / low-noise) MoE.

    WAN 2.2-A14B runs two ``WanTransformer3DModel`` experts switched at a sigma
    boundary (``boundary_ratio=0.875``): high-noise for ``sigma >= boundary``
    (coarse structure, early steps), low-noise for ``sigma < boundary`` (detail).
    The entire dual-expert mechanism lives ENGINE-SIDE in sglang and needs no
    adapter work: ``composed_pipeline_base.load_modules`` auto-loads ``transformer_2``
    when the checkpoint's ``model_index.json`` carries ``boundary_ratio`` + both
    ``transformer``/``transformer_2`` (the A14B-Diffusers ckpt does), and the generic
    ``DenoisingStage._select_and_manage_model`` routes per-step by the boundary
    timestep (and applies ``guidance_scale_2`` to the low-noise branch). So the
    UniRL side is byte-identical to WAN 2.1 — same UMT5 single-text fuse, same 6-D
    video trajectory + ``video_pickscore`` consumer, same segment contract (no aux
    audio). The trainside ``WAN22DiffusionStage`` replays with the SAME boundary
    routing, so rollout↔replay stays aligned.

    ``build_sampling`` additionally forwards ``guidance_scale_2`` so the engine's
    low-noise CFG branch matches the trainside; it is omitted (engine falls back to
    ``guidance_scale``) when unset, so a ``guidance_scale=1.0`` smoke is unaffected.
    """

    def build_sampling(self, sample: Sample, *, diffusion: Any) -> Dict[str, Any]:
        kwargs = super().build_sampling(sample, diffusion=diffusion)
        g2 = getattr(diffusion, "guidance_scale_2", None)
        if g2 is not None:
            kwargs["guidance_scale_2"] = float(g2)
        return kwargs


@register_adapter("wan21")
class Wan21T2VAdapter(VideoAdapter):
    """WAN 2.1 T2V — proper video output consumed by ``video_pickscore``.

    The text/conditions path is the generic UMT5 fuse from ``ImageAdapter``
    (single text encoder; no CFG negative branch when ``guidance_scale <= 1``);
    only the video-output overrides on ``VideoAdapter`` apply. The sglang server
    resolves the WAN pipeline from ``model_path`` (the ``Wan-AI/Wan2.1-T2V-1.3B``
    -Diffusers checkpoint), so no extra ``boot_kwargs`` are needed.
    """

    pass


__all__ = [
    "VideoAdapter",
    "MochiAdapter",
    "HunyuanVideoAdapter",
    "Wan21T2VAdapter",
    "Wan22T2VAdapter",
]
