"""VideoAlign reward scorer (REFL-compatible / BPTT-differentiable).

Wraps :class:`VideoRewardWrapper` (Qwen2-VL-based reward model producing
three scalar scores per (video, prompt) pair: VQ / MQ / TA) and exposes a
recipe-local differentiable REFL entry point.

Reward = ``w_vq * VQ + w_mq * MQ + w_ta * TA``  (defaults to 1 / 1 / 1).

Gradient flow
-------------
The Qwen2-VL vision encoder is differentiable w.r.t. the input pixels when
the *fast* image processor is used (the wrapper force-installs
``Qwen2VLImageProcessorFast`` on construction). The generated video arrives
via ``compute_rewards_differentiable`` as ``[B, C, T, H, W]`` float in
``[-1, 1]`` with a live ``grad_fn`` (BPTT path); we forward into the wrapper under
``torch.enable_grad`` so the linear combination of VQ/MQ/TA traces back
through the vision tower into the diffusion graph.

Self-containment
----------------
This scorer no longer requires the sibling ``mmrl`` repo on disk. The
Qwen2-VL reward backbone, prompt template, checkpoint loader and
inference wrapper all live under
:mod:`experimental.refl.reward.videoalign.model` / :mod:`...wrapper`. The
``mmrl_repo_root`` Spec field has been removed; ``MMRL_REPO_ROOT`` env
var is now irrelevant.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import torch

from unirl.reward.base import BaseRewardComponentSpec
from unirl.reward.local.base import LocalRewardBackend
from unirl.reward.local.device import resolve_device
from unirl.types.reward import RewardRequest

from .wrapper import VideoRewardWrapper

logger = logging.getLogger(__name__)


class VideoAlignRewardScorer(LocalRewardBackend):
    """Qwen2-VL VideoAlign reward — VQ + MQ + TA → scalar per sample."""

    canonical_model_name = "videoalign"
    input_kind = "video"

    def __init__(self, *, config: "VideoAlignSpec", base_device: str) -> None:
        super().__init__(
            model_name=self.canonical_model_name,
            device=resolve_device(config.device, base_device),
            batch_size=int(config.batch_size),
            reward_model_path=config.reward_model_path,
            resize_height=config.resize_height,
            resize_width=config.resize_width,
            micro_batch_size=config.micro_batch_size,
            reward_num_frames=config.reward_num_frames,
            use_norm=config.use_norm,
            w_vq=config.w_vq,
            w_mq=config.w_mq,
            w_ta=config.w_ta,
            differentiable=config.differentiable,
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        reward_model_path = str(self.model_kwargs["reward_model_path"])
        if not reward_model_path:
            raise ValueError(
                "VideoAlignRewardScorer: ``reward_model_path`` must be set "
                "(the directory containing ``model_config.json`` + the "
                "``checkpoint-*`` subdir)."
            )

        self._w_vq = float(self.model_kwargs["w_vq"])
        self._w_mq = float(self.model_kwargs["w_mq"])
        self._w_ta = float(self.model_kwargs["w_ta"])
        self._use_norm = bool(self.model_kwargs["use_norm"])
        self._reward_num_frames = int(self.model_kwargs["reward_num_frames"])
        self._differentiable = bool(self.model_kwargs["differentiable"])

        logger.info(
            "VideoAlignRewardScorer: loading VideoRewardWrapper from %s",
            reward_model_path,
        )
        self.model = VideoRewardWrapper(
            checkpoint_dir=reward_model_path,
            device=self.device,
            dtype=torch.bfloat16,
            use_norm=self._use_norm,
            resize_height=int(self.model_kwargs["resize_height"]),
            resize_width=int(self.model_kwargs["resize_width"]),
            micro_batch_size=int(self.model_kwargs["micro_batch_size"]),
        )

    def _compute_model_rewards(self, request: RewardRequest) -> List[float]:
        raise NotImplementedError("VideoAlignRewardScorer is REFL-only; use compute_rewards_differentiable().")

    # ------------------------------------------------------------------
    # Differentiable REFL reward entry point
    # ------------------------------------------------------------------

    def compute_rewards_differentiable(
        self,
        media_tensor: torch.Tensor,
        prompts: List[str],
        records: Optional[List[dict]] = None,
    ) -> torch.Tensor:
        del records
        if media_tensor.ndim != 5:
            raise ValueError(f"VideoAlignRewardScorer expects [B,C,T,H,W], got {tuple(media_tensor.shape)}")
        if len(prompts) != int(media_tensor.shape[0]):
            raise ValueError(
                f"VideoAlignRewardScorer: prompts length {len(prompts)} != batch size {int(media_tensor.shape[0])}."
            )

        # The wrapper expects per-sample [T, C, H, W] in [-1, 1].
        # NOTE: ``.clamp(-1.0, 1.0)`` mirrors mmrl's ``role.score`` — VAE
        # decode can produce slightly out-of-range pixels (e.g. -1.02 /
        # 1.03), and ``_pixels_neg1_to_255`` only clamps the [0, 1]
        # midpoint afterwards, so values can still spill above 255 or
        # below 0 without this safeguard. Required for numeric parity
        # with the mmrl baseline.
        per_sample_videos: List[torch.Tensor] = []
        for v in media_tensor:
            v = v.to(self.device).permute(1, 0, 2, 3).clamp(-1.0, 1.0).contiguous()  # → (T, C, H, W)
            per_sample_videos.append(v)

        if self._reward_num_frames > 0:
            ds: List[torch.Tensor] = []
            for v in per_sample_videos:
                if v.shape[0] > self._reward_num_frames:
                    idx = torch.linspace(
                        0,
                        v.shape[0] - 1,
                        self._reward_num_frames,
                        device=v.device,
                    ).long()
                    v = v[idx]
                ds.append(v)
            per_sample_videos = ds

        autograd_ctx = torch.enable_grad if self._differentiable else torch.no_grad
        with autograd_ctx():
            scores = self.model.forward_scores(
                per_sample_videos,
                prompts,
                use_norm=self._use_norm,
            )

        reward = self._w_vq * scores["VQ"] + self._w_mq * scores["MQ"] + self._w_ta * scores["TA"]
        return reward.float()

    # ------------------------------------------------------------------
    # Lifecycle hooks (CPU offload between rollouts to free VRAM)
    # ------------------------------------------------------------------

    def offload(self) -> None:
        if self.model is not None and getattr(self.model, "model", None) is not None:
            self.model.model.cpu()
            torch.cuda.empty_cache()

    def onload(self) -> None:
        if self.model is not None and getattr(self.model, "model", None) is not None:
            self.model.model.to(self.device)

    def is_available(self) -> bool:
        return bool(self._is_loaded)

    def dispose(self) -> None:
        self.offload()


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------


@dataclass
class VideoAlignSpec(BaseRewardComponentSpec):
    """Typed config for :class:`VideoAlignRewardScorer`.

    Args:
        reward_model_path: Directory containing ``model_config.json`` and
            the ``checkpoint-*`` subdir (with ``model.pth`` or LoRA split).
            Required.
        device: ``"auto"`` / ``"cuda"`` / ``"cuda:N"`` — resolved against
            ``base_device``.
        batch_size: kept for parity with sibling specs.
        resize_height / resize_width: bicubic target before the Qwen2-VL
            vision encoder. Defaults (336 × 588) match the published
            checkpoints.
        micro_batch_size: max samples per reward forward (peak-VRAM knob).
        reward_num_frames: temporal downsample to this many uniformly
            spaced frames before scoring; ``<= 0`` disables.
        use_norm: z-score normalise each dimension using the means / stds
            stored under ``inference_config`` in ``model_config.json``.
        w_vq, w_mq, w_ta: linear combination weights into the final scalar.
        differentiable: keep autograd on the reward forward (default True —
            required for REFL). Set False for historical GRPO / replay.
    """

    reward_model_path: str = ""

    device: str = "auto"
    batch_size: int = 1

    resize_height: int = 336
    resize_width: int = 588
    micro_batch_size: int = 1
    reward_num_frames: int = 36

    use_norm: bool = True
    w_vq: float = 1.0
    w_mq: float = 1.0
    w_ta: float = 1.0

    differentiable: bool = True


__all__ = [
    "VideoAlignRewardScorer",
    "VideoAlignSpec",
]
