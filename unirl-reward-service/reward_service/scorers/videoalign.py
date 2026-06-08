"""VideoAlign / VideoReward scorer.

VideoReward is the VLM-based reward model from
https://arxiv.org/abs/2501.13918 (Improving Video Generation with Human
Feedback). It scores generated videos along three axes — Visual Quality
(VQ), Motion Quality (MQ), Text Alignment (TA) — plus an unnormalised
Overall = VQ + MQ + TA.

The actual model class + processor + checkpoint loader are vendored
under :mod:`reward_service.scorers._videoalign`. This module only
glues the BaseScorer contract onto that inferencer.

Input contract:

* Each :class:`ScoreItem` must carry a video on its **last** turn —
  either as raw mp4 ``bytes`` (the server decoded ``video_b64``) or
  as a ``str`` path on the host filesystem (the server resolved
  ``video_path``). Items without a video raise ``ValueError``.
* The prompt is read from ``item.history[-1][0]``; the image, if any,
  is ignored — VideoReward consumes the video tokens only.

Bytes-input handling: each call writes incoming ``bytes`` to a
``tempfile.NamedTemporaryFile`` whose path is fed to decord/torchvision,
because the upstream readers only accept on-disk paths. Path inputs
are passed straight through (zero-copy on shared filesystems).
"""

from __future__ import annotations

import contextlib
import os
import tempfile

import torch

from reward_service.logging_utils import get_logger
from reward_service.scorers.base import BaseScorer, ScoreItem
from reward_service.scorers.registry import register

logger = get_logger(__name__)


_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _resolve_torch_dtype(name: str) -> torch.dtype:
    if name not in _DTYPE_MAP:
        raise ValueError(
            f"unknown dtype: {name!r}. expected one of {list(_DTYPE_MAP)}"
        )
    return _DTYPE_MAP[name]


class VideoAlignScorer(BaseScorer):
    name = "videoalign"
    # Overall (= VQ + MQ + TA) is listed first so the default consumer reduction
    # (UniRL's sub_metric_reduce="first") trains on the headline score
    # rather than Visual Quality alone. Callers wanting a single axis set
    # sub_metric_reduce or read the named sub-metric explicitly.
    sub_metric_names = ("Overall", "VQ", "MQ", "TA")

    def __init__(
        self,
        weights_path: str = "/path/to/RewardModel/VideoReward",
        checkpoint_step: int | None = -1,
        device: str = "cuda",
        dtype: str = "bfloat16",
        use_norm: bool = True,
        disable_flash_attn2: bool = False,
        fps: float | None = None,
        num_frames: int | None = None,
        max_pixels: int | None = None,
    ) -> None:
        """Load the VideoReward model.

        Args:
            weights_path: Directory containing ``model_config.json`` plus
                one or more ``checkpoint-<step>/`` subfolders. The
                upstream HF release has this exact layout.
            checkpoint_step: ``None`` / ``-1`` ⇒ pick the latest
                ``checkpoint-<n>``; otherwise the exact step (falls back
                to latest with a warning if the step is missing).
            device: Torch device string. Defaults to ``"cuda"`` and
                falls through to CPU if no CUDA is visible (the actor
                layer pins ``CUDA_VISIBLE_DEVICES`` so this is the
                normal cuda:0 inside the actor).
            dtype: ``"float32"`` / ``"float16"`` / ``"bfloat16"``.
                bfloat16 is the upstream training precision.
            use_norm: When True, apply per-dim mean/std rescaling using
                the ``inference_config`` block packaged inside the
                checkpoint's ``model_config.json``. Ignored if no such
                block was saved.
            disable_flash_attn2: Force ``attn_implementation="sdpa"``.
                The default builds against flash-attn 2 (pinned in
                ``envs/videoalign.txt``); flip this on environments
                where the flash-attn wheel doesn't match the active
                torch ABI.
            fps: Override the checkpoint's training FPS (mutually
                exclusive with ``num_frames``).
            num_frames: Override the checkpoint's frame count.
            max_pixels: Override the per-frame pixel budget; defaults to
                the checkpoint's ``max_frame_pixels``.
        """
        torch_dtype = _resolve_torch_dtype(dtype)

        # Imported here so the heavy transformers / peft / decord stack
        # only loads inside the actor's venv, not in the main process.
        from reward_service.scorers._videoalign import VideoRewardInferencer

        self._inferencer = VideoRewardInferencer(
            load_from_pretrained=weights_path,
            load_from_pretrained_step=checkpoint_step,
            device=device if torch.cuda.is_available() else "cpu",
            dtype=torch_dtype,
            disable_flash_attn2=disable_flash_attn2,
        )
        self._use_norm = use_norm
        self._fps = fps
        self._num_frames = num_frames
        self._max_pixels = max_pixels

    @torch.inference_mode()
    def score(self, items: list[ScoreItem]) -> list[dict[str, float]]:
        if not items:
            return []

        prompts: list[str] = []
        video_paths: list[str] = []
        # tempfile cleanup is deferred to ``finally`` so a forward-pass
        # failure still releases disk space.
        owned_tempfiles: list[str] = []

        try:
            for i, item in enumerate(items):
                if item.videos is None or item.videos[-1] is None:
                    raise ValueError(
                        f"videoalign requires a video on item[{i}]'s last turn; "
                        f"got videos={item.videos!r}"
                    )
                source = item.videos[-1]
                prompts.append(item.history[-1][0])
                video_paths.append(self._materialize_video(source, owned_tempfiles))

            rewards = self._inferencer.reward(
                video_paths=video_paths,
                prompts=prompts,
                fps=self._fps,
                num_frames=self._num_frames,
                max_pixels=self._max_pixels,
                use_norm=self._use_norm,
            )
        finally:
            for path in owned_tempfiles:
                with contextlib.suppress(OSError):
                    os.unlink(path)

        return [
            {
                "Overall": float(r["Overall"]),
                "VQ": float(r["VQ"]),
                "MQ": float(r["MQ"]),
                "TA": float(r["TA"]),
            }
            for r in rewards
        ]

    @staticmethod
    def _materialize_video(source, owned_tempfiles: list[str]) -> str:
        """Return a filesystem path for ``source``.

        ``str`` is passed through; ``bytes`` is spilled to a
        NamedTemporaryFile whose path is appended to ``owned_tempfiles``
        for cleanup by the caller.
        """
        if isinstance(source, str):
            return source
        if isinstance(source, (bytes, bytearray)):
            tf = tempfile.NamedTemporaryFile(
                prefix="videoalign_", suffix=".mp4", delete=False
            )
            try:
                tf.write(bytes(source))
                tf.flush()
            finally:
                tf.close()
            owned_tempfiles.append(tf.name)
            return tf.name
        raise TypeError(
            f"video source must be bytes or str (path), got {type(source).__name__}"
        )


register("videoalign", VideoAlignScorer)
