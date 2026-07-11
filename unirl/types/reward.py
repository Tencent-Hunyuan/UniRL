"""Shared reward data types."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image

from unirl.distributed.tensor.batch import Batch, concat_field, max_field


class RewardType(Enum):
    """Types of reward computation."""

    IMAGE_TEXT_ALIGNMENT = "image_text_alignment"
    AESTHETIC = "aesthetic"
    CUSTOM = "custom"


@dataclass
class RewardRequest:
    """Request for reward computation.

    Two typed primitive dicts mirror the ``RolloutReq`` contract:

    ``primitives``
        Input context — what was fed to the model.  Copied from
        ``RolloutReq.primitives``.  Typical keys: ``"text"`` (prompt
        ``Texts``), ``"image"`` (conditioning ``Images``).

    ``generated``
        Model output being scored — from ``RolloutTrack.decoded``.
        Typical keys: ``"image"`` (generated ``Images``), ``"video"``
        (generated ``Videos``), ``"text"`` (generated ``Texts``).

    Backward-compat properties (``prompts``, ``images``, ``videos``,
    ``texts``) bridge to the new structure with lazy format conversion
    so existing scorers work unchanged.
    """

    primitives: Dict[str, Any] = field(default_factory=dict)
    generated: Dict[str, Any] = field(default_factory=dict)
    metadata: Optional[List[Optional[Dict[str, Any]]]] = None
    prompt_ids: Optional[List[str]] = None
    sample_ids: Optional[List[str]] = None
    group_ids: Optional[List[str]] = None
    reward_types: List[RewardType] = field(default_factory=lambda: [RewardType.IMAGE_TEXT_ALIGNMENT])
    return_components: bool = False
    # Source sample rate (Hz) of any waveforms in ``generated["audio"]``. Set by
    # the reward service when a track carries a parallel audio stream (LTX-2.3
    # T2AV). ``None`` for non-audio requests. Audio reward scorers (CLAP /
    # ImageBind) read it to resample to their model's expected rate.
    audio_sample_rate: Optional[int] = None

    @property
    def prompts(self) -> List[str]:
        prim = self.primitives.get("text")
        if prim is None:
            return []
        return list(prim.texts)

    @property
    def images(self) -> Optional[List[Union[Image.Image, torch.Tensor]]]:
        prim = self.generated.get("image")
        if prim is None:
            return None
        from unirl.utils.media import tensor_frame_to_pil

        return [tensor_frame_to_pil(img) for img in prim.pixels.unbind(0)]

    @property
    def videos(self) -> Optional[List[torch.Tensor]]:
        prim = self.generated.get("video")
        if prim is None:
            return None
        return [v.frames.permute(1, 0, 2, 3).contiguous() for v in prim.to_list()]

    @property
    def texts(self) -> Optional[List[str]]:
        prim = self.generated.get("text")
        if prim is None:
            return None
        return list(prim.texts)

    @property
    def audio(self) -> Optional[List[torch.Tensor]]:
        """Generated audio waveforms, one ``[C, L]`` (or ``[L]``) tensor per sample.

        Populated when a track carries a parallel audio stream (LTX-2.3 T2AV);
        the reward service places the decoded ``Audios`` under
        ``generated["audio"]``. ``None`` for non-audio requests.
        """
        prim = self.generated.get("audio")
        if prim is None:
            return None
        return [a.waveform for a in prim.to_list()]

    @property
    def batch_size(self) -> int:
        for v in self.generated.values():
            if v is not None:
                return len(v)
        for v in self.primitives.values():
            if v is not None:
                return len(v)
        return 0

    @property
    def is_video(self) -> bool:
        return "video" in self.generated

    @property
    def has_audio(self) -> bool:
        return "audio" in self.generated


@dataclass
class RewardResponse(Batch):
    """
    Response from reward computation.

    Contains both aggregated rewards and optional per-component reward breakdowns.

    ``compute_time`` is reduced by ``max`` when multiple responses are
    concatenated — it is a wall-clock measurement, so the max across
    parallel-produced responses is the meaningful aggregate.
    """

    rewards: List[float] = concat_field(default_factory=list)
    component_rewards: Dict[str, List[float]] = concat_field(default_factory=dict)
    successes: List[bool] = concat_field(default_factory=list)
    errors: List[Optional[str]] = concat_field(default_factory=list)
    compute_time: float = max_field(default=0.0)

    @property
    def batch_size(self) -> int:
        return len(self.rewards)


__all__ = [
    "RewardRequest",
    "RewardResponse",
    "RewardType",
]
