import enum
import importlib.util
import sys
import types

import torch

# ``unirl.reward.local`` eagerly imports VideoAlign, whose type-level enum comes
# from torchvision. The helper under test is torch-only, so keep this CPU unit
# test runnable in minimal environments where the optional media wheel is absent.
if importlib.util.find_spec("torchvision") is None:
    torchvision = types.ModuleType("torchvision")
    transforms = types.ModuleType("torchvision.transforms")

    class _InterpolationMode(enum.Enum):
        BICUBIC = "bicubic"

    transforms.InterpolationMode = _InterpolationMode
    torchvision.transforms = transforms
    sys.modules["torchvision"] = torchvision
    sys.modules["torchvision.transforms"] = transforms

from unirl.reward.local.imagebind import ImageBindRewardScorer


def test_imagebind_temporal_subsampling_uses_video_time_axis() -> None:
    video = torch.arange(3 * 6, dtype=torch.float32).reshape(3, 6, 1, 1)

    clips = ImageBindRewardScorer._temporal_subsample_clips(
        video,
        num_clips=2,
        frames_per_clip=2,
    )

    assert [tuple(clip.shape) for clip in clips] == [(3, 2, 1, 1), (3, 2, 1, 1)]
    assert torch.equal(clips[0], video[:, torch.tensor([0, 1])])
    assert torch.equal(clips[1], video[:, torch.tensor([3, 4])])
