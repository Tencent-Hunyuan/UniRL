import pytest

from unirl.data.data_source import _load_videos
from unirl.types.media import MediaRef


def _video(role: str, uri: str) -> MediaRef:
    return MediaRef(modality="video", role=role, uri=uri)


def test_load_prompt_videos_preserves_batch_alignment() -> None:
    videos = _load_videos(
        [
            [_video("prompt", "/video/first.mp4")],
            [_video("prompt", "/video/second.mp4")],
        ]
    )

    assert videos is not None
    assert videos.frames is None
    assert videos.uris == ["/video/first.mp4", "/video/second.mp4"]
    assert len(videos) == 2


def test_load_prompt_videos_rejects_heterogeneous_batch() -> None:
    with pytest.raises(ValueError, match="Heterogeneous prompt-video batch"):
        _load_videos([[_video("prompt", "/video/first.mp4")], []])


def test_load_videos_rejects_mixed_roles() -> None:
    with pytest.raises(ValueError, match="cannot mix"):
        _load_videos(
            [
                [_video("condition", "/video/condition.mp4")],
                [_video("prompt", "/video/prompt.mp4")],
            ]
        )
