"""Unit tests for the videohpsv3 scorer.

The HPSv3 model itself is never loaded: ``VideoHPSv3Scorer`` composes
:class:`HPSv3Scorer`, so patching that one name yields a scorer whose
frame-scoring step is a stub while decode + aggregation run for real.
"""

from __future__ import annotations

import math
import os
import tempfile

import numpy as np
import pytest
from PIL import Image
from reward_service.scorers import video_hpsv3
from reward_service.scorers.base import ScoreItem
from reward_service.scorers.video_hpsv3 import (
    VideoHPSv3Scorer,
    extract_frames,
    top_ratio_mean,
)


class FakeFrameScorer:
    """Stand-in for HPSv3Scorer: returns a preset score per frame."""

    def __init__(self, frame_scores: list[float] | None = None, **kwargs) -> None:
        self.frame_scores = frame_scores
        self.init_kwargs = kwargs
        self.batches: list[int] = []
        self.closed = False

    def score(self, items: list[ScoreItem]) -> list[dict[str, float]]:
        self.batches.append(len(items))
        if self.frame_scores is None:
            return [{"hpsv3": float(i)} for i in range(len(items))]
        # BaseScorer.score promises one output per input; a fake that silently
        # returned fewer could mask a real mis-sizing bug.
        assert len(self.frame_scores) >= len(items), (
            f"fake has {len(self.frame_scores)} preset scores for {len(items)} frames"
        )
        return [{"hpsv3": s} for s in self.frame_scores[: len(items)]]

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def make_scorer(monkeypatch):
    """Build a VideoHPSv3Scorer whose frame scorer is a FakeFrameScorer."""

    def _make(frame_scores: list[float] | None = None, **scorer_kwargs):
        created: list[FakeFrameScorer] = []

        def _factory(**kwargs):
            fake = FakeFrameScorer(frame_scores=frame_scores, **kwargs)
            created.append(fake)
            return fake

        monkeypatch.setattr(video_hpsv3, "HPSv3Scorer", _factory)
        scorer = VideoHPSv3Scorer(**scorer_kwargs)
        return scorer, created[0]

    return _make


@pytest.fixture
def assert_no_leaked_tempfiles():
    """Fail if the scorer left a ``videohpsv3_`` tempfile behind."""
    tempdir = tempfile.gettempdir()
    before = set(os.listdir(tempdir))

    yield

    leaked = {
        name
        for name in set(os.listdir(tempdir)) - before
        if name.startswith("videohpsv3_")
    }
    assert not leaked, f"tempfiles leaked: {leaked}"


def _write_video(path, num_frames: int, size: tuple[int, int] = (64, 48)) -> str:
    """Write a solid-colour mp4 with ``num_frames`` frames; return its path."""
    cv2 = pytest.importorskip("cv2")
    width, height = size
    out = str(path)
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (width, height))
    if not writer.isOpened():
        pytest.skip("no mp4v encoder available in this OpenCV build")
    try:
        for i in range(num_frames):
            frame = np.full((height, width, 3), (i * 7) % 256, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()
    return out


# ───────────────────────── top_ratio_mean ─────────────────────────


def test_top_ratio_mean_matches_upstream_on_ten_frames():
    # Arrange: upstream keeps int(10 * 0.3) == 3 frames.
    scores = [1.0, 5.0, 3.0, 10.0, 2.0, 9.0, 4.0, 8.0, 6.0, 7.0]

    # Act
    result = top_ratio_mean(scores, 0.3)

    # Assert: mean of 10, 9, 8.
    assert result == pytest.approx(9.0)


@pytest.mark.parametrize(
    ("num_frames", "expected_kept"),
    [(1, 1), (2, 1), (3, 1), (4, 1), (10, 3), (81, 24), (100, 30)],
)
def test_top_ratio_mean_keeps_expected_frame_count(num_frames, expected_kept):
    # Arrange: descending scores so the mean identifies how many were kept.
    scores = [float(num_frames - i) for i in range(num_frames)]

    # Act
    result = top_ratio_mean(scores, 0.3)

    # Assert: mean of the top `expected_kept` values.
    top = [float(num_frames - i) for i in range(expected_kept)]
    assert result == pytest.approx(sum(top) / len(top))


def test_top_ratio_mean_uses_single_best_frame_where_upstream_divides_by_zero():
    # Arrange: int(3 * 0.3) == 0 upstream -> ZeroDivisionError.
    scores = [1.0, 7.0, 4.0]

    # Act
    result = top_ratio_mean(scores, 0.3)

    # Assert
    assert result == pytest.approx(7.0)


def test_top_ratio_mean_with_ratio_one_averages_all_frames():
    result = top_ratio_mean([1.0, 2.0, 3.0, 4.0], 1.0)

    assert result == pytest.approx(2.5)


def test_top_ratio_mean_rejects_empty_scores():
    with pytest.raises(ValueError, match="must not be empty"):
        top_ratio_mean([], 0.3)


@pytest.mark.parametrize("bad_ratio", [0.0, -0.1, 1.5])
def test_top_ratio_mean_rejects_out_of_range_ratio(bad_ratio):
    with pytest.raises(ValueError, match="top_ratio"):
        top_ratio_mean([1.0, 2.0], bad_ratio)


# ───────────────────────── extract_frames ─────────────────────────


def test_extract_frames_returns_rgb_pil_images(tmp_path):
    # Arrange
    video = _write_video(tmp_path / "clip.mp4", num_frames=5)

    # Act
    frames = extract_frames(video, frame_stride=1)

    # Assert
    assert len(frames) == 5
    assert all(isinstance(f, Image.Image) for f in frames)
    assert all(f.mode == "RGB" for f in frames)
    assert frames[0].size == (64, 48)


@pytest.mark.parametrize(
    ("stride", "expected"),
    [(1, 10), (2, 5), (3, 4), (10, 1)],
)
def test_extract_frames_subsamples_by_stride(tmp_path, stride, expected):
    # Arrange
    video = _write_video(tmp_path / f"clip_{stride}.mp4", num_frames=10)

    # Act
    frames = extract_frames(video, frame_stride=stride)

    # Assert
    assert len(frames) == expected


def test_extract_frames_rejects_non_positive_stride(tmp_path):
    video = _write_video(tmp_path / "clip.mp4", num_frames=2)

    with pytest.raises(ValueError, match="frame_stride must be positive"):
        extract_frames(video, frame_stride=0)


def test_extract_frames_raises_on_missing_file(tmp_path):
    with pytest.raises(ValueError, match="could not open video"):
        extract_frames(str(tmp_path / "does_not_exist.mp4"))


# ───────────────────────── constructor validation ─────────────────────────


def test_init_rejects_non_positive_frame_stride(make_scorer):
    with pytest.raises(ValueError, match="frame_stride must be positive"):
        make_scorer(frame_stride=0)


@pytest.mark.parametrize("bad_ratio", [0.0, -1.0, 2.0])
def test_init_rejects_out_of_range_top_ratio(make_scorer, bad_ratio):
    with pytest.raises(ValueError, match="top_ratio"):
        make_scorer(top_ratio=bad_ratio)


def test_init_forwards_checkpoint_params_to_frame_scorer(make_scorer):
    # Act
    _, fake = make_scorer(
        weights_path="/ckpt/HPSv3", config_path="/cfg.yaml", device="cpu", max_batch_size=2
    )

    # Assert
    assert fake.init_kwargs == {
        "weights_path": "/ckpt/HPSv3",
        "config_path": "/cfg.yaml",
        "device": "cpu",
        "max_batch_size": 2,
    }


# ───────────────────────── score ─────────────────────────


def test_score_returns_empty_list_for_empty_input(make_scorer):
    scorer, _ = make_scorer()

    assert scorer.score([]) == []


def test_score_aggregates_top_thirty_percent_of_frames(tmp_path, make_scorer):
    # Arrange: 10 frames, so the top 3 of the preset scores are averaged.
    frame_scores = [1.0, 5.0, 3.0, 10.0, 2.0, 9.0, 4.0, 8.0, 6.0, 7.0]
    scorer, fake = make_scorer(frame_scores=frame_scores)
    video = _write_video(tmp_path / "clip.mp4", num_frames=10)
    item = ScoreItem(history=[("a cat", None)], videos=(video,))

    # Act
    result = scorer.score([item])

    # Assert
    assert result == [{"videohpsv3": pytest.approx(9.0)}]
    assert fake.batches == [10]


def test_score_passes_prompt_to_every_frame(tmp_path, make_scorer):
    # Arrange
    captured: list[ScoreItem] = []
    scorer, fake = make_scorer()
    original_score = fake.score

    def _spy(items):
        captured.extend(items)
        return original_score(items)

    fake.score = _spy
    video = _write_video(tmp_path / "clip.mp4", num_frames=4)
    item = ScoreItem(history=[("a dog on grass", None)], videos=(video,))

    # Act
    scorer.score([item])

    # Assert
    assert len(captured) == 4
    assert all(fi.history[0][0] == "a dog on grass" for fi in captured)
    assert all(isinstance(fi.history[0][1], Image.Image) for fi in captured)


def test_score_handles_multiple_items_independently(tmp_path, make_scorer):
    # Arrange: constant per-frame score so each clip's mean is predictable.
    scorer, _ = make_scorer(frame_scores=[2.0] * 10)
    video_a = _write_video(tmp_path / "a.mp4", num_frames=10)
    video_b = _write_video(tmp_path / "b.mp4", num_frames=10)
    items = [
        ScoreItem(history=[("a", None)], videos=(video_a,)),
        ScoreItem(history=[("b", None)], videos=(video_b,)),
    ]

    # Act
    result = scorer.score(items)

    # Assert
    assert result == [
        {"videohpsv3": pytest.approx(2.0)},
        {"videohpsv3": pytest.approx(2.0)},
    ]


def test_score_respects_frame_stride(tmp_path, make_scorer):
    # Arrange
    scorer, fake = make_scorer(frame_scores=[1.0] * 10, frame_stride=5)
    video = _write_video(tmp_path / "clip.mp4", num_frames=10)
    item = ScoreItem(history=[("a", None)], videos=(video,))

    # Act
    scorer.score([item])

    # Assert: 10 frames at stride 5 -> indices 0 and 5.
    assert fake.batches == [2]


def test_score_respects_top_ratio(tmp_path, make_scorer):
    # Arrange: top_ratio=1.0 averages all 4 frames instead of only the best.
    scorer, _ = make_scorer(frame_scores=[1.0, 2.0, 3.0, 4.0], top_ratio=1.0)
    video = _write_video(tmp_path / "clip.mp4", num_frames=4)
    item = ScoreItem(history=[("a", None)], videos=(video,))

    # Act
    result = scorer.score([item])

    # Assert: 2.5, not the 4.0 the default 0.3 ratio would give.
    assert result == [{"videohpsv3": pytest.approx(2.5)}]


def test_score_accepts_raw_bytes_and_cleans_up_tempfile(
    tmp_path, make_scorer, assert_no_leaked_tempfiles
):
    # Arrange
    scorer, _ = make_scorer(frame_scores=[3.0] * 5)
    with open(_write_video(tmp_path / "clip.mp4", num_frames=5), "rb") as fh:
        video_bytes = fh.read()
    item = ScoreItem(history=[("a", None)], videos=(video_bytes,))

    # Act
    result = scorer.score([item])

    # Assert (leak check runs in the fixture teardown)
    assert result == [{"videohpsv3": pytest.approx(3.0)}]


def test_score_returns_nan_when_a_clip_cannot_be_decoded(make_scorer, assert_no_leaked_tempfiles):
    # Arrange: bytes that are not a decodable video.
    scorer, _ = make_scorer()
    item = ScoreItem(history=[("a", None)], videos=(b"not a video",))

    # Act
    result = scorer.score([item])

    # Assert: NaN is the in-band "unavailable" marker (see BaseScorer.score).
    assert math.isnan(result[0]["videohpsv3"])


def test_score_isolates_an_undecodable_clip_from_its_batch(
    tmp_path, make_scorer, assert_no_leaked_tempfiles
):
    # Arrange: a good clip either side of a corrupt one.
    scorer, _ = make_scorer(frame_scores=[4.0] * 5)
    items = [
        ScoreItem(history=[("a", None)], videos=(_write_video(tmp_path / "a.mp4", 5),)),
        ScoreItem(history=[("b", None)], videos=(b"not a video",)),
        ScoreItem(history=[("c", None)], videos=(_write_video(tmp_path / "c.mp4", 5),)),
    ]

    # Act
    result = scorer.score(items)

    # Assert: the corrupt clip does not cost the whole bucket its scores.
    assert result[0]["videohpsv3"] == pytest.approx(4.0)
    assert math.isnan(result[1]["videohpsv3"])
    assert result[2]["videohpsv3"] == pytest.approx(4.0)


def test_score_raises_when_videos_field_is_none(make_scorer):
    scorer, _ = make_scorer()
    item = ScoreItem(history=[("a", None)], videos=None)

    with pytest.raises(ValueError, match=r"requires a video on item\[0\]"):
        scorer.score([item])


def test_score_raises_when_last_turn_has_no_video(make_scorer):
    scorer, _ = make_scorer()
    item = ScoreItem(history=[("a", None), ("b", None)], videos=("/tmp/x.mp4", None))

    with pytest.raises(ValueError, match=r"requires a video on item\[0\]"):
        scorer.score([item])


def test_score_error_message_identifies_the_offending_item_index(tmp_path, make_scorer):
    # Arrange: first item is fine, second is missing its video.
    scorer, _ = make_scorer(frame_scores=[1.0] * 3)
    video = _write_video(tmp_path / "clip.mp4", num_frames=3)
    items = [
        ScoreItem(history=[("a", None)], videos=(video,)),
        ScoreItem(history=[("b", None)], videos=None),
    ]

    # Act / Assert
    with pytest.raises(ValueError, match=r"item\[1\]"):
        scorer.score(items)


def test_score_rejects_video_source_of_wrong_type(make_scorer):
    scorer, _ = make_scorer()
    item = ScoreItem(history=[("a", None)], videos=(12345,))

    with pytest.raises(TypeError, match="must be bytes or str"):
        scorer.score([item])


def test_close_releases_the_frame_scorer(make_scorer):
    scorer, fake = make_scorer()

    scorer.close()

    assert fake.closed
