"""Unit tests for the shared scorer helpers in ``_common``.

Only ``materialize_video`` is covered here — it is the one helper with
resource-ownership semantics (tempfile spilling) worth pinning down.
"""

from __future__ import annotations

import os

import pytest
from reward_service.scorers._common import materialize_video


def test_materialize_video_passes_a_path_through_without_spilling():
    # Arrange
    owned_tempfiles: list[str] = []

    # Act
    path = materialize_video("/data/clip.mp4", owned_tempfiles, prefix="test_")

    # Assert: paths are used in place, so nothing is owned.
    assert path == "/data/clip.mp4"
    assert owned_tempfiles == []


@pytest.mark.parametrize("source", [b"video-bytes", bytearray(b"video-bytes")])
def test_materialize_video_spills_bytes_to_an_owned_tempfile(source):
    # Arrange
    owned_tempfiles: list[str] = []

    # Act
    path = materialize_video(source, owned_tempfiles, prefix="test_")

    # Assert
    try:
        assert owned_tempfiles == [path]
        assert os.path.basename(path).startswith("test_")
        assert path.endswith(".mp4")
        with open(path, "rb") as fh:
            assert fh.read() == b"video-bytes"
    finally:
        os.unlink(path)


def test_materialize_video_records_the_tempfile_before_writing_it(monkeypatch):
    """A failed write must still leave the path with the caller to unlink.

    ``NamedTemporaryFile(delete=False)`` creates the file before any write,
    so recording it only after a successful write would strand it on disk
    exactly when the disk is the problem (ENOSPC).
    """
    # Arrange: a real tempfile whose write fails, as a full disk would.
    owned_tempfiles: list[str] = []

    def _raise_enospc(self, data):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("tempfile._TemporaryFileWrapper.write", _raise_enospc, raising=False)

    # Act
    with pytest.raises(OSError, match="No space left on device"):
        materialize_video(b"video-bytes", owned_tempfiles, prefix="test_")

    # Assert: the caller can clean up what was actually created.
    assert len(owned_tempfiles) == 1
    try:
        assert os.path.exists(owned_tempfiles[0])
    finally:
        os.unlink(owned_tempfiles[0])


def test_materialize_video_rejects_a_source_that_is_neither_bytes_nor_path():
    with pytest.raises(TypeError, match="must be bytes or str"):
        materialize_video(12345, [], prefix="test_")
