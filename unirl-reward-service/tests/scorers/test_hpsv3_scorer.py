"""Tests for HPSv3Scorer checkpoint resolution.

``_resolve_checkpoint_path`` is a staticmethod precisely so it can be tested
without constructing the scorer — ``__init__`` imports hpsv3 and loads a 17 GB
Qwen2-VL backbone.
"""

from __future__ import annotations

import huggingface_hub
import pytest
from reward_service.scorers.hpsv3_scorer import HPSv3Scorer

resolve = HPSv3Scorer._resolve_checkpoint_path


def test_resolve_finds_checkpoint_inside_a_directory(tmp_path):
    ckpt = tmp_path / "HPSv3.safetensors"
    ckpt.write_bytes(b"")

    assert resolve(str(tmp_path)) == str(ckpt)


def test_resolve_rejects_a_directory_without_the_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="HPSv3.safetensors not found"):
        resolve(str(tmp_path))


def test_resolve_accepts_the_checkpoint_file_itself(tmp_path):
    ckpt = tmp_path / "custom-name.safetensors"
    ckpt.write_bytes(b"")

    assert resolve(str(ckpt)) == str(ckpt)


def test_resolve_downloads_a_repo_id_that_is_not_on_disk(monkeypatch):
    calls = []

    def fake_download(repo_id, filename, repo_type=None):
        calls.append((repo_id, filename, repo_type))
        return "/hub/cache/HPSv3.safetensors"

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    assert resolve("MizzenAI/HPSv3") == "/hub/cache/HPSv3.safetensors"
    assert calls == [("MizzenAI/HPSv3", "HPSv3.safetensors", "model")]


def test_resolve_reports_a_bad_value_as_file_not_found(monkeypatch):
    def fake_download(repo_id, filename, repo_type=None):
        raise huggingface_hub.errors.HFValidationError("bad repo id")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    with pytest.raises(FileNotFoundError, match="neither a local HPSv3 checkpoint"):
        resolve("/typo/in/my/config")
