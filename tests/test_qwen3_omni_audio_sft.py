from __future__ import annotations

import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from unirl.data.sft import normalize_supervised_example
from unirl.models.qwen3_omni.media import load_qwen3_audio
from unirl.train.sft.track_builder import ARSupervisedTrackBuilder
from unirl.types.media import MediaRefs
from unirl.types.sample import Turn


class _Tokenizer:
    eos_token_id = 99

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        assert not add_special_tokens
        return {"input_ids": [len(word) for word in text.split()]}


class _AudioChatStage:
    def __init__(self) -> None:
        self.received: list[Turn] | None = None

    def embed(self, value):
        if isinstance(value, list):
            self.received = value
            return "audio-conditions"
        return value


def _builder() -> tuple[ARSupervisedTrackBuilder, _AudioChatStage]:
    stage = _AudioChatStage()
    pipeline = SimpleNamespace(
        chat_template=stage,
        bundle=SimpleNamespace(tokenizer=_Tokenizer(), device=torch.device("cpu")),
    )
    return ARSupervisedTrackBuilder(pipeline=pipeline, max_response_length=16), stage


def test_normalize_standalone_audio_sft_record(tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    audio.touch()
    record = normalize_supervised_example(
        {
            "prompt": "What sound is this?",
            "response": "The answer is A.",
            "media_refs": [{"modality": "audio", "role": "prompt", "uri": "clip.wav"}],
        },
        default_sample_id="audio:0",
        base_dir=str(tmp_path),
    )

    assert record["response"] == "The answer is A."
    assert record["media_refs"][0].uri == str(audio)


def test_track_builder_routes_prompt_audio_and_masks_only_response() -> None:
    builder, stage = _builder()
    records = [
        {
            "sample_id": "audio:0",
            "prompt": "What sound is this?",
            "response": "The answer is A.",
            "media_refs": [{"modality": "audio", "role": "prompt", "uri": "/tmp/clip.wav"}],
        },
        {
            "sample_id": "audio:pad",
            "prompt": "Padding row",
            "response": "The answer is B.",
            "_eval_pad": True,
        },
    ]

    assert builder._embed_prompts(records) == "audio-conditions"
    assert stage.received is not None
    media_refs = stage.received[0].content
    assert isinstance(media_refs, MediaRefs)
    assert media_refs.rows[0][0].modality == "audio"
    assert media_refs.rows[1] == []

    tokens, masks = builder._tokenize_responses(records)
    assert tokens[0][-1].item() == _Tokenizer.eos_token_id
    assert torch.all(masks[0] == 1)
    assert torch.all(masks[1] == 0)


def test_pyav_decodes_wav_for_qwen3_omni(tmp_path: Path) -> None:
    pytest.importorskip("av")
    path = tmp_path / "tone.wav"
    samples = (np.sin(np.linspace(0, 20, 1600)) * 10000).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(samples.tobytes())

    decoded = load_qwen3_audio(str(path), 16000)
    assert decoded is not None
    waveform, sample_rate = decoded
    assert waveform.dtype == np.float32
    assert waveform.size > 0
    assert sample_rate == 16000
