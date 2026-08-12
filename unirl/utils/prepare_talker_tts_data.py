"""Build the *offline-only* Mimi target manifest consumed by Talker SFT.

Every emitted SFT row contains mono 24 kHz Mimi codes with exact ``[16, T]``
layout, the valid code length, normalized transcript metadata, and both Talker
checkpoint and codec fingerprints.  Training never receives a waveform and
therefore cannot silently run a different online encoder.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from unirl.models.qwen3_omni.talker_contract import AUDIO_SAMPLE_RATE, NUM_CODE_GROUPS
from unirl.models.qwen3_omni.talker_data import (
    CODEC_DATA_SCHEMA,
    codec_fingerprint,
    model_fingerprint,
)


def normalize_transcript(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    return re.sub(r"\s+", " ", text).strip()


def _load_wav_mono_24k(path: str):
    import soundfile as sf
    import torch
    import torchaudio

    values, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(values.T.copy())
    if waveform.ndim != 2 or waveform.shape[0] < 1:
        raise ValueError(f"Audio {path!r} must decode to [channels, samples], got {tuple(waveform.shape)}")
    waveform = waveform.float().mean(dim=0, keepdim=True)
    if int(sample_rate) != AUDIO_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, int(sample_rate), AUDIO_SAMPLE_RATE)
    waveform = waveform.squeeze(0).contiguous()
    if waveform.numel() == 0 or not torch.isfinite(waveform).all():
        raise ValueError(f"Audio {path!r} is empty or contains non-finite samples")
    return waveform


def _unwrap_audio_codes(encoded: Any):
    import torch

    codes = getattr(encoded, "audio_codes", encoded)
    if isinstance(codes, (tuple, list)):
        if not codes:
            raise ValueError("Mimi encode returned an empty sequence")
        codes = codes[0]
    codes = torch.as_tensor(codes)
    while codes.ndim > 2 and codes.shape[0] == 1:
        codes = codes.squeeze(0)
    if codes.ndim != 2:
        raise ValueError(f"Mimi encode expected singleton batch dimensions then [Q, T], got {tuple(codes.shape)}")
    return codes


def _encode_mimi(wav, *, mimi, feature_extractor, device: str):
    import torch

    values = wav.detach().float().cpu().numpy()
    inputs = feature_extractor(values, sampling_rate=AUDIO_SAMPLE_RATE, return_tensors="pt")
    model_inputs = {
        key: value.to(device)
        for key, value in inputs.items()
        if hasattr(value, "to") and key in {"input_values", "padding_mask"}
    }
    if "padding_mask" not in model_inputs and hasattr(inputs.get("attention_mask"), "to"):
        model_inputs["padding_mask"] = inputs["attention_mask"].to(device)
    if "input_values" not in model_inputs:
        raise ValueError("Mimi feature extractor did not return input_values")
    with torch.no_grad():
        codes = _unwrap_audio_codes(mimi.encode(num_quantizers=NUM_CODE_GROUPS, **model_inputs)).detach().cpu().long()
    if codes.shape[0] != NUM_CODE_GROUPS:
        raise ValueError(
            f"Mimi emitted {codes.shape[0]} quantizers; Talker SFT requires exactly {NUM_CODE_GROUPS}, "
            "and will not truncate/pad incompatible codec targets."
        )
    if codes.shape[1] < 1:
        raise ValueError("Mimi emitted an empty code timeline")
    return codes


def _extract_text_speaker_audio(
    obj: Dict[str, Any],
    *,
    default_speaker: str,
    default_language: str,
) -> Tuple[str, str, str, Optional[str]]:
    metadata = obj.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise TypeError("Input row metadata must be a dictionary")
    speaker = str(obj.get("speaker") or metadata.get("speaker") or default_speaker).strip()
    language = str(obj.get("language") or metadata.get("language") or default_language).strip()
    audio = None
    if obj.get("audios"):
        audio = obj["audios"][0]
    elif obj.get("audio"):
        audio = obj["audio"]
    text = obj.get("text") or obj.get("prompt")
    if text is None and obj.get("messages"):
        for m in obj["messages"]:
            if m.get("role") == "user":
                text = m.get("content")
                break
    if not text:
        raise ValueError(f"Cannot extract text from {obj!r}")
    normalized = normalize_transcript(str(text))
    if not normalized:
        raise ValueError("Transcript is empty after Unicode/whitespace normalization")
    if not speaker or not language:
        raise ValueError("speaker and language must be non-empty")
    return normalized, speaker, language, (str(audio) if audio else None)


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_jsonl", required=True)
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--talker_model", required=True, help="Exact Qwen3-Omni checkpoint used for training")
    p.add_argument("--talker_revision", default=None)
    p.add_argument("--mimi_model", default="kyutai/mimi")
    p.add_argument("--mimi_revision", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max_samples", type=int, default=-1)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--default_speaker", default="Ethan")
    p.add_argument("--default_language", default="und")
    p.add_argument("--rl_prompts_only", action="store_true", help="Skip code encode; emit RL prompts only")
    args = p.parse_args(argv)

    talker_fp = model_fingerprint(
        args.talker_model,
        kind="qwen3_omni_talker",
        revision=args.talker_revision,
    )
    codec_fp = codec_fingerprint(args.mimi_model, revision=args.mimi_revision)
    mimi = feature_extractor = None
    if not args.rl_prompts_only:
        from transformers import AutoFeatureExtractor, MimiModel

        feature_extractor = AutoFeatureExtractor.from_pretrained(
            args.mimi_model,
            revision=args.mimi_revision,
        )
        mimi = (
            MimiModel.from_pretrained(
                args.mimi_model,
                revision=args.mimi_revision,
            )
            .to(args.device)
            .eval()
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)) or ".", exist_ok=True)
    output_path = Path(args.output_jsonl)
    tmp_path = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    n = 0
    try:
        with open(args.input_jsonl, encoding="utf-8") as fin, tmp_path.open("w", encoding="utf-8") as fout:
            for line_number, line in enumerate(fin, 1):
                if not line.strip():
                    continue
                if args.max_samples >= 0 and n >= args.max_samples:
                    break
                try:
                    obj = json.loads(line)
                    text, speaker, language, audio = _extract_text_speaker_audio(
                        obj,
                        default_speaker=args.default_speaker,
                        default_language=args.default_language,
                    )
                    if audio and not os.path.isabs(audio):
                        audio = os.path.abspath(os.path.join(os.path.dirname(args.input_jsonl), audio))
                    meta: Dict[str, Any] = {
                        "codec_data_schema": CODEC_DATA_SCHEMA,
                        "speaker": speaker,
                        "language": language,
                        "normalized_transcript": text,
                        "talker_model_fingerprint": talker_fp,
                        "codec_fingerprint": codec_fp,
                    }
                    if audio:
                        meta["audio_path"] = audio
                    if not args.rl_prompts_only:
                        if not audio:
                            raise ValueError("sample is missing an audio path")
                        wav = _load_wav_mono_24k(audio)
                        codes = _encode_mimi(
                            wav,
                            mimi=mimi,
                            feature_extractor=feature_extractor,
                            device=args.device,
                        )
                        meta["audio_codes"] = codes.tolist()
                        meta["audio_code_length"] = int(codes.shape[1])
                        meta["audio_num_samples_24k"] = int(wav.numel())
                    out = {
                        "sample_id": str(obj.get("sample_id") or f"tts-{n}"),
                        "prompt": text,
                        "metadata": meta,
                    }
                    fout.write(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")
                    n += 1
                    if args.log_every > 0 and n % args.log_every == 0:
                        print(
                            json.dumps(
                                {
                                    "progress_rows": n,
                                    "input": args.input_jsonl,
                                    "output": args.output_jsonl,
                                }
                            ),
                            flush=True,
                        )
                except Exception as exc:
                    raise ValueError(f"{args.input_jsonl}:{line_number}: {exc}") from exc
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    print(
        json.dumps(
            {
                "rows": n,
                "output": str(output_path),
                "talker_model_fingerprint": talker_fp["sha256"],
                "codec_fingerprint": codec_fp["sha256"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
