"""Evaluate cached-code reconstruction ceiling and generated Talker speech.

No metric has a heuristic fallback. Missing ASR, speaker, or MOS models either
fail immediately (``--missing_metric_policy fail``) or are represented as
``{"status": "unavailable", ...}`` in the output JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from unirl.models.qwen3_omni.config import Qwen3OmniPipelineConfig
from unirl.models.qwen3_omni.talker_bundle import Qwen3OmniTalkerBundle
from unirl.models.qwen3_omni.talker_conditions import Qwen3OmniTalkerConditions
from unirl.models.qwen3_omni.talker_contract import AUDIO_SAMPLE_RATE, NUM_CODE_GROUPS
from unirl.models.qwen3_omni.talker_data import assert_fingerprint, fingerprint_sha, model_fingerprint
from unirl.models.qwen3_omni.talker_pipeline import Qwen3OmniTalkerPipeline
from unirl.reward.local.tts_wer import word_error_rate
from unirl.types.primitives import Texts
from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams


def _char_error_rate(reference: str, hypothesis: str) -> float:
    ref = list(reference)
    hyp = list(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(hyp) + 1))
    for i, ref_char in enumerate(ref, 1):
        current = [i]
        for j, hyp_char in enumerate(hyp, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (ref_char != hyp_char),
                )
            )
        previous = current
    return float(previous[-1]) / float(len(ref))


def _load_audio(path: str) -> tuple[torch.Tensor, int]:
    import soundfile as sf

    values, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(values).mean(dim=1), int(sample_rate)


def _load_unirl_lora_checkpoint(bundle: Qwen3OmniTalkerBundle, checkpoint_path: str) -> Dict[str, Any]:
    from unirl.train.lora import inject_lora

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"UniRL checkpoint must be a dict, got {type(checkpoint).__name__}")
    policy = checkpoint.get("policy_state_dict")
    config = checkpoint.get("lora_config")
    if not isinstance(policy, dict) or not policy:
        raise ValueError("UniRL checkpoint has no non-empty policy_state_dict")
    if not isinstance(config, dict):
        raise ValueError("UniRL checkpoint has no lora_config")
    inject_lora(
        bundle.talker,
        rank=int(config["rank"]),
        alpha=int(config["alpha"]),
        target_modules=config["target_modules"],
        exclude_modules=config.get("exclude_modules"),
        dropout=float(config.get("dropout", 0.0)),
        bias=str(config.get("bias", "none")),
        task_type=str(config.get("task_type", "FEATURE_EXTRACTION")),
    )
    incompatible = bundle.talker.load_state_dict(policy, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint contains unexpected Talker keys: {incompatible.unexpected_keys[:8]}"
        )
    model_keys = set(bundle.talker.state_dict())
    missing_policy = sorted(set(policy) - model_keys)
    if missing_policy:
        raise RuntimeError(f"Checkpoint keys were not injected into Talker: {missing_policy[:8]}")
    bundle.talker.requires_grad_(False)
    bundle.talker.eval()
    return {
        "path": checkpoint_path,
        "step": int(checkpoint.get("step", -1)),
        "lora_tensors": len(policy),
    }


class _OptionalMetric:
    def __init__(self, name: str, policy: str) -> None:
        self.name = name
        self.policy = policy
        self.reason: Optional[str] = None

    def unavailable(self, reason: str) -> None:
        self.reason = reason
        if self.policy == "fail":
            raise RuntimeError(f"{self.name} unavailable: {reason}")

    @property
    def available(self) -> bool:
        return self.reason is None


class _ASR(_OptionalMetric):
    def __init__(self, model_id: str, *, device: str, policy: str) -> None:
        super().__init__("ASR WER/CER", policy)
        self.pipeline = None
        if not model_id:
            self.unavailable("--asr_model_id was not provided")
            return
        try:
            from transformers import pipeline

            pipeline_device: Any = 0 if device.startswith("cuda") and torch.cuda.is_available() else -1
            self.pipeline = pipeline("automatic-speech-recognition", model=model_id, device=pipeline_device)
        except Exception as exc:
            self.unavailable(f"failed to load {model_id!r}: {exc}")

    def score(self, waveform: torch.Tensor, reference: str) -> Dict[str, Any]:
        if not self.available or self.pipeline is None:
            return {"transcript": None, "wer": None, "cer": None}
        result = self.pipeline({"array": waveform.detach().float().cpu().numpy(), "sampling_rate": AUDIO_SAMPLE_RATE})
        hypothesis = str(result.get("text", "") if isinstance(result, dict) else result).strip()
        return {
            "transcript": hypothesis,
            "wer": word_error_rate(reference, hypothesis),
            "cer": _char_error_rate(reference, hypothesis),
        }


class _SpeakerSIM(_OptionalMetric):
    def __init__(self, model_id: str, *, device: str, policy: str) -> None:
        super().__init__("speaker SIM", policy)
        self.device = device
        self.processor = self.model = None
        if not model_id:
            self.unavailable("--speaker_model_id was not provided")
            return
        try:
            from transformers import AutoFeatureExtractor, AutoModel

            self.processor = AutoFeatureExtractor.from_pretrained(model_id)
            self.model = AutoModel.from_pretrained(model_id).to(device).eval()
        except Exception as exc:
            self.unavailable(f"failed to load {model_id!r}: {exc}")

    def _embed(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        assert self.processor is not None and self.model is not None
        inputs = self.processor(
            waveform.detach().float().cpu().numpy(),
            sampling_rate=int(sample_rate),
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            hidden = self.model(**inputs).last_hidden_state.mean(dim=1).squeeze(0)
        return torch.nn.functional.normalize(hidden.float(), dim=-1)

    def score(self, waveform: torch.Tensor, reference_path: Optional[str]) -> Optional[float]:
        if not self.available or not reference_path:
            return None
        reference, reference_rate = _load_audio(reference_path)
        generated = self._embed(waveform, AUDIO_SAMPLE_RATE)
        target = self._embed(reference, reference_rate)
        return float(torch.dot(generated, target).clamp(-1.0, 1.0).item())


class _MOS(_OptionalMetric):
    def __init__(self, policy: str) -> None:
        super().__init__("UTMOS", policy)
        self.predictor = None
        try:
            import utmos  # type: ignore

            self.predictor = utmos.Score()
        except Exception as exc:
            self.unavailable(f"utmos package/model failed to load: {exc}")

    def score(self, waveform: torch.Tensor) -> Optional[float]:
        if not self.available or self.predictor is None:
            return None
        return float(self.predictor.score(waveform.detach().float().cpu().numpy(), AUDIO_SAMPLE_RATE))


def _metric_summary(rows: Sequence[Dict[str, Any]], key: str, metric: _OptionalMetric) -> Dict[str, Any]:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not metric.available:
        return {"status": "unavailable", "reason": metric.reason, "n": 0, "mean": None}
    if not values:
        return {"status": "unavailable", "reason": f"no rows had inputs for {key}", "n": 0, "mean": None}
    return {"status": "available", "n": len(values), "mean": sum(values) / len(values)}


def _read_rows(path: str, max_samples: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise TypeError(f"{path}:{line_number}: expected a JSON object")
            rows.append(item)
            if max_samples >= 0 and len(rows) >= max_samples:
                break
    if not rows:
        raise ValueError(f"No eval rows found in {path}")
    return rows


def _validate_manifest_fingerprints(
    rows: Sequence[Dict[str, Any]],
    *,
    model_path: str,
    expected_codec_fingerprint: str,
) -> tuple[Dict[str, Any], str]:
    model_fp = model_fingerprint(model_path, kind="qwen3_omni_talker")
    first_meta = rows[0].get("metadata") or {}
    codec_sha = (
        fingerprint_sha(expected_codec_fingerprint, field_name="expected_codec_fingerprint")
        if expected_codec_fingerprint
        else fingerprint_sha(first_meta.get("codec_fingerprint"), field_name="codec_fingerprint")
    )
    for index, row in enumerate(rows):
        metadata = row.get("metadata") or {}
        sample_id = str(row.get("sample_id", index))
        assert_fingerprint(
            metadata.get("talker_model_fingerprint"),
            model_fp,
            field_name="talker_model_fingerprint",
            sample_id=sample_id,
        )
        assert_fingerprint(
            metadata.get("codec_fingerprint"),
            codec_sha,
            field_name="codec_fingerprint",
            sample_id=sample_id,
        )
    return model_fp, codec_sha


def _score_audio(
    waveform: torch.Tensor,
    *,
    transcript: str,
    reference_path: Optional[str],
    asr: _ASR,
    speaker: _SpeakerSIM,
    mos: _MOS,
) -> Dict[str, Any]:
    if waveform.numel() == 0 or not torch.isfinite(waveform).all():
        raise ValueError("decoded waveform is empty or non-finite")
    asr_scores = asr.score(waveform, transcript)
    return {
        **asr_scores,
        "sim": speaker.score(waveform, reference_path),
        "mos": mos.score(waveform),
        "duration_s": float(waveform.numel()) / AUDIO_SAMPLE_RATE,
    }


def _aggregate(mode_rows: Sequence[Dict[str, Any]], *, asr: _ASR, speaker: _SpeakerSIM, mos: _MOS) -> Dict[str, Any]:
    return {
        "wer": _metric_summary(mode_rows, "wer", asr),
        "cer": _metric_summary(mode_rows, "cer", asr),
        "sim": _metric_summary(mode_rows, "sim", speaker),
        "mos": _metric_summary(mode_rows, "mos", mos),
        "decode_failure_rate": sum(bool(row["decode_failure"]) for row in mode_rows) / len(mode_rows),
    }


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--checkpoint", default="", help="Optional UniRL Talker LoRA checkpoint.pt")
    parser.add_argument("--eval_jsonl", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--audio_out_dir", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--asr_model_id", default="")
    parser.add_argument("--speaker_model_id", default="")
    parser.add_argument("--expected_codec_fingerprint", default="")
    parser.add_argument("--missing_metric_policy", choices=("fail", "unavailable"), default="unavailable")
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args(argv)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    rows = _read_rows(args.eval_jsonl, args.max_samples)
    model_fp, codec_sha = _validate_manifest_fingerprints(
        rows,
        model_path=args.model_path,
        expected_codec_fingerprint=args.expected_codec_fingerprint,
    )
    asr = _ASR(args.asr_model_id, device=args.device, policy=args.missing_metric_policy)
    speaker_metric = _SpeakerSIM(
        args.speaker_model_id,
        device=args.device,
        policy=args.missing_metric_policy,
    )
    mos = _MOS(args.missing_metric_policy)

    config = Qwen3OmniPipelineConfig(
        pretrained_model_ckpt_path=args.model_path,
        model_precision="bf16",
        device=args.device,
        enable_talker=True,
    )
    bundle = Qwen3OmniTalkerBundle.from_config(config)
    checkpoint_info = (
        _load_unirl_lora_checkpoint(bundle, args.checkpoint)
        if args.checkpoint
        else None
    )
    pipeline = Qwen3OmniTalkerPipeline.from_bundle(bundle, decode_audio=True)
    codec_eos = int(bundle.config.talker_config.codec_eos_token_id)

    reconstruction_rows: List[Dict[str, Any]] = []
    generation_rows: List[Dict[str, Any]] = []
    audio_out_dir = Path(args.audio_out_dir) if args.audio_out_dir else None
    if audio_out_dir is not None:
        audio_out_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows):
        sample_id = str(row.get("sample_id", index))
        metadata = row.get("metadata") or {}
        transcript = str(metadata["normalized_transcript"])
        speaker_name = str(metadata["speaker"])
        reference_path = metadata.get("audio_path")

        reconstruction: Dict[str, Any] = {"sample_id": sample_id, "decode_failure": False}
        try:
            codes = torch.as_tensor(metadata["audio_codes"], dtype=torch.long)
            length = int(metadata["audio_code_length"])
            if codes.ndim != 2 or codes.shape[0] != NUM_CODE_GROUPS or not 0 < length <= codes.shape[1]:
                raise ValueError(f"invalid cached codes shape/length: {tuple(codes.shape)}, {length}")
            waveform = pipeline.ar.decode_codes_to_audio(
                layer0_codes=[codes[0, :length]],
                residual_codes=[codes[1:, :length]],
            )[0]
            reconstruction.update(
                _score_audio(
                    waveform,
                    transcript=transcript,
                    reference_path=reference_path,
                    asr=asr,
                    speaker=speaker_metric,
                    mos=mos,
                )
            )
            if audio_out_dir is not None:
                import soundfile as sf

                audio_path = audio_out_dir / f"{sample_id}.reconstruction.wav"
                sf.write(audio_path, waveform.detach().float().cpu().numpy(), AUDIO_SAMPLE_RATE)
                reconstruction["audio_path"] = str(audio_path)
        except Exception as exc:
            reconstruction.update(
                {
                    "decode_failure": True,
                    "failure": f"{type(exc).__name__}: {exc}",
                    "wer": None,
                    "cer": None,
                    "sim": None,
                    "mos": None,
                }
            )
        reconstruction_rows.append(reconstruction)

        generated: Dict[str, Any] = {
            "sample_id": sample_id,
            "decode_failure": False,
            "eos_failure": False,
        }
        try:
            request = Sample(
                parts=[
                    Part.input(
                        [sample_id],
                        primitives={"text": Texts(texts=[transcript])},
                        metadata=[{"speaker": speaker_name}],
                    )
                ]
            ).fork(
                1,
                sampling_params=ARSamplingParams(
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                ),
            )
            output = pipeline.generate(request)
            frontier = output.frontier_gen_part(ARSamplingParams)
            if frontier.segment is None or frontier.segment.tokens is None:
                raise ValueError("generation did not return layer-0 codec tokens")
            tokens = frontier.segment.tokens
            generated["num_codec_steps"] = int(tokens.numel())
            generated["eos_failure"] = not bool(tokens.numel() and int(tokens[-1].item()) == codec_eos)
            conditions = Qwen3OmniTalkerConditions.from_dict(frontier.conditions)
            if conditions.residual_codes is None:
                raise ValueError("generation did not return MTP residual codes")
            audio = frontier.primitives.get("audio")
            if audio is None:
                raise ValueError("generation did not return decoded audio")
            audio_items = audio.to_list()
            if not audio_items:
                raise ValueError("generation returned an empty audio batch")
            waveform = audio_items[0].waveform
            generated.update(
                _score_audio(
                    waveform,
                    transcript=transcript,
                    reference_path=reference_path,
                    asr=asr,
                    speaker=speaker_metric,
                    mos=mos,
                )
            )
            if audio_out_dir is not None:
                import soundfile as sf

                audio_path = audio_out_dir / f"{sample_id}.generated.wav"
                sf.write(audio_path, waveform.detach().float().cpu().numpy(), AUDIO_SAMPLE_RATE)
                generated["audio_path"] = str(audio_path)
        except Exception as exc:
            if "num_codec_steps" not in generated:
                generated["eos_failure"] = True
            generated.update(
                {
                    "decode_failure": True,
                    "failure": f"{type(exc).__name__}: {exc}",
                    "wer": None,
                    "cer": None,
                    "sim": None,
                    "mos": None,
                }
            )
        generation_rows.append(generated)

    result = {
        "n": len(rows),
        "model_fingerprint": model_fp,
        "codec_fingerprint": codec_sha,
        "checkpoint": checkpoint_info,
        "reconstruction_ceiling": {
            "metrics": _aggregate(reconstruction_rows, asr=asr, speaker=speaker_metric, mos=mos),
            "rows": reconstruction_rows,
        },
        "generation": {
            "metrics": {
                **_aggregate(generation_rows, asr=asr, speaker=speaker_metric, mos=mos),
                "eos_failure_rate": sum(bool(row["eos_failure"]) for row in generation_rows) / len(generation_rows),
            },
            "rows": generation_rows,
        },
    }
    output_path = Path(args.out_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    try:
        tmp_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    print(json.dumps({"n": len(rows), "output": str(output_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
