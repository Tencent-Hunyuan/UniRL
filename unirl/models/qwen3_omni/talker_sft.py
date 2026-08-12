"""Supervised track builder for Qwen3-Omni Talker TTS SFT."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence, Tuple

import torch

from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.distributed.group.remote import Remote
from unirl.models.qwen3_omni.talker_conditions import Qwen3OmniTalkerConditions
from unirl.models.qwen3_omni.talker_contract import NUM_CODE_GROUPS
from unirl.models.qwen3_omni.talker_data import (
    CODEC_DATA_SCHEMA,
    assert_fingerprint,
    fingerprint_sha,
    model_fingerprint,
)
from unirl.models.qwen3_omni.talker_pipeline import tokenize_tts_batch
from unirl.types.sample import Part
from unirl.types.segments import TextSegment

logger = logging.getLogger(__name__)

Record = Dict[str, Any]


def _sample_ids(records: Sequence[Record]) -> List[str]:
    return [str(r.get("sample_id", f"sft:{i}")) for i, r in enumerate(records)]


def _pad_flags(records: Sequence[Record]) -> List[bool]:
    return [bool(r.get("_eval_pad", False)) for r in records]


def _record_text(record: Record) -> str:
    meta = record.get("metadata") or {}
    if not isinstance(meta, dict):
        raise TypeError(f"Talker SFT record {record.get('sample_id')!r} metadata must be a dict")
    normalized = meta.get("normalized_transcript")
    if not isinstance(normalized, str) or not normalized.strip():
        raise ValueError(
            f"Talker SFT record {record.get('sample_id')!r} missing metadata.normalized_transcript; "
            "run prepare_talker_tts_data."
        )
    prompt = record.get("prompt")
    if prompt is not None and str(prompt) != normalized:
        raise ValueError(f"Talker SFT record {record.get('sample_id')!r} prompt differs from normalized_transcript.")
    return normalized


def _record_speaker(record: Record, default: str) -> str:
    meta = record.get("metadata") or {}
    if isinstance(meta, dict) and meta.get("speaker"):
        return str(meta["speaker"])
    if record.get("speaker"):
        return str(record["speaker"])
    return default


def _record_codes(record: Record) -> Tuple[torch.Tensor, int]:
    """Return an exact ``([16, T], valid_length)`` offline Mimi target."""
    meta = record.get("metadata") or {}
    if not isinstance(meta, dict):
        raise TypeError(f"Talker SFT record {record.get('sample_id')!r} metadata must be a dict")
    codes = record.get("audio_codes")
    if codes is None:
        codes = meta.get("audio_codes")
    if codes is None:
        raise ValueError(
            f"Talker SFT record {record.get('sample_id')!r} missing audio_codes "
            "(run prepare_talker_tts_data to cache Mimi codes)."
        )
    t = torch.as_tensor(codes, dtype=torch.long)
    if t.dim() != 2 or t.shape[0] != NUM_CODE_GROUPS:
        raise ValueError(
            f"Talker SFT record {record.get('sample_id')!r}: audio_codes expected "
            f"[{NUM_CODE_GROUPS}, T], got {tuple(t.shape)}"
        )
    length = meta.get("audio_code_length", record.get("audio_code_length"))
    if not isinstance(length, int) or isinstance(length, bool):
        raise ValueError(f"Talker SFT record {record.get('sample_id')!r} missing integer metadata.audio_code_length")
    if length < 1 or length > t.shape[1]:
        raise ValueError(
            f"Talker SFT record {record.get('sample_id')!r}: audio_code_length={length} is outside [1, {t.shape[1]}]."
        )
    return t, length


class TalkerSupervisedTrackBuilder(Remote):
    """JSONL TTS records → Talker training ``Part`` (layer0 TextSegment + residual conditions).

    Expected record fields (normalized by ``SupervisedDataSource``):
    - ``prompt``: text to speak
    - ``metadata.speaker`` / ``speaker``
    - ``metadata.audio_codes`` or ``audio_codes``: ``[16, T]`` Mimi codes
    """

    def __init__(
        self,
        *,
        pipeline: Any,
        max_response_length: int = 4096,
        max_prompt_length: int = 4096,
        append_codec_eos: bool = True,
        expected_codec_fingerprint: str,
        expected_model_fingerprint: str = "",
        strict_fingerprints: bool = True,
    ) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.max_response_length = int(max_response_length)
        self.max_prompt_length = int(max_prompt_length)
        self.append_codec_eos = bool(append_codec_eos)
        self.strict_fingerprints = bool(strict_fingerprints)
        if not getattr(pipeline.bundle, "enable_talker", False):
            raise ValueError("TalkerSupervisedTrackBuilder requires pipeline.bundle.enable_talker=True")
        if self.max_response_length <= int(self.append_codec_eos):
            raise ValueError("max_response_length leaves no room for an audio-code target")
        if self.strict_fingerprints and not str(expected_codec_fingerprint).strip():
            raise ValueError(
                "TalkerSupervisedTrackBuilder requires expected_codec_fingerprint in strict mode; "
                "copy the digest printed by prepare_talker_tts_data."
            )
        self.expected_codec_fingerprint = (
            fingerprint_sha(expected_codec_fingerprint, field_name="expected_codec_fingerprint")
            if str(expected_codec_fingerprint).strip()
            else ""
        )
        if str(expected_model_fingerprint).strip():
            self.expected_model_fingerprint = fingerprint_sha(
                expected_model_fingerprint,
                field_name="expected_model_fingerprint",
            )
        else:
            self.expected_model_fingerprint = model_fingerprint(
                pipeline.bundle.pretrained_path,
                kind="qwen3_omni_talker",
            )["sha256"]

    def _validate_record(self, record: Record) -> None:
        if not self.strict_fingerprints:
            return
        sample_id = str(record.get("sample_id", "<unknown>"))
        meta = record.get("metadata") or {}
        if meta.get("codec_data_schema") != CODEC_DATA_SCHEMA:
            raise ValueError(
                f"Talker SFT record {sample_id!r} has codec_data_schema={meta.get('codec_data_schema')!r}; "
                f"expected {CODEC_DATA_SCHEMA!r}. Online/legacy codec targets are not accepted."
            )
        assert_fingerprint(
            meta.get("talker_model_fingerprint"),
            self.expected_model_fingerprint,
            field_name="talker_model_fingerprint",
            sample_id=sample_id,
        )
        assert_fingerprint(
            meta.get("codec_fingerprint"),
            self.expected_codec_fingerprint,
            field_name="codec_fingerprint",
            sample_id=sample_id,
        )
        for field_name in ("speaker", "language", "normalized_transcript"):
            if not isinstance(meta.get(field_name), str) or not meta[field_name].strip():
                raise ValueError(f"Talker SFT record {sample_id!r} missing metadata.{field_name}")

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def build(self, records: List[Record]) -> Part:
        if not records:
            raise ValueError("TalkerSupervisedTrackBuilder.build: empty shard")
        bundle = self.pipeline.bundle
        omni = bundle.omni
        device = bundle.device
        codec_eos = int(omni.config.talker_config.codec_eos_token_id)
        codebook_size = int(omni.config.code2wav_config.codebook_size)
        default_speaker = bundle.default_speaker

        for record in records:
            self._validate_record(record)
        texts = [_record_text(r) for r in records]
        speakers = [_record_speaker(r, default_speaker) for r in records]
        prompt = tokenize_tts_batch(
            bundle,
            texts,
            system_instruction=getattr(self.pipeline, "system_instruction", None),
            max_length=self.max_prompt_length,
        )

        tokens: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []
        residuals: List[torch.Tensor] = []
        mtp_masks: List[torch.Tensor] = []
        for r, is_pad in zip(records, _pad_flags(records)):
            codes, valid_length = _record_codes(r)
            t = min(valid_length, self.max_response_length - int(self.append_codec_eos))
            codes = codes[:, :t].contiguous()
            if int(codes.min().item()) < 0 or int(codes.max().item()) >= codebook_size:
                raise ValueError(
                    f"Talker SFT record {r.get('sample_id')!r}: Mimi code outside [0, {codebook_size - 1}]."
                )
            layer0 = codes[0].tolist()
            if self.append_codec_eos:
                layer0 = layer0 + [codec_eos]
                residual = torch.cat(
                    [codes[1:, :], torch.zeros((NUM_CODE_GROUPS - 1, 1), dtype=torch.long)],
                    dim=1,
                )
                mtp_mask = torch.cat([torch.ones(t, dtype=torch.float32), torch.zeros(1, dtype=torch.float32)])
            else:
                residual = codes[1:, :].contiguous()
                mtp_mask = torch.ones(t, dtype=torch.float32)
            tokens.append(torch.tensor(layer0, dtype=torch.long, device=device))
            fill = 0.0 if is_pad else 1.0
            masks.append(torch.full((len(layer0),), fill, dtype=torch.float32, device=device))
            residuals.append(residual.to(device=device))
            mtp_masks.append((mtp_mask * fill).to(device=device))

        conds = Qwen3OmniTalkerConditions(
            prompt=prompt,
            speakers=speakers,
            residual_codes=residuals,
            mtp_loss_masks=mtp_masks,
        )
        segment = TextSegment.pack(tokens=tokens, loss_mask=masks)
        return Part(
            sample_ids=_sample_ids(records),
            conditions=conds.to_dict(),
            segment=segment,
            metadata=[dict(record.get("metadata") or {}) for record in records],
        )


__all__ = ["TalkerSupervisedTrackBuilder"]
