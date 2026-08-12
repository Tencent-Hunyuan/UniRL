"""Adapters around the official Qwen3-Omni implementation used for parity tests.

This module deliberately keeps oracle calls separate from production helpers:
tests compare independently constructed prefixes/mixes/decodes instead of
having both sides call the same implementation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch


@dataclass(frozen=True)
class OracleTalkerPrefix:
    inputs_embeds: torch.Tensor
    input_ids: torch.Tensor
    trailing_text_hidden: torch.Tensor
    tts_pad_embed: torch.Tensor


def checkpoint_skip_reason(path: Optional[str]) -> Optional[str]:
    """Return why ``path`` cannot be used as a real Omni oracle checkpoint."""
    if not path:
        return "QWEN3_OMNI_PATH is not set"
    root = Path(path)
    config_path = root / "config.json"
    if not config_path.is_file():
        return f"{config_path} is missing"
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return f"cannot read {config_path}: {exc}"
    if not config.get("enable_audio_output", False):
        return "checkpoint config has enable_audio_output=false"
    if "talker_config" not in config or "code2wav_config" not in config:
        return "checkpoint config does not contain Talker and Code2Wav configs"
    has_weights = (root / "model.safetensors").is_file() or (root / "model.safetensors.index.json").is_file()
    has_weights = has_weights or any(root.glob("model-*.safetensors"))
    if not has_weights:
        return f"{root} has config assets but no model safetensors"
    return None


def build_oracle_talker_prefix(
    omni: Any,
    *,
    thinker_outputs: Any,
    input_ids: torch.Tensor,
    speaker_name: str,
    device: torch.device,
    batch_idx: int = 0,
) -> OracleTalkerPrefix:
    """Build a text-only prefix through the official private oracle methods."""
    config = omni.config
    sample_ids = input_ids[batch_idx : batch_idx + 1].to(device)
    thinker_embed = thinker_outputs.hidden_states[0][batch_idx : batch_idx + 1].to(device)
    accept_layer = int(config.talker_config.accept_hidden_layer)
    thinker_hidden = thinker_outputs.hidden_states[accept_layer][batch_idx : batch_idx + 1].to(device)
    multimodal_mask = torch.zeros_like(sample_ids, dtype=torch.bool, device=device)

    starts = torch.nonzero(sample_ids[0] == config.im_start_token_id).view(-1)
    starts = torch.cat((starts, torch.tensor([sample_ids.shape[1]], dtype=starts.dtype, device=device)))
    special = torch.tensor(
        [[config.tts_bos_token_id, config.tts_eos_token_id, config.tts_pad_token_id]],
        dtype=sample_ids.dtype,
        device=device,
    )
    thinker_embeddings = omni.thinker.get_input_embeddings()
    if hasattr(thinker_embeddings, "base_layer"):
        thinker_embeddings = thinker_embeddings.base_layer
    tts_bos, tts_eos, tts_pad = (
        omni.talker.text_projection(thinker_embeddings(special)).to(device).chunk(3, dim=1)
    )
    speaker_id = config.talker_config.speaker_id.get(str(speaker_name).lower())
    if speaker_id is None:
        raise ValueError(f"unknown oracle speaker {speaker_name!r}")

    embed_parts = []
    id_parts = []
    trailing = None
    for index in range(len(starts) - 1):
        start = starts[index]
        end = starts[index + 1]
        role = sample_ids[0, start + 1]
        if role == config.system_token_id:
            continue
        if role == config.user_token_id:
            embed_parts.append(
                omni._get_talker_user_parts(
                    start,
                    end,
                    multimodal_mask,
                    thinker_hidden,
                    thinker_embed,
                )
            )
            id_parts.append(sample_ids[:, start:end])
        elif role == config.assistant_token_id and index == len(starts) - 2:
            assistant_embed, assistant_ids, trailing = omni._get_talker_assistant_parts(
                start,
                end,
                int(speaker_id),
                thinker_embed,
                tts_pad,
                tts_bos,
                tts_eos,
            )
            embed_parts.append(assistant_embed)
            id_parts.append(assistant_ids)
        elif role != config.assistant_token_id:
            raise AssertionError(f"unexpected role token {int(role)}")
    if trailing is None:
        raise RuntimeError("official oracle did not find the final assistant span")
    return OracleTalkerPrefix(
        inputs_embeds=torch.cat(embed_parts, dim=1),
        input_ids=torch.cat(id_parts, dim=1),
        trailing_text_hidden=trailing,
        tts_pad_embed=tts_pad,
    )


def oracle_prepare_residual_step(
    talker: Any,
    *,
    layer0_ids: torch.Tensor,
    hidden_states: Any,
    generation_step: int,
    trailing_text_hidden: torch.Tensor,
    tts_pad_embed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Invoke the official ``prepare_inputs_for_generation`` residual branch."""
    layer0_ids = layer0_ids.to(dtype=torch.long).view(-1, 1)
    prepared = talker.prepare_inputs_for_generation(
        layer0_ids,
        attention_mask=torch.ones_like(layer0_ids),
        is_first_iteration=False,
        use_cache=True,
        hidden_states=hidden_states,
        generation_step=int(generation_step),
        trailing_text_hidden=trailing_text_hidden,
        tts_pad_embed=tts_pad_embed,
    )
    return prepared["inputs_embeds"], prepared["residual_codes"]


def manual_chunked_decode(
    code2wav: Any,
    codes: torch.Tensor,
    *,
    chunk_size: int = 300,
    left_context_size: int = 25,
) -> torch.Tensor:
    """Independent transcription of the official chunked decoder."""
    chunks = []
    start = 0
    while start < codes.shape[-1]:
        end = min(start + int(chunk_size), codes.shape[-1])
        context = int(left_context_size) if start - int(left_context_size) > 0 else start
        wav = code2wav(codes[..., start - context : end])
        chunks.append(wav[..., context * int(code2wav.total_upsample) :])
        start = end
    if not chunks:
        return torch.empty((*codes.shape[:-2], 1, 0), dtype=torch.float32, device=codes.device)
    return torch.cat(chunks, dim=-1)


__all__ = [
    "OracleTalkerPrefix",
    "build_oracle_talker_prefix",
    "checkpoint_skip_reason",
    "manual_chunked_decode",
    "oracle_prepare_residual_step",
]
