"""Direct-TTS Talker prefix construction independent of private oracle APIs."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch


def resolve_speaker_id(omni: Any, speaker_name: str) -> int:
    """Map a speaker display name to the discrete Talker speaker token id."""
    speaker_map = getattr(omni.config.talker_config, "speaker_id", None) or {}
    speaker_id = speaker_map.get(str(speaker_name).lower())
    if speaker_id is None:
        known = sorted(speaker_map.keys())
        raise ValueError(f"Speaker {speaker_name!r} not in talker_config.speaker_id; known={known}")
    return int(speaker_id)


def build_talker_prefix_tts(
    model: Any,
    *,
    input_ids: torch.Tensor,
    device: torch.device,
    thinker_outputs: Any = None,
    speaker_name: Optional[str] = None,
    speaker_id: Optional[int] = None,
    expected_prefix_ids: Optional[torch.Tensor] = None,
    batch_idx: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build Talker prefix embeds/ids for one TTS sample.

    Returns
    -------
    talker_input_embed : [1, P, H]
    talker_input_ids : [1, P]
    trailing_text_hidden : [1, T_text, H]
    tts_pad_embed : [1, 1, H]
    """
    config = model.config
    talker = model.talker
    sample_input_ids = input_ids[batch_idx : batch_idx + 1].to(device)
    provider = getattr(model, "input_embedding_provider", None)
    if provider is None:
        thinker = getattr(model, "thinker", None)
        get_embeddings = getattr(thinker, "get_input_embeddings", None)
        if not callable(get_embeddings):
            raise TypeError(
                "build_talker_prefix_tts requires an input_embedding_provider "
                "or a legacy thinker.get_input_embeddings() accessor"
            )
        provider = get_embeddings()
        if hasattr(provider, "base_layer"):
            provider = provider.base_layer
    if thinker_outputs is not None:
        # Compatibility/oracle path. hidden_states[0] is the input embedding,
        # not a transformed Thinker hidden state.
        thinker_embed = thinker_outputs.hidden_states[0][batch_idx : batch_idx + 1].to(device)
    else:
        # Direct TTS performs exactly one frozen embedding lookup. It does not
        # execute the Thinker decoder, vision tower, or audio tower.
        with torch.no_grad():
            thinker_embed = provider(sample_input_ids).to(device)

    im_start_positions = torch.nonzero(sample_input_ids[0] == config.im_start_token_id).view(-1)
    im_start_indexes = torch.cat(
        (im_start_positions, torch.tensor([sample_input_ids.shape[1]], device=device, dtype=im_start_positions.dtype)),
        dim=0,
    )

    talker_special_tokens = torch.tensor(
        [[config.tts_bos_token_id, config.tts_eos_token_id, config.tts_pad_token_id]],
        device=device,
        dtype=sample_input_ids.dtype,
    )

    tts_bos_embed, tts_eos_embed, tts_pad_embed = (
        talker.text_projection(provider(talker_special_tokens))
        .to(device)
        .chunk(3, dim=1)
    )

    if speaker_id is None:
        if speaker_name is None:
            raise ValueError("build_talker_prefix_tts requires speaker_id or speaker_name")
        speaker_id = resolve_speaker_id(model, speaker_name)
    speaker_id = int(speaker_id)

    talker_input_embeds = []
    talker_input_ids = []
    trailing_text_hidden = None

    for i in range(len(im_start_indexes) - 1):
        im_start_index = im_start_indexes[i]
        segment_end_index = im_start_indexes[i + 1]
        role_token = sample_input_ids[0][im_start_index + 1]

        if role_token == config.system_token_id:
            continue
        if role_token == config.user_token_id:
            # Direct TTS is text-only: every user token follows the official
            # text_projection(Thinker input embedding) branch.
            user_part = talker.text_projection(
                thinker_embed[:, im_start_index:segment_end_index]
            ).to(device)
            talker_input_embeds.append(user_part.to(dtype=talker.dtype))
            talker_input_ids.append(sample_input_ids[:, im_start_index:segment_end_index])
        elif role_token == config.assistant_token_id and i == len(im_start_indexes) - 2:
            assistant_hidden = talker.text_projection(
                thinker_embed[:, im_start_index:segment_end_index]
            ).to(device)
            if assistant_hidden.shape[1] < 4:
                raise RuntimeError(
                    "build_talker_prefix_tts: final assistant span must contain at least "
                    "<im_start>, role, newline, and one content token"
                )
            assistant_text_hidden = torch.cat(
                (
                    assistant_hidden[:, :3],
                    tts_pad_embed.expand(-1, 4, -1),
                    tts_bos_embed,
                    assistant_hidden[:, 3:4],
                ),
                dim=1,
            )
            codec_special_tokens = torch.tensor(
                [
                    [
                        config.talker_config.codec_nothink_id,
                        config.talker_config.codec_think_bos_id,
                        config.talker_config.codec_think_eos_id,
                        speaker_id,
                        config.talker_config.codec_pad_id,
                        config.talker_config.codec_bos_id,
                    ]
                ],
                dtype=torch.long,
                device=device,
            )
            assistant_codec_hidden = torch.cat(
                (
                    torch.zeros(
                        (1, 3, config.talker_config.text_config.hidden_size),
                        device=device,
                        dtype=talker.dtype,
                    ),
                    talker.get_input_embeddings()(codec_special_tokens).to(device),
                ),
                dim=1,
            )
            assistant_embeds = assistant_text_hidden + assistant_codec_hidden
            assistant_ids = torch.full(
                (1, assistant_text_hidden.shape[1]),
                fill_value=config.tts_pad_token_id,
                dtype=torch.long,
                device=device,
            )
            trailing_text_hidden = torch.cat((assistant_hidden[:, 4:], tts_eos_embed), dim=1)
            talker_input_embeds.append(assistant_embeds)
            talker_input_ids.append(assistant_ids)
        elif role_token == config.assistant_token_id:
            continue
        else:
            raise AssertionError(
                f"Expect role id after <|im_start|> (assistant/user/system), got token={int(role_token)}"
            )

    if trailing_text_hidden is None:
        raise RuntimeError("build_talker_prefix_tts: failed to build trailing_text_hidden (missing assistant span)")

    talker_input_embed = torch.cat([embed.to(device) for embed in talker_input_embeds], dim=1)
    talker_input_id = torch.cat([ids.to(device) for ids in talker_input_ids], dim=1)
    if expected_prefix_ids is not None:
        expected = torch.as_tensor(
            expected_prefix_ids,
            device=device,
            dtype=talker_input_id.dtype,
        )
        if expected.numel() != talker_input_id.numel():
            raise ValueError(
                "build_talker_prefix_tts: replay prefix ID length "
                f"{expected.numel()} != reconstructed {talker_input_id.numel()}"
            )
        expected = expected.view_as(talker_input_id)
        if not torch.equal(talker_input_id, expected):
            raise ValueError(
                "build_talker_prefix_tts: reconstructed prefix IDs differ from "
                "the replay trajectory"
            )
    return talker_input_embed, talker_input_id, trailing_text_hidden.to(device), tts_pad_embed


__all__ = ["build_talker_prefix_tts", "resolve_speaker_id"]
