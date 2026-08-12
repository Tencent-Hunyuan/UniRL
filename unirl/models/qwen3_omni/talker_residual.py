"""Official Qwen3-Omni MTP residual-hidden mixing primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch


@dataclass(frozen=True)
class TalkerResidualStep:
    """Residual codec ids and the mixed hidden used by the next Talker step."""

    residual_codes: torch.Tensor
    next_codec_hidden: torch.Tensor
    log_probs: Optional[torch.Tensor] = None


def _predictor_embeddings(code_predictor: Any) -> Sequence[torch.nn.Module]:
    embeddings = code_predictor.get_input_embeddings()
    if not isinstance(embeddings, (torch.nn.ModuleList, list, tuple)):
        raise TypeError("code_predictor.get_input_embeddings() must return per-codebook embedding modules")
    return embeddings


def mix_residual_hidden(
    *,
    layer0_hidden: torch.Tensor,
    residual_codes: torch.Tensor,
    predictor_hidden_states: Sequence[Any],
    predictor_embeddings: Sequence[torch.nn.Module],
) -> torch.Tensor:
    """Reproduce the official residual mixer in ``prepare_inputs_for_generation``.

    The first term is the sampled layer-0 embedding.  Intermediate terms are
    MTP model hidden states, not embeddings of the corresponding residual ids.
    Only the final residual code uses its embedding table.
    """
    if layer0_hidden.dim() != 3 or layer0_hidden.shape[1] != 1:
        raise ValueError(f"layer0_hidden must be [B, 1, H], got {tuple(layer0_hidden.shape)}")
    if residual_codes.dim() != 2:
        raise ValueError(f"residual_codes must be [B, R], got {tuple(residual_codes.shape)}")
    num_residual = int(residual_codes.shape[1])
    if num_residual < 1:
        raise ValueError("residual_codes must contain at least one residual code")
    if len(predictor_embeddings) < num_residual:
        raise ValueError(f"need {num_residual} residual embedding tables, got {len(predictor_embeddings)}")

    # HF generate returns one hidden-state entry per predictor forward.  The
    # first predicts residual-1 from [Talker hidden, layer0 embedding].  The
    # following R-1 entries expose the hidden for residuals 1..R-1.
    middle_entries = list(predictor_hidden_states[1:])
    if len(middle_entries) != num_residual - 1:
        raise ValueError(
            "predictor_hidden_states must contain one entry per generated residual; "
            f"expected {num_residual}, got {len(predictor_hidden_states)}"
        )

    middle_hiddens = []
    for entry in middle_entries:
        hidden = entry[0] if isinstance(entry, (tuple, list)) else entry
        if hidden.dim() != 3:
            raise ValueError(f"predictor hidden must be rank 3, got {tuple(hidden.shape)}")
        middle_hiddens.append(hidden.to(device=layer0_hidden.device, dtype=layer0_hidden.dtype))

    final_hidden = predictor_embeddings[num_residual - 1](residual_codes[..., -1:]).to(
        device=layer0_hidden.device, dtype=layer0_hidden.dtype
    )
    codec_hiddens = [layer0_hidden, *middle_hiddens, final_hidden]
    return torch.cat(codec_hiddens, dim=1).sum(dim=1, keepdim=True)


def residual_step_from_generate(
    *,
    talker: Any,
    layer0_ids: torch.Tensor,
    predictor_result: Any,
) -> TalkerResidualStep:
    """Convert an MTP ``generate`` result to the official next-step state."""
    layer0_ids = layer0_ids.to(dtype=torch.long).view(-1, 1)
    residual_codes = predictor_result.sequences.to(device=layer0_ids.device, dtype=torch.long)
    layer0_hidden = talker.get_input_embeddings()(layer0_ids)
    next_hidden = mix_residual_hidden(
        layer0_hidden=layer0_hidden,
        residual_codes=residual_codes,
        predictor_hidden_states=predictor_result.hidden_states,
        predictor_embeddings=_predictor_embeddings(talker.code_predictor),
    )
    return TalkerResidualStep(residual_codes=residual_codes, next_codec_hidden=next_hidden)


def predict_residual_step(
    *,
    talker: Any,
    past_hidden: torch.Tensor,
    layer0_ids: torch.Tensor,
) -> TalkerResidualStep:
    """Run the frozen MTP decoder using the exact official generation settings."""
    layer0_ids = layer0_ids.to(device=past_hidden.device, dtype=torch.long).view(-1, 1)
    layer0_hidden = talker.get_input_embeddings()(layer0_ids)
    result = talker.code_predictor.generate(
        inputs_embeds=torch.cat((past_hidden, layer0_hidden), dim=1),
        max_new_tokens=int(talker.config.num_code_groups) - 1,
        do_sample=True,
        top_k=50,
        top_p=0.8,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )
    return residual_step_from_generate(talker=talker, layer0_ids=layer0_ids, predictor_result=result)


def replay_residual_step(
    *,
    talker: Any,
    past_hidden: torch.Tensor,
    layer0_ids: torch.Tensor,
    residual_codes: torch.Tensor,
    return_log_probs: bool = False,
    use_cache: bool = True,
) -> TalkerResidualStep:
    """Teacher-force frozen residual ids while preserving official MTP hiddens."""
    layer0_ids = layer0_ids.to(device=past_hidden.device, dtype=torch.long).view(-1, 1)
    residual_codes = residual_codes.to(device=past_hidden.device, dtype=torch.long)
    if residual_codes.dim() != 2:
        raise ValueError(f"residual_codes must be [B, R], got {tuple(residual_codes.shape)}")
    expected = int(talker.config.num_code_groups) - 1
    if residual_codes.shape[1] != expected:
        raise ValueError(f"expected {expected} residual groups, got {residual_codes.shape[1]}")

    code_predictor = talker.code_predictor
    layer0_hidden = talker.get_input_embeddings()(layer0_ids)
    predictor_hidden_states = []
    token_log_probs = []
    past_key_values = None
    attention_mask = None
    full_inputs_embeds = torch.cat((past_hidden, layer0_hidden), dim=1)
    for index in range(expected):
        if use_cache and index == 0:
            inputs_embeds = full_inputs_embeds
            attention_mask = torch.ones(
                (inputs_embeds.shape[0], inputs_embeds.shape[1]),
                dtype=torch.long,
                device=inputs_embeds.device,
            )
            output = code_predictor(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
        elif use_cache:
            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                ),
                dim=1,
            )
            output = code_predictor(
                input_ids=residual_codes[:, index - 1 : index],
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                generation_steps=index,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
        else:
            if index > 0:
                previous_embedding = _predictor_embeddings(code_predictor)[index - 1](
                    residual_codes[:, index - 1 : index]
                ).to(device=full_inputs_embeds.device, dtype=full_inputs_embeds.dtype)
                full_inputs_embeds = torch.cat((full_inputs_embeds, previous_embedding), dim=1)
            attention_mask = torch.ones(
                (full_inputs_embeds.shape[0], full_inputs_embeds.shape[1]),
                dtype=torch.long,
                device=full_inputs_embeds.device,
            )
            output = code_predictor(
                inputs_embeds=full_inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
        # Cached generation exposes one token per entry.  Full recomputation
        # exposes the complete prefix, so retain only the equivalent final token.
        hidden_entry = output.hidden_states
        if not use_cache:
            hidden_entry = tuple(hidden[:, -1:, :] for hidden in hidden_entry)
        predictor_hidden_states.append(hidden_entry)
        if return_log_probs:
            logits = output.logits[:, -1, :].float()
            target = residual_codes[:, index]
            token_log_probs.append(torch.log_softmax(logits, dim=-1).gather(-1, target[:, None]).squeeze(-1))
        if use_cache:
            past_key_values = output.past_key_values
            if past_key_values is None and index + 1 < expected:
                raise RuntimeError(
                    "Talker MTP replay requested use_cache=True but the code predictor "
                    "returned no cache; refusing to advance a misordered residual state"
                )

    next_hidden = mix_residual_hidden(
        layer0_hidden=layer0_hidden,
        residual_codes=residual_codes,
        predictor_hidden_states=predictor_hidden_states,
        predictor_embeddings=_predictor_embeddings(code_predictor),
    )
    log_probs = torch.stack(token_log_probs, dim=-1) if token_log_probs else None
    return TalkerResidualStep(
        residual_codes=residual_codes,
        next_codec_hidden=next_hidden,
        log_probs=log_probs,
    )


def add_trailing_text_hidden(
    codec_hidden: torch.Tensor,
    *,
    generation_step: int,
    trailing_text_hidden: torch.Tensor,
    tts_pad_embed: torch.Tensor,
) -> torch.Tensor:
    """Add the official aligned text hidden (or TTS pad after text ends)."""
    if int(generation_step) < trailing_text_hidden.shape[1]:
        text_hidden = trailing_text_hidden[:, int(generation_step) : int(generation_step) + 1]
    else:
        text_hidden = tts_pad_embed
    return codec_hidden + text_hidden.to(device=codec_hidden.device, dtype=codec_hidden.dtype)


__all__ = [
    "TalkerResidualStep",
    "add_trailing_text_hidden",
    "mix_residual_hidden",
    "predict_residual_step",
    "replay_residual_step",
    "residual_step_from_generate",
]
