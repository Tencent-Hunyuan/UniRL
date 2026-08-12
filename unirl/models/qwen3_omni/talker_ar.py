"""Autoregression and replay for the Qwen3-Omni Talker (TTS codec policy)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from unirl.models.types.ar import ARSamplingParams, ARStage, ARStep
from unirl.types.segments import SegmentStatus, TextSegment
from unirl.utils.dtypes import parse_torch_dtype

from .bundle import Qwen3OmniBundle
from .talker_bundle import Qwen3OmniTalkerBundle
from .talker_conditions import Qwen3OmniTalkerConditions
from .talker_contract import NUM_CODE_GROUPS
from .talker_prefix import build_talker_prefix_tts, resolve_speaker_id
from .talker_residual import add_trailing_text_hidden, predict_residual_step, replay_residual_step
from .talker_sampling import (
    TalkerSamplingConfig,
    TalkerSamplingProcessor,
    append_token_history,
    suppress_special_codec_ids,
)

logger = logging.getLogger(__name__)


@dataclass
class Qwen3OmniTalkerARParams:
    """Per-request Talker AR knobs."""

    max_tokens: int = 1024
    temperature: float = 0.9
    top_p: float = 1.0
    top_k: int = 50
    repetition_penalty: float = 1.05
    do_sample: bool = True
    suppress_special_tokens: bool = True
    suppress_token_ids: Optional[List[int]] = None
    eos_token_id: Optional[int] = None
    disable_eos: bool = False


class Qwen3OmniTalkerARStep(ARStep):
    """Sample one codec layer-0 token from logits with behavior log-probs."""

    def __init__(
        self,
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        suppress_token_ids: Optional[List[int]] = None,
        eos_token_id: Optional[int] = None,
        do_sample: bool = True,
    ) -> None:
        self.processor = TalkerSamplingProcessor(
            TalkerSamplingConfig(
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
                repetition_penalty=float(repetition_penalty),
                suppress_token_ids=tuple(int(t) for t in (suppress_token_ids or [])),
                eos_token_id=eos_token_id,
                do_sample=bool(do_sample),
            )
        )

    def step(
        self,
        logits: torch.Tensor,
        token_history: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.processor.sample(logits, token_history=token_history)


TalkerBundle = Union[Qwen3OmniBundle, Qwen3OmniTalkerBundle]


@dataclass
class _TalkerDecodeState:
    """One exact Talker trajectory state, shared by rollout and replay."""

    prefix_embeds: torch.Tensor
    prefix_ids: torch.Tensor
    prefix_position_ids: torch.Tensor
    rope_deltas: torch.Tensor
    trailing_text_hidden: torch.Tensor
    tts_pad_embed: torch.Tensor
    use_cache: bool
    generation_step: int = 0
    past_key_values: Any = None
    history: Optional[torch.Tensor] = None
    generated_embeds: Optional[torch.Tensor] = None

    @property
    def prefix_length(self) -> int:
        return int(self.prefix_embeds.shape[1])

    def _generated_position_ids(self, count: int) -> torch.Tensor:
        if count <= 0:
            return self.prefix_position_ids[..., :0]
        batch_size = int(self.prefix_embeds.shape[0])
        delta = self.rope_deltas.reshape(batch_size, -1)[:, :1]
        positions = torch.arange(count, device=self.prefix_embeds.device, dtype=delta.dtype).view(1, -1)
        positions = positions + delta + self.prefix_length
        return positions.unsqueeze(0).expand(self.prefix_position_ids.shape[0], -1, -1)

    def model_inputs(self) -> Dict[str, Any]:
        if self.generation_step == 0:
            inputs_embeds = self.prefix_embeds
            position_ids = self.prefix_position_ids
        else:
            if self.generated_embeds is None or self.generated_embeds.shape[1] != self.generation_step:
                raise RuntimeError("Talker decode state has a misaligned generated-embedding timeline")
            generated_positions = self._generated_position_ids(self.generation_step)
            if self.use_cache:
                inputs_embeds = self.generated_embeds[:, -1:, :]
                position_ids = generated_positions[..., -1:]
            else:
                inputs_embeds = torch.cat((self.prefix_embeds, self.generated_embeds), dim=1)
                position_ids = torch.cat((self.prefix_position_ids, generated_positions), dim=-1)
        total_length = self.prefix_length + self.generation_step
        attention_mask = torch.ones(
            (self.prefix_embeds.shape[0], total_length),
            dtype=torch.long,
            device=self.prefix_embeds.device,
        )
        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": self.past_key_values if self.use_cache else None,
            "use_cache": self.use_cache,
            # Official prefill returns generation_step=0; the first incremental
            # forward consumes that value.  Our state counts actions already
            # appended, hence the one-step offset here.
            "generation_step": max(self.generation_step - 1, 0),
            "talker_input_ids": self.prefix_ids,
            "trailing_text_hidden": self.trailing_text_hidden,
            "tts_pad_embed": self.tts_pad_embed,
            "output_hidden_states": True,
            "return_dict": True,
        }

    def advance(self, *, output: Any, token_id: torch.Tensor, next_embed: torch.Tensor) -> None:
        if self.use_cache:
            cache = getattr(output, "past_key_values", None)
            if cache is None:
                raise RuntimeError(
                    "Talker decode requested use_cache=True but the model returned no cache; "
                    "refusing to silently restart the prefix with an advanced timeline"
                )
            self.past_key_values = cache
        self.history = append_token_history(self.history, token_id)
        self.generated_embeds = (
            next_embed
            if self.generated_embeds is None
            else torch.cat((self.generated_embeds, next_embed), dim=1)
        )
        self.generation_step += 1


def _talker_owner(bundle: TalkerBundle) -> Any:
    """Return the object that owns config/talker/code2wav.

    New direct-TTS bundles own these directly. The fallback keeps manually
    constructed legacy full-Omni bundles usable without making the production
    path depend on them.
    """
    if hasattr(bundle, "config") and hasattr(bundle, "input_embedding_provider"):
        return bundle
    omni = getattr(bundle, "omni", None)
    if omni is None:
        raise ValueError(
            "Qwen3OmniTalkerARStage requires Qwen3OmniTalkerBundle or a legacy bundle containing a full Omni model"
        )
    return omni


def _suppress_special_codec_ids(model: Any) -> List[int]:
    cfg = model.config.talker_config
    return list(
        suppress_special_codec_ids(
            vocab_size=int(cfg.text_config.vocab_size),
            codec_eos_token_id=int(cfg.codec_eos_token_id),
        )
    )


def _build_decode_state(
    *,
    owner: Any,
    input_ids: torch.Tensor,
    speaker_id: int,
    device: torch.device,
    batch_idx: int,
    expected_prefix_ids: Optional[Any],
    use_cache: bool,
) -> _TalkerDecodeState:
    prefix_embeds, prefix_ids, trailing_text_hidden, tts_pad_embed = build_talker_prefix_tts(
        owner,
        input_ids=input_ids,
        speaker_id=speaker_id,
        device=device,
        expected_prefix_ids=expected_prefix_ids,
        batch_idx=batch_idx,
    )
    prefix_mask = torch.ones(prefix_ids.shape, dtype=torch.long, device=device)
    prefix_position_ids, rope_deltas = owner.talker.get_rope_index(
        prefix_ids,
        attention_mask=prefix_mask,
    )
    return _TalkerDecodeState(
        prefix_embeds=prefix_embeds,
        prefix_ids=prefix_ids,
        prefix_position_ids=prefix_position_ids.to(device),
        rope_deltas=rope_deltas.to(device),
        trailing_text_hidden=trailing_text_hidden,
        tts_pad_embed=tts_pad_embed,
        use_cache=bool(use_cache),
    )


def _talker_hidden_sequence(output: Any) -> torch.Tensor:
    hidden_states = getattr(output, "hidden_states", None)
    if not isinstance(hidden_states, (tuple, list)) or not hidden_states:
        raise RuntimeError("Talker forward did not return hidden states required by the residual transition")
    model_hiddens = hidden_states[0] if isinstance(hidden_states[0], (tuple, list)) else hidden_states
    if not isinstance(model_hiddens, (tuple, list)) or not model_hiddens:
        raise RuntimeError("Talker forward returned a malformed hidden-state payload")
    hidden = model_hiddens[-1]
    if not isinstance(hidden, torch.Tensor) or hidden.dim() != 3:
        raise RuntimeError("Talker forward returned a malformed final hidden state")
    return hidden


def _last_talker_hidden(output: Any) -> torch.Tensor:
    return _talker_hidden_sequence(output)[:, -1:, :]


class Qwen3OmniTalkerARStage(ARStage[Qwen3OmniTalkerConditions]):
    """Talker(+MTP) codec AR stage for TTS RL / SFT."""

    def __init__(
        self,
        *,
        model: TalkerBundle,
        autocast_precision: str = "bf16",
        logprob_precision: str = "fp32",
    ) -> None:
        if not getattr(model, "enable_talker", False):
            raise ValueError("Qwen3OmniTalkerARStage requires a Talker-enabled bundle.")
        self.model = model
        self.owner = _talker_owner(model)
        self.autocast_dtype = parse_torch_dtype(
            autocast_precision, field_name="Qwen3OmniTalkerARStage.autocast_precision"
        )
        self.logprob_dtype = parse_torch_dtype(logprob_precision, field_name="Qwen3OmniTalkerARStage.logprob_precision")

    def trainable_module(self) -> torch.nn.Module:
        return self.model.transformer

    def autoregress(
        self,
        conditions: Qwen3OmniTalkerConditions,
        *,
        sampling_params: ARSamplingParams,
        params: Optional[Qwen3OmniTalkerARParams] = None,
        **_kwargs: Any,
    ) -> Tuple[TextSegment, Dict[str, Any]]:
        """Sample layer-0 codes (+ MTP residuals) per sample; return segment + talker control."""
        if conditions.prompt is None or conditions.prompt.input_ids is None:
            raise ValueError("Qwen3OmniTalkerARStage.autoregress: requires conditions.prompt.input_ids")
        if conditions.prompt.attention_mask is None:
            raise ValueError("Qwen3OmniTalkerARStage.autoregress: requires conditions.prompt.attention_mask")

        owner = self.owner
        talker = owner.talker
        device = next(talker.parameters()).device
        input_ids = conditions.prompt.input_ids.to(device)
        batch_size = int(input_ids.shape[0])
        if conditions.speaker_ids is not None:
            speaker_ids = [int(v) for v in conditions.speaker_ids]
        else:
            speakers = list(conditions.speakers or [self.model.default_speaker] * batch_size)
            speaker_ids = [resolve_speaker_id(owner, name) for name in speakers]
        if len(speaker_ids) != batch_size:
            raise ValueError(
                f"Qwen3OmniTalkerARStage.autoregress: speaker_ids len {len(speaker_ids)} != batch {batch_size}"
            )

        knobs = params or Qwen3OmniTalkerARParams()
        max_new = int(sampling_params.max_new_tokens if sampling_params.max_new_tokens else knobs.max_tokens)
        temperature = float(
            sampling_params.temperature if sampling_params.temperature is not None else knobs.temperature
        )
        top_p = float(sampling_params.top_p if sampling_params.top_p is not None else knobs.top_p)
        top_k = int(sampling_params.top_k if sampling_params.top_k is not None else knobs.top_k)
        if knobs.disable_eos:
            codec_eos = None
        elif sampling_params.stop_token_id is not None:
            codec_eos = int(sampling_params.stop_token_id)
        elif knobs.eos_token_id is not None:
            codec_eos = int(knobs.eos_token_id)
        else:
            codec_eos = int(owner.config.talker_config.codec_eos_token_id)
        if knobs.suppress_token_ids is not None:
            suppress = [int(token_id) for token_id in knobs.suppress_token_ids]
        elif knobs.suppress_special_tokens:
            suppress = _suppress_special_codec_ids(owner)
        else:
            suppress = []
        behavior_config = TalkerSamplingConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=float(knobs.repetition_penalty),
            suppress_token_ids=tuple(suppress),
            eos_token_id=codec_eos,
            do_sample=bool(knobs.do_sample),
        )
        step = Qwen3OmniTalkerARStep(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=float(knobs.repetition_penalty),
            suppress_token_ids=suppress,
            eos_token_id=codec_eos,
            do_sample=bool(knobs.do_sample),
        )

        all_tokens: List[List[int]] = []
        all_logps: List[List[float]] = []
        all_residuals: List[torch.Tensor] = []
        all_prefix_ids: List[torch.Tensor] = []
        all_statuses: List[int] = []

        for b in range(batch_size):
            state = _build_decode_state(
                owner=owner,
                input_ids=input_ids,
                speaker_id=speaker_ids[b],
                device=device,
                batch_idx=b,
                expected_prefix_ids=None,
                use_cache=True,
            )
            all_prefix_ids.append(state.prefix_ids.squeeze(0).detach().cpu())
            tokens_b: List[int] = []
            logps_b: List[float] = []
            residuals_b: List[torch.Tensor] = []

            finished = False

            for _ in range(max_new):
                if finished:
                    break
                with torch.no_grad():
                    out = talker(**state.model_inputs())
                logits = out.logits[:, -1, :]
                token_id, logp0 = step.step(logits, token_history=state.history)
                tid = int(token_id.item())
                tokens_b.append(tid)

                with torch.no_grad():
                    residual_step = predict_residual_step(
                        talker=talker,
                        past_hidden=_last_talker_hidden(out),
                        layer0_ids=token_id,
                    )
                residual = residual_step.residual_codes
                residuals_b.append(residual.squeeze(0))  # [15]
                # Phase-1 policy action is layer0 only.  Frozen MTP residuals
                # define the next state and waveform, but are not folded into
                # the policy log-probability.
                logps_b.append(float(logp0.item()))

                if bool(step.processor.is_eos(token_id).item()):
                    finished = True

                next_embed = add_trailing_text_hidden(
                    residual_step.next_codec_hidden,
                    generation_step=state.generation_step,
                    trailing_text_hidden=state.trailing_text_hidden,
                    tts_pad_embed=state.tts_pad_embed,
                ).to(talker.dtype)
                state.advance(output=out, token_id=token_id, next_embed=next_embed)

            all_tokens.append(tokens_b)
            all_logps.append(logps_b)
            all_statuses.append(
                int(SegmentStatus.COMPLETED if finished else SegmentStatus.TRUNCATED)
            )
            if residuals_b:
                all_residuals.append(torch.stack(residuals_b, dim=1))  # [15, T]
            else:
                all_residuals.append(torch.zeros((NUM_CODE_GROUPS - 1, 0), dtype=torch.long, device=device))

        segment = TextSegment.pack(
            tokens=[torch.tensor(toks, dtype=torch.long, device=device) for toks in all_tokens],
            log_probs=[torch.tensor(lps, dtype=torch.float32, device=device) for lps in all_logps],
            status=torch.tensor(all_statuses, dtype=torch.long, device=device),
        )
        control = {
            "residual_codes": [r.detach().cpu() for r in all_residuals],
            "speaker_ids": speaker_ids,
            "prefix_ids": all_prefix_ids,
            "behavior_sampling": behavior_config.to_dict(),
        }
        return segment, control

    def replay(
        self,
        conditions: Qwen3OmniTalkerConditions,
        *,
        segment: TextSegment,
        temperature: Optional[float] = None,
    ) -> torch.Tensor:
        """Replay processed layer-0 behavior log-probs packed like ``segment``."""
        layer0_logps, _ = self._replay_teacher_forced(
            conditions,
            segment=segment,
            temperature=temperature,
            include_mtp=False,
        )
        return layer0_logps

    def replay_sft(
        self,
        conditions: Qwen3OmniTalkerConditions,
        *,
        segment: TextSegment,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return vectorized teacher-forced ``(layer0, MTP)`` log-probs.

        Each sample executes one full-sequence Talker forward and exactly 15
        frame-batched MTP forwards. Layer-0 includes the codec EOS target, while
        the corresponding MTP row is zero-filled and excluded by the SFT mask.
        This path is deliberately independent of rollout replay's exact
        frame-by-frame state machine.
        """
        if conditions.prompt is None or conditions.prompt.input_ids is None:
            raise ValueError("Qwen3OmniTalkerARStage.replay_sft: conditions.prompt.input_ids is None")
        if conditions.prompt.attention_mask is None:
            raise ValueError("Qwen3OmniTalkerARStage.replay_sft: conditions.prompt.attention_mask is None")
        if segment.tokens is None or segment.cu_seqlens is None or segment.lengths is None:
            raise ValueError("Qwen3OmniTalkerARStage.replay_sft: segment requires packed tokens")

        owner = self.owner
        talker = owner.talker
        device = next(talker.parameters()).device
        input_ids = conditions.prompt.input_ids.to(device)
        batch_size = int(input_ids.shape[0])
        lengths = [int(length) for length in segment.lengths.tolist()]
        cu = [int(offset) for offset in segment.cu_seqlens.tolist()]
        if len(lengths) != batch_size:
            raise ValueError(
                f"Qwen3OmniTalkerARStage.replay_sft: segment batch {len(lengths)} != prompt batch {batch_size}"
            )

        if conditions.speaker_ids is not None:
            speaker_ids = [int(value) for value in conditions.speaker_ids]
        else:
            speakers = list(conditions.speakers or [self.model.default_speaker] * batch_size)
            speaker_ids = [resolve_speaker_id(owner, name) for name in speakers]
        if len(speaker_ids) != batch_size:
            raise ValueError(
                f"Qwen3OmniTalkerARStage.replay_sft: speaker_ids len {len(speaker_ids)} != batch {batch_size}"
            )

        residual_codes = conditions.residual_codes
        if residual_codes is None or len(residual_codes) != batch_size:
            actual = None if residual_codes is None else len(residual_codes)
            raise ValueError(
                f"Qwen3OmniTalkerARStage.replay_sft: residual_codes len {actual} != batch {batch_size}"
            )
        mtp_masks = conditions.mtp_loss_masks
        if mtp_masks is not None and len(mtp_masks) != batch_size:
            raise ValueError(
                f"Qwen3OmniTalkerARStage.replay_sft: mtp_loss_masks len {len(mtp_masks)} != batch {batch_size}"
            )

        codec_eos = int(owner.config.talker_config.codec_eos_token_id)
        predictor_embeddings = talker.code_predictor.get_input_embeddings()
        if not isinstance(predictor_embeddings, (torch.nn.ModuleList, list, tuple)):
            raise TypeError("Talker code_predictor.get_input_embeddings() must return per-codebook embeddings")
        if len(predictor_embeddings) != NUM_CODE_GROUPS - 1:
            raise ValueError(
                f"Talker SFT requires {NUM_CODE_GROUPS - 1} MTP embedding tables, "
                f"got {len(predictor_embeddings)}"
            )
        layer0_embedding = talker.get_input_embeddings()

        packed_layer0: List[torch.Tensor] = []
        packed_mtp: List[torch.Tensor] = []
        for batch_idx, packed_length in enumerate(lengths):
            if packed_length < 2:
                raise ValueError(
                    f"Talker SFT sample {batch_idx}: expected at least one codec frame plus EOS, "
                    f"got length {packed_length}"
                )
            layer0_targets = segment.tokens[
                cu[batch_idx] : cu[batch_idx] + packed_length
            ].to(device=device, dtype=torch.long)
            if int(layer0_targets[-1].item()) != codec_eos:
                raise ValueError(
                    f"Talker SFT sample {batch_idx}: final layer-0 target must be codec EOS {codec_eos}"
                )
            frame_count = packed_length - 1
            layer0_frames = layer0_targets[:frame_count].view(1, frame_count)

            residuals = residual_codes[batch_idx].to(device=device, dtype=torch.long)
            expected_shape = (NUM_CODE_GROUPS - 1, packed_length)
            if tuple(residuals.shape) != expected_shape:
                raise ValueError(
                    f"Talker SFT residual_codes[{batch_idx}] expected {expected_shape}, "
                    f"got {tuple(residuals.shape)}"
                )
            if torch.any(residuals[:, -1] != 0):
                raise ValueError(f"Talker SFT residual_codes[{batch_idx}] EOS column must be zero-filled")
            if mtp_masks is not None:
                mtp_mask = torch.as_tensor(mtp_masks[batch_idx]).flatten()
                if mtp_mask.shape[0] != packed_length:
                    raise ValueError(
                        f"Talker SFT mtp_loss_masks[{batch_idx}] length {mtp_mask.shape[0]} "
                        f"!= target length {packed_length}"
                    )
                if float(mtp_mask[-1].item()) != 0.0:
                    raise ValueError(f"Talker SFT mtp_loss_masks[{batch_idx}] must exclude the EOS row")

            prefix_embeds, _prefix_ids, trailing_text_hidden, tts_pad_embed = build_talker_prefix_tts(
                owner,
                input_ids=input_ids,
                speaker_id=speaker_ids[batch_idx],
                device=device,
                expected_prefix_ids=(
                    conditions.prefix_ids[batch_idx]
                    if conditions.prefix_ids is not None
                    else None
                ),
                batch_idx=batch_idx,
            )

            layer0_frame_embeds = layer0_embedding(layer0_frames)
            codec_frame_embeds = layer0_frame_embeds
            for predictor_idx, embedding in enumerate(predictor_embeddings):
                residual_frame_ids = residuals[predictor_idx, :frame_count].view(1, frame_count)
                codec_frame_embeds = codec_frame_embeds + embedding(residual_frame_ids)

            text_length = min(frame_count, int(trailing_text_hidden.shape[1]))
            aligned_text = trailing_text_hidden[:, :text_length]
            if text_length < frame_count:
                aligned_text = torch.cat(
                    (
                        aligned_text,
                        tts_pad_embed.expand(-1, frame_count - text_length, -1),
                    ),
                    dim=1,
                )
            codec_inputs = (
                codec_frame_embeds
                + aligned_text.to(device=codec_frame_embeds.device, dtype=codec_frame_embeds.dtype)
            ).to(dtype=talker.dtype)
            full_inputs_embeds = torch.cat((prefix_embeds, codec_inputs), dim=1)
            attention_mask = torch.ones(
                full_inputs_embeds.shape[:2],
                dtype=torch.long,
                device=device,
            )
            talker_output = talker(
                inputs_embeds=full_inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
                trailing_text_hidden=trailing_text_hidden,
                tts_pad_embed=tts_pad_embed,
                output_hidden_states=True,
                return_dict=True,
            )

            prefix_length = int(prefix_embeds.shape[1])
            target_start = prefix_length - 1
            target_end = target_start + packed_length
            target_logits = talker_output.logits[:, target_start:target_end, :].float()
            if int(target_logits.shape[1]) != packed_length:
                raise RuntimeError(
                    f"Talker SFT sample {batch_idx}: Talker returned {target_logits.shape[1]} "
                    f"target positions, expected {packed_length}"
                )
            layer0_logp = (
                torch.log_softmax(target_logits, dim=-1)
                .gather(-1, layer0_targets.view(1, packed_length, 1))
                .reshape(packed_length)
            )

            hidden_sequence = _talker_hidden_sequence(talker_output)
            codec_hidden = hidden_sequence[:, target_start : target_start + frame_count, :]
            if int(codec_hidden.shape[1]) != frame_count:
                raise RuntimeError(
                    f"Talker SFT sample {batch_idx}: Talker returned {codec_hidden.shape[1]} "
                    f"codec hidden states, expected {frame_count}"
                )
            hidden_size = int(codec_hidden.shape[-1])
            hidden_flat = codec_hidden.reshape(frame_count, 1, hidden_size)
            layer0_flat = layer0_frame_embeds.reshape(frame_count, 1, hidden_size)

            frame_mtp_logps = []
            for predictor_idx in range(NUM_CODE_GROUPS - 1):
                mtp_inputs = [hidden_flat, layer0_flat]
                for previous_idx in range(predictor_idx):
                    previous_ids = residuals[previous_idx, :frame_count].view(1, frame_count)
                    previous_embed = predictor_embeddings[previous_idx](previous_ids)
                    mtp_inputs.append(previous_embed.reshape(frame_count, 1, hidden_size))
                predictor_output = talker.code_predictor(
                    inputs_embeds=torch.cat(mtp_inputs, dim=1).to(dtype=talker.dtype),
                    generation_steps=predictor_idx,
                    use_cache=False,
                )
                mtp_logits = predictor_output.logits[:, -1, :].float()
                mtp_targets = residuals[predictor_idx, :frame_count]
                frame_mtp_logps.append(
                    torch.log_softmax(mtp_logits, dim=-1)
                    .gather(-1, mtp_targets[:, None])
                    .squeeze(-1)
                )

            mtp_logp = torch.stack(frame_mtp_logps, dim=-1)
            mtp_logp = torch.cat(
                (
                    mtp_logp,
                    mtp_logp.new_zeros((1, NUM_CODE_GROUPS - 1)),
                ),
                dim=0,
            )
            packed_layer0.append(layer0_logp.to(dtype=self.logprob_dtype))
            packed_mtp.append(mtp_logp.to(dtype=self.logprob_dtype))

        if not packed_layer0:
            return (
                torch.zeros(0, dtype=self.logprob_dtype, device=device),
                torch.zeros((0, NUM_CODE_GROUPS - 1), dtype=self.logprob_dtype, device=device),
            )
        return torch.cat(packed_layer0, dim=0), torch.cat(packed_mtp, dim=0)

    def _replay_teacher_forced(
        self,
        conditions: Qwen3OmniTalkerConditions,
        *,
        segment: TextSegment,
        temperature: Optional[float],
        include_mtp: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if conditions.prompt is None or conditions.prompt.input_ids is None:
            raise ValueError("Qwen3OmniTalkerARStage.replay: conditions.prompt.input_ids is None")
        if conditions.prompt.attention_mask is None:
            raise ValueError("Qwen3OmniTalkerARStage.replay: conditions.prompt.attention_mask is None")
        if segment.tokens is None or segment.cu_seqlens is None or segment.lengths is None:
            raise ValueError("Qwen3OmniTalkerARStage.replay: segment requires packed tokens")

        owner = self.owner
        talker = owner.talker
        device = next(talker.parameters()).device
        input_ids = conditions.prompt.input_ids.to(device)
        batch_size = int(input_ids.shape[0])
        if conditions.speaker_ids is not None:
            speaker_ids = [int(v) for v in conditions.speaker_ids]
        else:
            speakers = list(conditions.speakers or [self.model.default_speaker] * batch_size)
            speaker_ids = [resolve_speaker_id(owner, name) for name in speakers]
        if len(speaker_ids) != batch_size:
            raise ValueError(f"Qwen3OmniTalkerARStage.replay: speaker_ids len {len(speaker_ids)} != batch {batch_size}")
        lengths = [int(n) for n in segment.lengths.tolist()]
        cu = [int(c) for c in segment.cu_seqlens.tolist()]
        behavior_processor = None
        if not include_mtp:
            behavior_config = conditions.validate_rollout_replay(lengths=lengths)
            if temperature is not None and float(temperature) != float(behavior_config.temperature):
                raise ValueError(
                    "Talker replay temperature differs from the stored behavior distribution: "
                    f"requested={temperature}, rollout={behavior_config.temperature}"
                )
            behavior_processor = TalkerSamplingProcessor(behavior_config)

        residual_codes = conditions.residual_codes
        if residual_codes is None:
            raise ValueError(
                "Qwen3OmniTalkerARStage.replay requires residual_codes; "
                "zero-filled residuals do not reproduce the rollout state"
            )
        if len(residual_codes) != batch_size:
            raise ValueError(
                f"Qwen3OmniTalkerARStage.replay: residual_codes len {len(residual_codes)} != batch {batch_size}"
            )

        flat_parts: List[torch.Tensor] = []
        flat_mtp_parts: List[torch.Tensor] = []
        for b in range(batch_size):
            n = lengths[b]
            if n == 0:
                continue
            layer0 = segment.tokens[cu[b] : cu[b] + n].to(device=device, dtype=torch.long).view(1, -1)
            residuals = residual_codes[b].to(device=device, dtype=torch.long)
            if residuals.dim() != 2 or residuals.shape[0] != NUM_CODE_GROUPS - 1:
                raise ValueError(f"residual_codes[{b}] expected [15, T], got {tuple(residuals.shape)}")
            if int(residuals.shape[1]) != n:
                raise ValueError(
                    f"Qwen3OmniTalkerARStage.replay: residual_codes[{b}] timeline "
                    f"{residuals.shape[1]} != layer-0 timeline {n}"
                )

            state = _build_decode_state(
                owner=owner,
                input_ids=input_ids,
                speaker_id=speaker_ids[b],
                device=device,
                batch_idx=b,
                expected_prefix_ids=(
                    conditions.prefix_ids[b]
                    if conditions.prefix_ids is not None
                    else None
                ),
                # Training replay must remain correct when gradient
                # checkpointing disables caches.  Eval rollout/replay may use
                # the exact incremental path and verifies that a cache exists.
                use_cache=(not torch.is_grad_enabled() and not talker.training),
            )

            token_logps = []
            mtp_logps = []
            for pos in range(n):
                talker_out = talker(**state.model_inputs())
                target = layer0[:, pos]
                if include_mtp:
                    token_logps.append(
                        torch.log_softmax(talker_out.logits[:, -1, :].float(), dim=-1)
                        .gather(-1, target[:, None])
                        .squeeze(-1)
                    )
                else:
                    assert behavior_processor is not None
                    token_logps.append(
                        behavior_processor.score(
                            talker_out.logits[:, -1, :],
                            target,
                            token_history=state.history,
                        )
                    )
                residual_step = replay_residual_step(
                    talker=talker,
                    past_hidden=_last_talker_hidden(talker_out),
                    layer0_ids=target,
                    residual_codes=residuals[:, pos].view(1, -1),
                    return_log_probs=include_mtp,
                    use_cache=state.use_cache,
                )
                if include_mtp:
                    if residual_step.log_probs is None:
                        raise RuntimeError("MTP replay did not return target log-probs")
                    mtp_logps.append(residual_step.log_probs)
                next_embed = add_trailing_text_hidden(
                    residual_step.next_codec_hidden,
                    generation_step=state.generation_step,
                    trailing_text_hidden=state.trailing_text_hidden,
                    tts_pad_embed=state.tts_pad_embed,
                ).to(talker.dtype)
                state.advance(output=talker_out, token_id=target, next_embed=next_embed)
            logp0 = torch.cat(token_logps, dim=0)
            flat_parts.append(logp0.to(dtype=self.logprob_dtype))
            if include_mtp:
                flat_mtp_parts.append(torch.cat(mtp_logps, dim=0).to(dtype=self.logprob_dtype))

        if not flat_parts:
            empty_layer0 = torch.zeros(0, dtype=self.logprob_dtype, device=device)
            empty_mtp = torch.zeros(
                (0, NUM_CODE_GROUPS - 1),
                dtype=self.logprob_dtype,
                device=device,
            )
            return empty_layer0, empty_mtp if include_mtp else None
        return (
            torch.cat(flat_parts, dim=0),
            torch.cat(flat_mtp_parts, dim=0) if include_mtp else None,
        )

    def decode_codes_to_audio(
        self,
        *,
        layer0_codes: List[torch.Tensor],
        residual_codes: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Decode per-sample codes → mono waveforms ``[L]`` at 24 kHz."""
        owner = self.owner
        code2wav = owner.code2wav
        device = next(code2wav.parameters()).device
        wavs: List[torch.Tensor] = []
        for layer0, residual in zip(layer0_codes, residual_codes):
            layer0 = layer0.to(device=device, dtype=torch.long).view(-1)
            residual = residual.to(device=device, dtype=torch.long)
            if layer0.numel() == 0:
                wavs.append(torch.zeros(0, device=device))
                continue
            # Drop trailing EOS if present for vocoder input.
            eos = int(owner.config.talker_config.codec_eos_token_id)
            if int(layer0[-1].item()) == eos:
                layer0 = layer0[:-1]
                residual = residual[:, : layer0.numel()]
            if layer0.numel() == 0:
                wavs.append(torch.zeros(0, device=device))
                continue
            # codes layout expected by code2wav: [num_quantizers, T]
            codes = torch.cat([layer0.view(1, -1), residual[:, : layer0.numel()]], dim=0)
            with torch.no_grad():
                audio = code2wav.chunked_decode(
                    codes.unsqueeze(0).to(device),
                    chunk_size=300,
                    left_context_size=25,
                )
            audio = audio.detach().float().reshape(-1).cpu()
            wavs.append(audio)
        return wavs


__all__ = [
    "Qwen3OmniTalkerARParams",
    "Qwen3OmniTalkerARStage",
    "Qwen3OmniTalkerARStep",
]
