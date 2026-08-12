"""Typed TTS rollout pipeline for Qwen3-Omni Talker."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import torch

from unirl.models.types.pipeline import Pipeline
from unirl.types.conditions import TextTokenCondition
from unirl.types.primitives import Audios, Texts
from unirl.types.sample import Sample
from unirl.types.sampling import ARSamplingParams

from .bundle import Qwen3OmniBundle
from .talker_ar import Qwen3OmniTalkerARParams, Qwen3OmniTalkerARStage
from .talker_bundle import Qwen3OmniTalkerBundle
from .talker_conditions import Qwen3OmniTalkerConditions
from .talker_contract import AUDIO_SAMPLE_RATE
from .talker_prefix import resolve_speaker_id

TalkerBundle = Union[Qwen3OmniBundle, Qwen3OmniTalkerBundle]


_DEFAULT_TTS_SYSTEM = (
    "You are a high-quality TTS model. Convert the user's text into natural speech audio."
)


def build_tts_messages(
    *,
    text: str,
    system_instruction: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Teacher-provided assistant transcript for native Talker text alignment.

    Qwen3-Omni Talker injects the final assistant span into
    ``trailing_text_hidden`` one text token per codec step. Using
    ``assistant=<audio>`` leaves that timeline nearly empty and lets SFT reduce
    codec CE without learning what to say. Direct TTS therefore supplies the
    desired transcript as the assistant text while the user turn states the
    read-aloud task.
    """
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_instruction or _DEFAULT_TTS_SYSTEM}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": f"Read the following text aloud:\n{text}"}],
        },
        {"role": "assistant", "content": [{"type": "text", "text": text}]},
    ]


def tokenize_tts_batch(
    bundle: TalkerBundle,
    texts: List[str],
    *,
    system_instruction: Optional[str] = None,
    max_length: int = 4096,
) -> TextTokenCondition:
    """Tokenize a batch of TTS prompts into left-aligned padded tensors."""
    tokenizer = bundle.tokenizer
    processor = bundle.processor
    input_rows: List[List[int]] = []
    for text in texts:
        messages = build_tts_messages(text=text, system_instruction=system_instruction)
        # Use the Omni processor with explicit text content mappings. Its
        # checkpoint owns the chat template; the nested tokenizer does not.
        tmpl = getattr(processor, "apply_chat_template", None) or getattr(tokenizer, "apply_chat_template", None)
        if tmpl is None:
            raise RuntimeError("tokenize_tts_batch: processor/tokenizer missing apply_chat_template")
        ids = tmpl(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors=None,
        )
        if isinstance(ids, dict):
            ids = ids.get("input_ids") or ids.get("input_ids".upper())
        if isinstance(ids, torch.Tensor):
            ids = ids.view(-1).tolist()
        elif (
            isinstance(ids, (list, tuple))
            and len(ids) == 1
            and isinstance(ids[0], (list, tuple))
        ):
            ids = ids[0]
        input_rows.append([int(t) for t in ids])

    max_len = min(max(len(r) for r in input_rows), int(max_length))
    pad_id = int(getattr(tokenizer, "pad_token_id", None) or 0)
    batch_ids = []
    batch_mask = []
    for row in input_rows:
        row = row[:max_len]
        pad = max_len - len(row)
        # Left pad so the assistant span stays right-aligned for Talker prefix.
        batch_ids.append([pad_id] * pad + row)
        batch_mask.append([0] * pad + [1] * len(row))
    device = bundle.device
    return TextTokenCondition(
        input_ids=torch.tensor(batch_ids, dtype=torch.long, device=device),
        attention_mask=torch.tensor(batch_mask, dtype=torch.long, device=device),
    )


class Qwen3OmniTalkerPipeline(Pipeline):
    """Talker TTS pipeline: text(+speaker) → codec TextSegment + Audios."""

    def __init__(
        self,
        *,
        bundle: TalkerBundle,
        ar: Optional[Qwen3OmniTalkerARStage] = None,
        max_prompt_length: int = 4096,
        system_instruction: Optional[str] = None,
        autocast_precision: str = "bf16",
        logprob_precision: str = "fp32",
        do_sample: bool = True,
        repetition_penalty: float = 1.05,
        suppress_special_tokens: bool = True,
        suppress_token_ids: Optional[List[int]] = None,
        eos_token_id: Optional[int] = None,
        disable_eos: bool = False,
        decode_audio: bool = True,
    ) -> None:
        super().__init__()
        if not bundle.enable_talker:
            raise ValueError("Qwen3OmniTalkerPipeline requires bundle.enable_talker=True")
        self.bundle = bundle
        self.max_prompt_length = int(max_prompt_length)
        self.system_instruction = system_instruction or bundle.tts_system_instruction or _DEFAULT_TTS_SYSTEM
        self.decode_audio = bool(decode_audio)
        self.do_sample = bool(do_sample)
        self.repetition_penalty = float(repetition_penalty)
        self.suppress_special_tokens = bool(suppress_special_tokens)
        self.suppress_token_ids = (
            None if suppress_token_ids is None else [int(token_id) for token_id in suppress_token_ids]
        )
        self.eos_token_id = None if eos_token_id is None else int(eos_token_id)
        self.disable_eos = bool(disable_eos)
        self.ar = ar or Qwen3OmniTalkerARStage(
            model=bundle,
            autocast_precision=autocast_precision,
            logprob_precision=logprob_precision,
        )

    @classmethod
    def from_bundle(
        cls,
        bundle: TalkerBundle,
        *,
        max_prompt_length: int = 4096,
        system_instruction: Optional[str] = None,
        autocast_precision: str = "bf16",
        logprob_precision: str = "fp32",
        do_sample: bool = True,
        repetition_penalty: float = 1.05,
        suppress_special_tokens: bool = True,
        suppress_token_ids: Optional[List[int]] = None,
        eos_token_id: Optional[int] = None,
        disable_eos: bool = False,
        decode_audio: bool = True,
        **_kwargs: Any,
    ) -> "Qwen3OmniTalkerPipeline":
        return cls(
            bundle=bundle,
            max_prompt_length=max_prompt_length,
            system_instruction=system_instruction,
            autocast_precision=autocast_precision,
            logprob_precision=logprob_precision,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            suppress_special_tokens=suppress_special_tokens,
            suppress_token_ids=suppress_token_ids,
            eos_token_id=eos_token_id,
            disable_eos=disable_eos,
            decode_audio=decode_audio,
        )

    def _speaker_ids_from_sample(self, sample: Sample, batch_size: int) -> List[int]:
        meta = sample.root_metadata(-1)
        owner = getattr(self.bundle, "omni", None) or self.bundle
        speaker_ids: List[int] = []
        for i in range(batch_size):
            m = meta[i] if meta is not None and i < len(meta) else None
            if isinstance(m, dict) and m.get("speaker_id") is not None:
                speaker_ids.append(int(m["speaker_id"]))
            else:
                speaker = (
                    str(m["speaker"])
                    if isinstance(m, dict) and m.get("speaker")
                    else self.bundle.default_speaker
                )
                speaker_ids.append(resolve_speaker_id(owner, speaker))
        return speaker_ids

    def _conditions_for(self, sample: Sample) -> Qwen3OmniTalkerConditions:
        texts_prim = None
        for prim in sample.conditioning():
            if isinstance(prim, Texts):
                texts_prim = prim
                break
        if texts_prim is None:
            raise ValueError("Qwen3OmniTalkerPipeline: conditioning must include Texts prompts")
        texts = list(texts_prim.texts)
        speaker_ids = self._speaker_ids_from_sample(sample, len(texts))
        prompt = tokenize_tts_batch(
            self.bundle,
            texts,
            system_instruction=self.system_instruction,
            max_length=self.max_prompt_length,
        )
        return Qwen3OmniTalkerConditions(prompt=prompt, speaker_ids=speaker_ids)

    def generate(self, sample: Sample) -> Sample:
        frontier = sample.frontier_gen_part(ARSamplingParams)
        ar = frontier.sampling_params
        assert isinstance(ar, ARSamplingParams)

        conds = self._conditions_for(sample)
        params = Qwen3OmniTalkerARParams(
            max_tokens=int(ar.max_new_tokens),
            temperature=float(ar.temperature),
            top_p=float(ar.top_p),
            top_k=int(ar.top_k),
            repetition_penalty=self.repetition_penalty,
            do_sample=self.do_sample,
            suppress_special_tokens=self.suppress_special_tokens,
            suppress_token_ids=self.suppress_token_ids,
            eos_token_id=self.eos_token_id,
            disable_eos=self.disable_eos,
        )
        sampling_params = ARSamplingParams(
            samples_per_prompt=int(ar.samples_per_prompt),
            max_new_tokens=int(params.max_tokens),
            temperature=float(params.temperature),
            top_p=float(params.top_p),
            top_k=int(params.top_k),
            stop_token_id=ar.stop_token_id,
        )
        segment, talker_ctrl = self.ar.autoregress(conds, sampling_params=sampling_params, params=params)

        # Attach residuals onto conditions so TrainStack/GSPO replay can see them.
        conds = Qwen3OmniTalkerConditions(
            prompt=conds.prompt,
            speaker_ids=talker_ctrl["speaker_ids"],
            prefix_ids=talker_ctrl["prefix_ids"],
            residual_codes=talker_ctrl["residual_codes"],
            behavior_sampling=dict(talker_ctrl["behavior_sampling"]),
        )

        primitives: Dict[str, Any] = {}
        # Keep the spoken transcript on the frontier for text-routed diagnostics.
        texts_prim = None
        for prim in sample.conditioning():
            if isinstance(prim, Texts):
                texts_prim = prim
                break
        if texts_prim is not None:
            primitives["text"] = texts_prim

        if self.decode_audio and segment.tokens is not None and segment.cu_seqlens is not None:
            cu = [int(c) for c in segment.cu_seqlens.tolist()]
            layer0_list = [segment.tokens[cu[i] : cu[i + 1]].detach().cpu() for i in range(len(cu) - 1)]
            wavs = self.ar.decode_codes_to_audio(
                layer0_codes=layer0_list,
                residual_codes=list(talker_ctrl["residual_codes"]),
            )
            from unirl.types.primitives import Audio

            audio_items = []
            for w in wavs:
                if w.numel() == 0:
                    audio_items.append(Audio(waveform=torch.zeros(1)))
                else:
                    # Audios pack along L; keep mono [L].
                    audio_items.append(Audio(waveform=w.reshape(-1)))
            primitives["audio"] = Audios.from_list(audio_items)

        filled = frontier.fill(
            segment=segment,
            primitives=primitives,
            conditions=conds.to_dict(),
            primitive_metadata={"audio": {"sample_rate": AUDIO_SAMPLE_RATE}} if "audio" in primitives else None,
            status=segment.status,
        )
        return sample.replace_frontier(filled)


__all__ = [
    "Qwen3OmniTalkerPipeline",
    "build_tts_messages",
    "tokenize_tts_batch",
]
