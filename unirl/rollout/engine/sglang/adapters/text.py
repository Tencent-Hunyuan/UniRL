"""``TextLMAdapter`` — the per-shape base adapter for a packed-text generation Part.

Holds the conversion logic once: chat-template encoding into per-prompt
``/generate`` payloads (``build_inputs``) and the predecessor's
``build_response`` packing fanned out per ``Part`` field
(``build_response`` is the template; ``build_segment`` /
``build_decoded`` / ``build_conditions`` each derive one field from
``(req, prepared, raw)``). The VLM adapter overrides the steps that differ.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import torch

from unirl.config.require import require
from unirl.rollout.engine.sglang.adapters.base import (
    ModelAdapter,
    PreparedInputs,
    register_adapter,
)
from unirl.rollout.engine.sglang.backends import RawResult
from unirl.rollout.engine.sglang.utils import (
    ResolvedSampling,
    build_text_conversations,
    pack_prompt_condition,
)
from unirl.types.primitives import Texts
from unirl.types.sample import Sample
from unirl.types.segments.base import SegmentStatus
from unirl.types.segments.text import TextSegment

logger = logging.getLogger(__name__)

#: SGLang ``finish_reason`` → terminal :class:`SegmentStatus` (LIN-531). The agentic
#: rollout engine reads this per-candidate status to tell a *terminal* turn
#: (COMPLETED / TRUNCATED / ABORTED) apart from an unfinished one; unknown reasons
#: (or a partial carrying none) fall back to PENDING.
_FINISH_TO_STATUS = {
    "stop": SegmentStatus.COMPLETED,
    "length": SegmentStatus.TRUNCATED,
    "abort": SegmentStatus.ABORTED,
}


@register_adapter("text")
class TextLMAdapter(ModelAdapter):
    """Text-only LLM conversion (e.g. Qwen3). The base the VLM adapter derives."""

    def validate(self) -> None:
        super().validate()
        if not self._has_chat_template():
            logger.info(
                "%s: tokenizer has no chat template — raw-text completion rollouts",
                type(self).__name__,
            )

    def _has_chat_template(self) -> bool:
        return hasattr(self._tokenizer, "apply_chat_template") and bool(getattr(self._tokenizer, "chat_template", None))

    # ------------------------------------------------------------------ #
    # build_inputs — request ``Sample`` → per-prompt /generate payloads
    # ------------------------------------------------------------------ #

    def build_inputs(self, sample: Sample, *, sampling: ResolvedSampling) -> PreparedInputs:
        use_template = self._has_chat_template()
        require(
            use_template or sampling.system_instruction is None,
            f"{type(self).__name__}: system_instruction is configured but the tokenizer "
            "has no chat template to render it (raw-text completion mode)",
        )

        wire: List[Dict[str, Any]] = []
        prompt_token_ids: List[List[int]] = []

        if use_template:
            # Render the whole trajectory (role-tagged turns), de-expanded to the
            # unique conversations the backend fans out ``n`` per. Degenerates to a
            # single user turn on single-turn requests (byte-identical to before).
            conversations, k = build_text_conversations(sample, sampling.system_instruction)
            require(
                k == sampling.n,
                f"{type(self).__name__}.build_inputs: de-expanded fan-out k={k} != "
                f"resolved n={sampling.n}; conversation grouping and the sampling block "
                "disagree on the gen branch.",
            )
            for messages in conversations:
                payload = self.base_payload(sampling)
                ids = self.apply_chat_template(messages)
                payload["input_ids"] = ids
                prompt_token_ids.append(list(ids))
                wire.append(payload)
        else:
            # Raw-text completion mode — no chat template, so no roles/turns: encode
            # the raw prompt so the replay's prompt condition still carries the ids
            # the server tokenized.
            for prompt in self.extract_prompts(sample):
                payload = self.base_payload(sampling)
                payload["text"] = prompt
                ids = list(self._tokenizer.encode(prompt))
                prompt_token_ids.append(list(ids))
                wire.append(payload)

        return PreparedInputs(
            wire=wire,
            prompt_token_ids=prompt_token_ids,
            resolved_n=sampling.n,
        )

    def extract_prompts(self, sample: Sample) -> List[str]:
        text_primitive = sample.parts[0].primitives.get("text")
        require(
            text_primitive is not None and isinstance(text_primitive, Texts),
            f"{type(self).__name__} requires the input Part.primitives['text']: Texts",
        )
        return list(text_primitive.texts)

    def base_payload(self, sampling: ResolvedSampling) -> Dict[str, Any]:
        """The sampling fields every ``/generate`` payload carries."""
        return {
            "sampling_params": dict(sampling.block),
            "return_logprob": sampling.return_logprob,
        }

    def apply_chat_template(self, messages: List[Dict[str, Any]]) -> List[int]:
        """Tokenize a chat conversation into ``input_ids`` via the chat template.

        Only called in templated mode (``_has_chat_template``). ``messages`` is the
        role-tagged conversation :func:`build_text_conversations` assembled (system
        prefix + one message per turn). A failure raises: a set-but-broken template
        (bad ``chat_template_kwargs``, jinja error) is a config bug — silently
        switching the run's prompt format would corrupt training.
        """
        # tokenize=True + return_dict=False yields a bare List[int]. transformers
        # >=5 defaults return_dict=True, handing back a BatchEncoding that then
        # leaks into the JSON /generate payload ("Object of type BatchEncoding is
        # not JSON serializable"). Force the list form and normalize any
        # tensor/batch dim a template kwarg might introduce.
        template_kwargs: Dict[str, Any] = {
            "add_generation_prompt": True,
            "tokenize": True,
            "return_dict": False,
        }
        template_kwargs.update(self.cfg.chat_template_kwargs or {})
        ids = self._tokenizer.apply_chat_template(messages, **template_kwargs)
        if hasattr(ids, "input_ids"):  # BatchEncoding (return_dict re-enabled)
            ids = ids["input_ids"]
        if hasattr(ids, "tolist"):  # torch / numpy tensor (return_tensors)
            ids = ids.tolist()
        if ids and isinstance(ids[0], (list, tuple)):  # leading batch dim of 1
            ids = ids[0]
        return [int(t) for t in ids]

    # ------------------------------------------------------------------ #
    # build_response — the template: one fan-out stage per ``Part`` field
    # ------------------------------------------------------------------ #

    def build_response(self, sample: Sample, prepared: PreparedInputs, raw: List[RawResult]) -> Sample:
        """Fill the frontier gen ``Part`` from the seam's per-candidate results.

        ``raw`` is in prompt-major order: candidate ``k`` of prompt ``i`` is at
        index ``i * n + k`` (the seam's ordering contract) — the same group-by-
        parent order the gen shell was forked in, so row ``j`` of the gen part
        maps to ``raw[j]``. Each stage derives its field from ``(sample, prepared,
        raw)`` independently in that shared order.
        """
        gen_part = sample.parts[-1]
        n = int(prepared.resolved_n)
        n_prompts = len(prepared.prompt_token_ids)
        require(
            len(raw) == n_prompts * n,
            f"{type(self).__name__}.build_response: expected {n_prompts * n} "
            f"candidates ({n_prompts} prompts × n={n}); got {len(raw)}",
        )
        require(
            len(gen_part.sample_ids) == len(raw),
            f"{type(self).__name__}.build_response: gen shell holds "
            f"{len(gen_part.sample_ids)} samples but the seam returned {len(raw)}",
        )

        filled = gen_part.fill(
            segment=self.build_segment(sample, prepared, raw),
            primitives={"text": self.build_decoded(sample, prepared, raw)},
            conditions=self.build_conditions(sample, prepared, raw),
            status=self.build_status(raw),
        )
        # Preserve every input Part: multi-input multimodal chains image / cot_text
        # input Parts before the gen shell, so the filled gen Part's parent id must
        # stay in the returned chain (text-only: parts[:-1] == [head], unchanged).
        return Sample(parts=[*sample.parts[:-1], filled])

    def build_segment(self, sample: Sample, prepared: PreparedInputs, raw: List[RawResult]) -> TextSegment:
        """Pack the per-candidate tokens/logprobs, each row pointing at its own slot."""
        return TextSegment.pack(
            tokens=[torch.tensor(list(r.token_ids or []), dtype=torch.long) for r in raw],
            log_probs=[torch.tensor(list(r.logprobs or []), dtype=torch.float32) for r in raw],
        )

    @staticmethod
    def build_status(raw: List[RawResult]) -> torch.Tensor:
        """Per-candidate terminal status (LIN-531) from the seam's ``finish_reason``:
        ``stop`` → COMPLETED, ``length`` → TRUNCATED, ``abort`` → ABORTED, else PENDING.
        A ``[n_candidates]`` long tensor (one ``SegmentStatus`` value per row)."""
        return torch.tensor(
            [int(_FINISH_TO_STATUS.get(str(r.finish_reason), SegmentStatus.PENDING)) for r in raw],
            dtype=torch.long,
        )

    def build_decoded(self, sample: Sample, prepared: PreparedInputs, raw: List[RawResult]) -> Texts:
        """Emit the RAW sampler text per candidate (verl-reference parity).

        Reward grading scores the full decoded response. The predecessor's
        think-stripping (``content or text``) silently dropped boxed answers
        living inside think markup — Qwen3-Base emits it organically on math —
        depressing MathBoxed rewards ~3x (observed: LIN-381 e2e #1/#2 flat at
        ~0.035 vs the b182a511-lineage v1 references at 0.09-0.25).
        """
        return Texts(texts=[r.text or "" for r in raw])

    def build_conditions(self, sample: Sample, prepared: PreparedInputs, raw: List[RawResult]) -> Dict[str, Any]:
        """The replay conditions — the prompt ids the server saw, per sample.

        Each prompt's ids are replicated across its ``n`` siblings (every
        sibling was generated under the identical prompt). Overridden by the
        VLM adapter to add the multimodal conditions.
        """
        per_sample, _ = self.replicate_per_sample(prepared)
        conditions: Dict[str, Any] = {}
        prompt_condition = pack_prompt_condition(per_sample, pad_token_id=self.pad_token_id())
        if prompt_condition is not None:
            conditions["prompt"] = prompt_condition
        return conditions

    @staticmethod
    def replicate_per_sample(prepared: PreparedInputs) -> Tuple[List[List[int]], List[int]]:
        """Replicate per-prompt values to per-sample rows (prompt-major order)."""
        n = int(prepared.resolved_n)
        per_sample: List[List[int]] = []
        prompt_index: List[int] = []
        for i, ids in enumerate(prepared.prompt_token_ids):
            for _ in range(n):
                per_sample.append(list(ids))
                prompt_index.append(i)
        return per_sample, prompt_index


__all__ = ["TextLMAdapter"]
