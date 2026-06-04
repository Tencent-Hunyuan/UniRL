"""``TextLMAdapter`` — the per-shape base adapter for the packed-text ``ar`` track.

Holds the conversion logic once: chat-template encoding into per-prompt
``/generate`` payloads (``build_inputs``) and the predecessor's
``build_rollout_resp`` packing decomposed into steps (``build_response`` /
``build_conditions``). The VLM adapter overrides the steps that differ.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch

from unirl.config.require import require
from unirl.rollout.engine.sglang_v2.adapters.base import (
    ModelAdapter,
    PreparedInputs,
    register_adapter,
)
from unirl.rollout.engine.sglang_v2.backends import RawResult
from unirl.rollout.engine.sglang_v2.utils import (
    ResolvedSampling,
    pack_prompt_condition,
    split_thinking_tags,
)
from unirl.types.primitives import Texts
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp, RolloutTrack
from unirl.types.segments.text import TextSegment

logger = logging.getLogger(__name__)


@register_adapter("text")
class TextLMAdapter(ModelAdapter):
    """Text-only LLM conversion (e.g. Qwen3). The base the VLM adapter derives."""

    #: The single track this engine emits.
    track_name: str = "ar"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._chat_template_logged = False

    # ------------------------------------------------------------------ #
    # build_inputs — RolloutReq → per-prompt /generate payloads
    # ------------------------------------------------------------------ #

    def build_inputs(self, req: RolloutReq, *, sampling: ResolvedSampling) -> PreparedInputs:
        prompts = self.extract_prompts(req)
        require(
            req.primitives.get("image") is None,
            f"{type(self).__name__}: req contains images but config.image_token "
            "is None (text-only mode). Set image_token in the engine config to "
            "enable VLM.",
        )

        wire: List[Dict[str, Any]] = []
        prompt_token_ids: List[List[int]] = []
        for prompt in prompts:
            payload = self.base_payload(sampling)
            ids = self.apply_chat_template(prompt, sampling.system_instruction)
            if ids is not None:
                payload["input_ids"] = ids
                prompt_token_ids.append(list(ids))
            else:
                payload["text"] = prompt
                # No chat template — encode the raw prompt so the replay's
                # prompt condition still carries the ids the server tokenized.
                prompt_token_ids.append(list(self._tokenizer.encode(prompt)))
            wire.append(payload)

        return PreparedInputs(
            wire=wire,
            prompt_token_ids=prompt_token_ids,
            resolved_n=sampling.n,
        )

    def extract_prompts(self, req: RolloutReq) -> List[str]:
        text_primitive = req.primitives.get("text")
        require(
            text_primitive is not None and isinstance(text_primitive, Texts),
            f"{type(self).__name__} requires req.primitives['text']: Texts",
        )
        prompts = list(text_primitive.texts)
        require(
            len(prompts) == int(req.batch_size),
            f"{type(self).__name__}: prompt count {len(prompts)} != req.batch_size {int(req.batch_size)}",
        )
        return prompts

    def base_payload(self, sampling: ResolvedSampling) -> Dict[str, Any]:
        """The sampling fields every ``/generate`` payload carries."""
        return {
            "sampling_params": dict(sampling.block),
            "return_logprob": sampling.return_logprob,
            "logprob_start_len": 0,
        }

    def apply_chat_template(
        self,
        user_prompt: str,
        system_instruction: Optional[str] = None,
    ) -> Optional[List[int]]:
        """Build chat-formatted ``input_ids`` via the tokenizer's chat template.

        Returns ``None`` if the chat template is unavailable or fails; the
        caller then falls back to the raw-``text`` payload variant (one-time
        warning).
        """
        if not hasattr(self._tokenizer, "apply_chat_template"):
            return None

        messages: List[Dict[str, Any]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": user_prompt})

        try:
            input_ids = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                **(self.cfg.chat_template_kwargs or {}),
            )
        except Exception as exc:
            if not self._chat_template_logged:
                self._chat_template_logged = True
                logger.warning("apply_chat_template failed, falling back to raw text: %s", exc)
            return None

        if not self._chat_template_logged:
            self._chat_template_logged = True
            decoded_preview = self._tokenizer.decode(input_ids[:30], skip_special_tokens=False)
            logger.info(
                "Chat template applied: %d tokens, preview=%r",
                len(input_ids),
                decoded_preview,
            )

        return input_ids

    # ------------------------------------------------------------------ #
    # build_response — seam results → typed RolloutResp
    # ------------------------------------------------------------------ #

    def build_response(self, req: RolloutReq, prepared: PreparedInputs, raw: List[RawResult]) -> RolloutResp:
        """Pack the seam's per-candidate results into a typed ``RolloutResp``.

        ``raw`` is in prompt-major order: candidate ``k`` of prompt ``i`` is at
        index ``i * n + k`` (the seam's ordering contract). The output rows are
        in the same order, with ``sample_indices`` pointing each row at its own
        slot. For ``n > 1`` the sample-id is mangled as ``f"{sid}#{k}"`` to keep
        uniqueness while group membership stays intact.
        """
        n = int(prepared.resolved_n)
        n_prompts = len(prepared.prompt_token_ids)
        require(
            len(raw) == n_prompts * n,
            f"{type(self).__name__}.build_response: expected {n_prompts * n} "
            f"candidates ({n_prompts} prompts × n={n}); got {len(raw)}",
        )

        decoded_texts: List[str] = []
        per_sample_tokens: List[torch.Tensor] = []
        per_sample_logprobs: List[torch.Tensor] = []
        sample_indices: List[int] = []
        sample_ids: List[str] = []
        group_ids: List[str] = []

        has_req_sids = bool(req.sample_ids)
        has_req_gids = bool(req.group_ids)

        for prompt_idx in range(n_prompts):
            base = prompt_idx * n
            req_sid = req.sample_ids[prompt_idx] if has_req_sids else f"s{prompt_idx}"
            req_gid = req.group_ids[prompt_idx] if has_req_gids else req_sid
            for k in range(n):
                r = raw[base + k]
                out_idx = base + k
                content, _reasoning = split_thinking_tags(r.text)
                decoded_texts.append(content or r.text or "")
                per_sample_tokens.append(torch.tensor(list(r.token_ids or []), dtype=torch.long))
                per_sample_logprobs.append(torch.tensor(list(r.logprobs or []), dtype=torch.float32))
                sample_indices.append(out_idx)
                sample_ids.append(f"{req_sid}#{k}" if n > 1 else req_sid)
                group_ids.append(req_gid)

        segment = TextSegment.pack(
            tokens=per_sample_tokens,
            log_probs=per_sample_logprobs,
            sample_indices=torch.tensor(sample_indices, dtype=torch.long),
        )

        return RolloutResp(
            tracks={
                self.track_name: RolloutTrack(
                    sample_ids=sample_ids,
                    parent_ids=list(group_ids) if group_ids else None,
                    conditions=self.build_conditions(prepared),
                    segment=segment,
                    decoded=Texts(texts=decoded_texts),
                ),
            }
        )

    def build_conditions(self, prepared: PreparedInputs) -> Dict[str, Any]:
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
