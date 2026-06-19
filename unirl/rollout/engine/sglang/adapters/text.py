"""``TextLMAdapter`` — the per-shape base adapter for the packed-text ``ar`` track.

Holds the conversion logic once: chat-template encoding into per-prompt
``/generate`` payloads (``build_inputs``) and the predecessor's
``build_rollout_resp`` packing fanned out per ``RolloutTrack`` field
(``build_response`` is the template; ``build_ids`` / ``build_segment`` /
``build_decoded`` / ``build_conditions`` each derive one field from
``(req, prepared, raw)``). The VLM adapter overrides the steps that differ.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

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
    pack_prompt_condition,
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

        use_template = self._has_chat_template()
        require(
            use_template or sampling.system_instruction is None,
            f"{type(self).__name__}: system_instruction is configured but the tokenizer "
            "has no chat template to render it (raw-text completion mode)",
        )

        wire: List[Dict[str, Any]] = []
        prompt_token_ids: List[List[int]] = []
        for prompt in prompts:
            payload = self.base_payload(sampling)
            if use_template:
                ids = self.apply_chat_template(prompt, sampling.system_instruction)
                payload["input_ids"] = ids
            else:
                # Raw-text completion mode — encode the raw prompt so the
                # replay's prompt condition still carries the ids the server
                # tokenized.
                payload["text"] = prompt
                ids = list(self._tokenizer.encode(prompt))
            prompt_token_ids.append(list(ids))
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
    ) -> List[int]:
        """Build chat-formatted ``input_ids`` via the tokenizer's chat template.

        Only called in templated mode (``_has_chat_template``). A failure
        raises: a set-but-broken template (bad ``chat_template_kwargs``, jinja
        error) is a config bug — silently switching the run's prompt format
        would corrupt training.
        """
        messages: List[Dict[str, Any]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": user_prompt})

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
    # build_response — the template: one fan-out stage per RolloutTrack field
    # ------------------------------------------------------------------ #

    def build_response(self, req: RolloutReq, prepared: PreparedInputs, raw: List[RawResult]) -> RolloutResp:
        """Pack the seam's per-candidate results into a typed ``RolloutResp``.

        ``raw`` is in prompt-major order: candidate ``k`` of prompt ``i`` is at
        index ``i * n + k`` (the seam's ordering contract). The count is checked
        once here; each stage then derives its field from ``(req, prepared,
        raw)`` independently, iterating ``raw`` in that shared order.
        """
        n = int(prepared.resolved_n)
        n_prompts = len(prepared.prompt_token_ids)
        require(
            len(raw) == n_prompts * n,
            f"{type(self).__name__}.build_response: expected {n_prompts * n} "
            f"candidates ({n_prompts} prompts × n={n}); got {len(raw)}",
        )

        sample_ids, group_ids = self.build_ids(req, prepared, raw)
        return RolloutResp(
            tracks={
                self.track_name: RolloutTrack(
                    sample_ids=sample_ids,
                    parent_ids=list(group_ids) if group_ids else None,
                    conditions=self.build_conditions(req, prepared, raw),
                    segment=self.build_segment(req, prepared, raw),
                    decoded=self.build_decoded(req, prepared, raw),
                ),
            }
        )

    def build_ids(self, req: RolloutReq, prepared: PreparedInputs, raw: List[RawResult]) -> Tuple[List[str], List[str]]:
        """The per-row ``(sample_ids, group_ids)``, prompt-major.

        For ``n > 1`` the sample-id is mangled as ``f"{sid}#{k}"`` to keep
        uniqueness while group membership stays intact.
        """
        n = int(prepared.resolved_n)
        has_req_sids = bool(req.sample_ids)
        has_req_gids = bool(req.group_ids)

        sample_ids: List[str] = []
        group_ids: List[str] = []
        for prompt_idx in range(len(prepared.prompt_token_ids)):
            req_sid = req.sample_ids[prompt_idx] if has_req_sids else f"s{prompt_idx}"
            req_gid = req.group_ids[prompt_idx] if has_req_gids else req_sid
            for k in range(n):
                sample_ids.append(f"{req_sid}#{k}" if n > 1 else req_sid)
                group_ids.append(req_gid)
        return sample_ids, group_ids

    def build_segment(self, req: RolloutReq, prepared: PreparedInputs, raw: List[RawResult]) -> TextSegment:
        """Pack the per-candidate tokens/logprobs, each row pointing at its own slot."""
        return TextSegment.pack(
            tokens=[torch.tensor(list(r.token_ids or []), dtype=torch.long) for r in raw],
            log_probs=[torch.tensor(list(r.logprobs or []), dtype=torch.float32) for r in raw],
        )

    def build_decoded(self, req: RolloutReq, prepared: PreparedInputs, raw: List[RawResult]) -> Texts:
        """Emit the RAW sampler text per candidate (verl-reference parity).

        Reward grading scores the full decoded response. The predecessor's
        think-stripping (``content or text``) silently dropped boxed answers
        living inside think markup — Qwen3-Base emits it organically on math —
        depressing MathBoxed rewards ~3x (observed: LIN-381 e2e #1/#2 flat at
        ~0.035 vs the b182a511-lineage v1 references at 0.09-0.25).
        """
        return Texts(texts=[r.text or "" for r in raw])

    def build_conditions(self, req: RolloutReq, prepared: PreparedInputs, raw: List[RawResult]) -> Dict[str, Any]:
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
