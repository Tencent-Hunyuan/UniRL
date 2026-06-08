"""SD3TextEmbedStage — triple-encoder text → TextEmbedCondition.

Implements ``EmbedStage[Texts, TextEmbedCondition]``. Runs the SD3 triple-encoder
stack (CLIP-L + CLIP-G + T5-XXL) following diffusers'
``StableDiffusion3Pipeline.encode_prompt`` byte-for-byte at the spec level:

- ``pooled`` = ``cat([clip1.text_embeds, clip2.text_embeds], dim=-1)`` — fed
  to the SD3 transformer's ``pooled_projections`` head.
- ``embeds`` = ``cat([clip_merged_padded, t5_last_hidden], dim=-2)`` where
  ``clip_merged = cat([clip1.hidden_states[-2], clip2.hidden_states[-2]],
  dim=-1)`` is zero-padded along the last dim to match T5's hidden dim — fed
  to ``encoder_hidden_states``. Using T5 alone here (without the CLIP
  penultimate hidden states) systematically shifts rewards relative to the
  SGLang/vLLM-omni rollout path, which follows the diffusers spec.

The stage is strictly unary (matches the ``EmbedStage[P, C]`` Protocol).
For CFG, the pipeline calls ``embed`` twice — once for positive prompts,
once for negatives — and assembles both branches into the model-specific
``SD3Conditions(text=pos, negative_text=neg)`` container.

Triple-encoder math mirrors the upstream SD3 ``_encode_prompt`` spec.
"""

from __future__ import annotations

from typing import List

import torch

from unirl.models.types.embedding import EmbedStage
from unirl.types.conditions import TextEmbedCondition
from unirl.types.primitives import Texts

from .bundle import SD3Bundle


class SD3TextEmbedStage(EmbedStage[Texts, TextEmbedCondition]):
    """SD3 triple-encoder text → TextEmbedCondition stage."""

    def __init__(
        self,
        bundle: SD3Bundle,
        *,
        max_sequence_length: int = 256,
        clip_max_length: int = 77,
    ) -> None:
        self.bundle = bundle
        self.max_sequence_length = max_sequence_length
        self.clip_max_length = clip_max_length

    def embed(self, p: Texts) -> TextEmbedCondition:
        """Encode prompts into a ``TextEmbedCondition``.

        LIN-387: GRPO requests arrive samples_per_prompt-expanded (e.g. 48
        strings = 3 unique × K=16), so encode each unique prompt once and
        gather rows back to the request order — the duplicate CLIP×2 + T5-XXL
        forwards are pure waste (~15/16 of encode time at K=16).
        """
        prompts = list(p.texts)
        index_of: dict[str, int] = {}
        inverse: list[int] = []
        uniq: list[str] = []
        for s in prompts:
            i = index_of.get(s)
            if i is None:
                i = len(uniq)
                index_of[s] = i
                uniq.append(s)
            inverse.append(i)
        cond = self._encode(uniq)
        if len(uniq) == len(prompts):
            return cond
        inv = torch.tensor(inverse, device=cond.embeds.device)
        return TextEmbedCondition(
            embeds=cond.embeds.index_select(0, inv),
            pooled=cond.pooled.index_select(0, inv),
        )

    def _encode(self, prompts: List[str]) -> TextEmbedCondition:
        bundle = self.bundle
        device = bundle.device

        # CLIP 1 — collect pooled (for ``pooled_projections``) AND the
        # penultimate hidden state (for ``encoder_hidden_states``, per the
        # diffusers SD3 spec with clip_skip=None default).
        clip1_inputs = bundle.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.clip_max_length,
            truncation=True,
            return_tensors="pt",
        )
        clip1_ids = clip1_inputs.input_ids.to(device)
        with torch.no_grad():
            clip1_out = bundle.text_encoder(clip1_ids, output_hidden_states=True)
            pooled_1 = clip1_out.text_embeds
            clip1_hidden = clip1_out.hidden_states[-2]

        # CLIP 2 — same shape contract as CLIP 1.
        clip2_inputs = bundle.tokenizer_2(
            prompts,
            padding="max_length",
            max_length=self.clip_max_length,
            truncation=True,
            return_tensors="pt",
        )
        clip2_ids = clip2_inputs.input_ids.to(device)
        with torch.no_grad():
            clip2_out = bundle.text_encoder_2(clip2_ids, output_hidden_states=True)
            pooled_2 = clip2_out.text_embeds
            clip2_hidden = clip2_out.hidden_states[-2]

        pooled = torch.cat([pooled_1, pooled_2], dim=-1)

        # T5
        t5_inputs = bundle.tokenizer_3(
            prompts,
            padding="max_length",
            max_length=self.max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        t5_ids = t5_inputs.input_ids.to(device)
        with torch.no_grad():
            t5_out = bundle.text_encoder_3(t5_ids)
            t5_embeds = t5_out.last_hidden_state

        # Build ``encoder_hidden_states`` per diffusers SD3 spec:
        #   cat([cat([clip1_h, clip2_h], dim=-1) padded to t5_dim,
        #        t5_last_hidden_state],
        #       dim=-2)
        # Shape: [B, clip_seq + t5_seq, t5_dim].
        clip_merged = torch.cat([clip1_hidden, clip2_hidden], dim=-1)
        clip_merged = torch.nn.functional.pad(clip_merged, (0, t5_embeds.shape[-1] - clip_merged.shape[-1]))
        embeds = torch.cat([clip_merged, t5_embeds], dim=-2)

        return TextEmbedCondition(
            embeds=embeds,
            pooled=pooled,
        )


__all__ = ["SD3TextEmbedStage"]
