"""WAN21TextEmbedStage — UMT5 prompt encoding → TextEmbedCondition.

Implements ``EmbedStage[Texts, TextEmbedCondition]``. Runs WAN's single
UMT5 (falls back to T5) encoder over a list of prompts and emits a
``TextEmbedCondition`` where:

- ``embeds`` is the encoder's ``last_hidden_state`` with **padded
  positions explicitly zeroed** via ``embeds *= attn_mask.unsqueeze(-1)``.
  This is the WAN training-time convention; skipping it shifts the
  distribution that the diffusion transformer sees relative to how it
  was trained. (Mirrors ``WANTextEncoderWrapper.encode_prompt`` in
  ``unirl/models/wan21.py:687-690``.)
- ``pooled`` is ``None`` — UMT5 doesn't emit a pooled vector and WAN's
  transformer doesn't consume one.
- ``attn_mask`` is preserved on the condition for transport even though
  the diffusion stage doesn't consult it (the masking is already baked
  into ``embeds``); downstream stages or weight-sync hooks may rely on
  it.

The stage is strictly unary (matches the ``EmbedStage[P, C]`` Protocol).
For CFG, the pipeline calls ``embed`` twice — once for positive prompts,
once for negatives — and assembles both branches into
``WAN21Conditions(text=pos, negative_text=neg)``.

UMT5 math mirrors the in-repo reference at
``samplers/fsdp/wan_sampler.py::FSDPWanSampler._encode_prompt`` (legacy)
and ``unirl/models/wan21.py:665-697`` (WANTextEncoderWrapper) but
is intentionally re-inlined here: this module does not import
legacy code, so the two encoders must stay in spec sync via review /
test, not via shared helpers.
"""

from __future__ import annotations

from typing import Any, List, Protocol, runtime_checkable

import torch

from unirl.models.types.embedding import EmbedStage
from unirl.types.conditions import TextEmbedCondition
from unirl.types.primitives import Texts


@runtime_checkable
class _TextEncoderBundle(Protocol):
    """Structural Protocol for bundles this stage can encode against.

    The stage only needs the four surfaces below — frozen text encoder,
    tokenizer, target device, and the WAN UMT5 max sequence length. Both
    :class:`unirl.models.wan21.bundle.WAN21Bundle` and
    :class:`unirl.models.wan22.bundle.WAN22Bundle` satisfy
    this Protocol structurally, which is why :class:`WAN22Pipeline` can
    plug a :class:`WAN22Bundle` into this same stage without subclassing.

    Defining the Protocol locally (rather than importing a specific
    bundle class) keeps the stage genuinely model-agnostic at the text
    encoding layer — only the surfaces this code actually consumes are
    written down.
    """

    text_encoder: Any
    tokenizer: Any
    device: torch.device
    max_sequence_length: int


class WAN21TextEmbedStage(EmbedStage[Texts, TextEmbedCondition]):
    """WAN 2.1 UMT5 text → TextEmbedCondition stage."""

    def __init__(
        self,
        bundle: _TextEncoderBundle,
        *,
        max_sequence_length: int = 512,
    ) -> None:
        self.bundle = bundle
        # Caller can override; defaults pull from the config-time setting
        # cached on the bundle (which is in turn set by
        # ``WAN21PipelineConfig.max_sequence_length``).
        self.max_sequence_length = int(
            max_sequence_length if max_sequence_length is not None else bundle.max_sequence_length
        )

    def embed(self, p: Texts) -> TextEmbedCondition:
        """Encode prompts into a ``TextEmbedCondition``."""
        return self._encode(list(p.texts))

    def _encode(self, prompts: List[str]) -> TextEmbedCondition:
        bundle = self.bundle
        device = bundle.device

        text_inputs = bundle.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(device)
        attention_mask = text_inputs.attention_mask.to(device)

        with torch.no_grad():
            encoder_out = bundle.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            embeds = encoder_out.last_hidden_state

        # WAN-specific padding policy: zero out padded positions in the
        # encoder output BEFORE the diffusion transformer sees it. This
        # is the training-time convention from WAN's reference
        # implementation; skipping it shifts the distribution and
        # produces systematically different rewards from rollout (which
        # always applies the mask).
        embeds = embeds * attention_mask.unsqueeze(-1).to(dtype=embeds.dtype)

        return TextEmbedCondition(
            embeds=embeds,
            pooled=None,
            attn_mask=attention_mask,
        )


__all__ = ["WAN21TextEmbedStage"]
