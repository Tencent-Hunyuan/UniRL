"""MiniMax-H3 text embedding stage -- Qwen3-VL layer-50 hidden states."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch

from unirl.config.require import require
from unirl.types.conditions import TextEmbedCondition
from unirl.types.primitives import Texts

from .vendor import MINIMAX_H3_TEXT_ENCODER_LAYER

if TYPE_CHECKING:
    from .bundle import MiniMaxH3Bundle


class MiniMaxH3TextEmbedStage:
    """Encode prompts into the conditioning MiniMax-H3 was trained on."""

    def __init__(self, bundle: "MiniMaxH3Bundle") -> None:
        self.text_encoder = bundle.text_encoder
        self.processor = bundle.processor
        self.tokenizer = bundle.tokenizer
        self.dtype = bundle.dtype
        self.device = bundle.device

    @property
    def _encoder_device(self) -> torch.device:
        return next(self.text_encoder.parameters()).device

    @property
    def _decoder(self):
        """The decoder stack that owns ``.layers``."""
        model = self.text_encoder.model
        return getattr(model, "language_model", model)

    @torch.no_grad()
    def embed(self, texts: Texts) -> TextEmbedCondition:
        """Encode one batch of prompts into a ``TextEmbedCondition``."""
        # ``Texts.texts`` is the raw list[str]; ``Texts.to_list()`` returns
        # list[Text] dataclass wrappers, which the tokenizer rejects. Same
        # accessor ltx2 and wan21 use.
        prompts: List[str] = list(texts.texts)
        require(len(prompts) > 0, "MiniMaxH3TextEmbedStage: no prompts to embed")

        num_layers = len(self._decoder.layers)
        require(
            num_layers > MINIMAX_H3_TEXT_ENCODER_LAYER,
            f"MiniMaxH3TextEmbedStage: MiniMax-H3 conditions on hidden_states[{MINIMAX_H3_TEXT_ENCODER_LAYER}] of "
            f"its Qwen3-VL conditioner, which needs more than {MINIMAX_H3_TEXT_ENCODER_LAYER} decoder layers, but "
            f"the loaded conditioner has {num_layers}.",
        )

        encoder_device = self._encoder_device
        embeds = []
        for prompt in prompts:
            token_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=encoder_device)
            # Qwen3-VL lays its 3D rotary positions out per modality run, read
            # off the token type ids the processor derives (0 text, 1 image,
            # 2 video). Text-only here, but the conditioner still wants them.
            mm_token_type_ids = torch.tensor(
                self.processor.create_mm_token_type_ids([token_ids]), dtype=torch.long, device=encoder_device
            )
            outputs = self.text_encoder.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                mm_token_type_ids=mm_token_type_ids,
                use_cache=False,
                output_hidden_states=True,
            )
            embeds.append(outputs.hidden_states[MINIMAX_H3_TEXT_ENCODER_LAYER].to(device=self.device, dtype=self.dtype))

        lengths = {int(e.shape[1]) for e in embeds}
        require(
            len(lengths) == 1,
            f"MiniMaxH3TextEmbedStage: prompts tokenized to differing lengths {sorted(lengths)}. The packed sequence "
            f"geometry must be identical across the batch (LatentSegment stores latents in a CONCAT field), so a "
            f"mixed-length batch cannot be packed. Pad or group prompts by token length upstream.",
        )
        text_embeds = torch.cat(embeds, dim=0)
        return TextEmbedCondition(
            embeds=text_embeds,
            attn_mask=torch.ones(text_embeds.shape[:2], dtype=torch.bool, device=text_embeds.device),
        )


__all__ = ["MiniMaxH3TextEmbedStage"]
