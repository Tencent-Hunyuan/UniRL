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
    """Encode prompts into the conditioning MiniMax-H3 was trained on.

    Three things differ from an ordinary text-encoder stage, and all three are
    checkpoint contracts rather than preferences:

    * The conditioning is ``hidden_states[50]`` of the Qwen3-VL conditioner --
      an intermediate, **unnormalized** hidden state, not ``last_hidden_state``.
      The final layer is post-norm and is not what H3 consumes.
    * There is no chat template. The prompt is tokenized raw with
      ``add_special_tokens=False``.
    * The call goes to ``text_encoder.model``, the decoder submodule, not to
      ``text_encoder``. H3 never uses the language-model head, and skipping it
      avoids a vocabulary-wide projection over every token.
    """

    def __init__(self, bundle: "MiniMaxH3Bundle") -> None:
        self.text_encoder = bundle.text_encoder
        self.processor = bundle.processor
        self.tokenizer = bundle.tokenizer
        self.hidden_layer = bundle.text_encoder_hidden_layer
        self.dtype = bundle.dtype
        self.device = bundle.device

    @torch.no_grad()
    def embed(self, texts: Texts) -> TextEmbedCondition:
        """Encode one batch of prompts into a ``TextEmbedCondition``.

        t2va is text-only, so every row of the packed text block carries the
        text tag and the caller can derive the token count from the embedding
        length. (``fl2va`` interleaves a keyframe vision block whose rows are
        tagged VIDEO -- that arrives with keyframes, not before.)
        """
        prompts: List[str] = list(texts.to_list()) if hasattr(texts, "to_list") else list(texts)
        require(len(prompts) > 0, "MiniMaxH3TextEmbedStage: no prompts to embed")

        num_layers = len(self.text_encoder.model.layers)
        require(
            num_layers > MINIMAX_H3_TEXT_ENCODER_LAYER,
            f"MiniMaxH3TextEmbedStage: MiniMax-H3 conditions on hidden_states[{MINIMAX_H3_TEXT_ENCODER_LAYER}] of "
            f"its Qwen3-VL conditioner, which needs more than {MINIMAX_H3_TEXT_ENCODER_LAYER} decoder layers, but "
            f"the loaded conditioner has {num_layers}.",
        )

        embeds = []
        for prompt in prompts:
            token_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
            # Qwen3-VL lays its 3D rotary positions out per modality run, read
            # off the token type ids the processor derives (0 text, 1 image,
            # 2 video). Text-only here, but the conditioner still wants them.
            mm_token_type_ids = torch.tensor(
                self.processor.create_mm_token_type_ids([token_ids]), dtype=torch.long, device=self.device
            )
            outputs = self.text_encoder.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                mm_token_type_ids=mm_token_type_ids,
                use_cache=False,
                output_hidden_states=True,
            )
            embeds.append(outputs.hidden_states[self.hidden_layer].to(device=self.device, dtype=self.dtype))

        lengths = {int(e.shape[1]) for e in embeds}
        require(
            len(lengths) == 1,
            f"MiniMaxH3TextEmbedStage: prompts tokenized to differing lengths {sorted(lengths)}. The packed sequence "
            f"geometry must be identical across the batch (LatentSegment stores latents in a CONCAT field), so a "
            f"mixed-length batch cannot be packed. Pad or group prompts by token length upstream.",
        )
        return TextEmbedCondition(embeds=torch.cat(embeds, dim=0))


__all__ = ["MiniMaxH3TextEmbedStage"]
