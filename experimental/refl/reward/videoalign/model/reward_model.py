"""Qwen2-VL reward model with a 3-dim (VQ, MQ, TA) regression head."""

from __future__ import annotations

from typing import Any, List, Optional

import torch
import torch.nn as nn
from transformers import Qwen2VLForConditionalGeneration


def _cfg_get(config: Any, name: str) -> Any:
    """Read a ``Qwen2VLConfig`` field: media token ids sit on the top-level config, LM fields under ``text_config``."""
    if hasattr(config, name):
        val = getattr(config, name)
        if val is not None:
            return val
    text_cfg = getattr(config, "text_config", None)
    if text_cfg is not None and hasattr(text_cfg, name):
        return getattr(text_cfg, name)
    if hasattr(config, name):
        return getattr(config, name)
    raise AttributeError(f"{type(config).__name__} has no attribute {name!r} (checked top-level and .text_config).")


class Qwen2VLRewardModelBT(Qwen2VLForConditionalGeneration):
    """Qwen2-VL backbone + linear reward head; ``forward`` returns ``{'logits': Tensor[B, output_dim]}``."""

    def __init__(
        self,
        config,
        output_dim: int = 4,
        reward_token: str = "last",
        special_token_ids: Optional[List[int]] = None,
    ) -> None:
        super().__init__(config)
        self.output_dim = output_dim
        hidden_size = _cfg_get(config, "hidden_size")
        self.rm_head = nn.Linear(hidden_size, output_dim, bias=False)
        self.reward_token = reward_token

        self.special_token_ids = special_token_ids
        if self.special_token_ids is not None:
            self.reward_token = "special"

    @property
    def visual(self):  # type: ignore[override]
        inner = self._modules.get("model", None)
        visual = getattr(inner, "visual", None) if inner is not None else None
        if visual is None:
            raise AttributeError(
                "Qwen2VLRewardModelBT: vision tower not found at "
                "self.model.visual — this code targets the locked "
                "transformers 5.6 stack; align the environment."
            )
        return visual

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,  # noqa: ARG002 — kept for API parity
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,  # noqa: ARG002 — kept for API parity
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        def _as_tensor(visual_out):
            t = getattr(visual_out, "pooler_output", None)
            if t is None:
                raise TypeError(
                    "Qwen2-VL vision tower returned "
                    f"{type(visual_out).__name__} with no pooler_output. This "
                    "code targets the locked transformers 5.6 stack — align "
                    "the environment instead of widening this path."
                )
            return t

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype())
                image_embeds = _as_tensor(self.visual(pixel_values, grid_thw=image_grid_thw))
                image_token_id = _cfg_get(self.config, "image_token_id")
                image_mask = (input_ids == image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                video_embeds = _as_tensor(self.visual(pixel_values_videos, grid_thw=video_grid_thw))
                video_token_id = _cfg_get(self.config, "video_token_id")
                video_mask = (input_ids == video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.rm_head(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        try:
            pad_token_id = _cfg_get(self.config, "pad_token_id")
        except AttributeError:
            pad_token_id = None
        if pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = torch.eq(input_ids, pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        if self.reward_token == "last":
            pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
        elif self.reward_token == "mean":
            valid_lengths = torch.clamp(sequence_lengths, min=0, max=logits.size(1) - 1)
            pooled_logits = torch.stack([logits[i, : valid_lengths[i]].mean(dim=0) for i in range(batch_size)])
        elif self.reward_token == "special":
            special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for special_token_id in self.special_token_ids:
                special_token_mask = special_token_mask | (input_ids == special_token_id)
            pooled_logits = logits[special_token_mask, ...]
            pooled_logits = pooled_logits.view(batch_size, 3, -1)
            if self.output_dim == 3:
                pooled_logits = pooled_logits.diagonal(dim1=1, dim2=2)
            pooled_logits = pooled_logits.view(batch_size, -1)
        else:
            raise ValueError(f"Invalid reward_token: {self.reward_token!r}")

        return {"logits": pooled_logits}


__all__ = ["Qwen2VLRewardModelBT"]
