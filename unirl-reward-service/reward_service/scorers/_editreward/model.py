"""Qwen2.5-VL Reward Model with multi-head support.

Vendored from EditReward/model/qwen2_5_vl_trainer.py — only the model class,
no trainer/callback/dataset code. All heavy training imports removed.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration


class Qwen2_5_VLRewardModelBT_MultiHead(Qwen2_5_VLForConditionalGeneration):
    def __init__(
        self,
        config,
        output_dim=4,
        reward_token="last",
        special_token_ids=None,
        rm_head_type="default",
        rm_head_kwargs=None,
        pooling_strategy="min",
    ):
        super().__init__(config)
        self.output_dim = output_dim
        self.rm_head_type = rm_head_type
        self.reward_token = reward_token
        self.special_token_ids = special_token_ids
        self.pooling_strategy = pooling_strategy

        self.rm_head = None
        self.rm_heads = None

        if rm_head_type == "default":
            self.rm_head = nn.Linear(config.hidden_size, output_dim, bias=False)

        elif rm_head_type in ("ranknet", "ranknet_share_head"):
            if rm_head_kwargs is not None:
                layers = []
                num_layers = rm_head_kwargs.get("num_layers", 3)
                for layer in range(num_layers):
                    if layer == 0:
                        layers = [
                            nn.Linear(config.hidden_size, rm_head_kwargs["hidden_size"]),
                            nn.ReLU(),
                            nn.Dropout(rm_head_kwargs.get("dropout", 0.1)),
                        ]
                    elif layer < num_layers - 1:
                        layers += [
                            nn.Linear(rm_head_kwargs["hidden_size"], rm_head_kwargs["hidden_size"]),
                            nn.ReLU(),
                            nn.Dropout(rm_head_kwargs.get("dropout", 0.1)),
                        ]
                    else:
                        layers.append(
                            nn.Linear(rm_head_kwargs["hidden_size"], output_dim, bias=rm_head_kwargs.get("bias", False))
                        )
                self.rm_head = nn.Sequential(*layers)
            else:
                self.rm_head = nn.Sequential(
                    nn.Linear(config.hidden_size, 1024),
                    nn.ReLU(),
                    nn.Dropout(0.05),
                    nn.Linear(1024, 16),
                    nn.ReLU(),
                    nn.Linear(16, output_dim),
                )

        elif rm_head_type in ("ranknet_multi_head", "ranknet_multi_head_regression"):
            num_heads = 2
            if rm_head_kwargs is not None:
                num_heads = rm_head_kwargs.get("num_heads", 2)
            self.rm_heads = nn.ModuleDict()
            for i in range(num_heads):
                if rm_head_kwargs is not None:
                    head_layers = []
                    num_layers = rm_head_kwargs.get("num_layers", 3)
                    for layer in range(num_layers):
                        if layer == 0:
                            head_layers += [
                                nn.Linear(config.hidden_size, rm_head_kwargs["hidden_size"]),
                                nn.ReLU(),
                                nn.Dropout(rm_head_kwargs.get("dropout", 0.1)),
                            ]
                        elif layer < num_layers - 1:
                            head_layers += [
                                nn.Linear(rm_head_kwargs["hidden_size"], rm_head_kwargs["hidden_size"]),
                                nn.ReLU(),
                                nn.Dropout(rm_head_kwargs.get("dropout", 0.1)),
                            ]
                        else:
                            head_layers.append(
                                nn.Linear(rm_head_kwargs["hidden_size"], output_dim, bias=rm_head_kwargs.get("bias", False))
                            )
                    self.rm_heads[f"head_{i}"] = nn.Sequential(*head_layers)
                else:
                    self.rm_heads[f"head_{i}"] = nn.Sequential(
                        nn.Linear(config.hidden_size, 1024),
                        nn.ReLU(),
                        nn.Dropout(0.05),
                        nn.Linear(1024, 16),
                        nn.ReLU(),
                        nn.Linear(16, output_dim),
                    )

        if self.rm_head is not None:
            self.rm_head.to(torch.float32)
        if self.rm_heads is not None:
            for h in self.rm_heads.values():
                h.to(torch.float32)

        if self.special_token_ids is not None:
            self.reward_token = "special"

    def _pool_logits(self, logits, input_ids, batch_size):
        if self.reward_token == "last":
            if self.config.pad_token_id is None:
                sequence_lengths = -1
            else:
                sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1)
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            return logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        elif self.reward_token == "mean":
            if self.config.pad_token_id is None:
                return logits.mean(dim=1)
            else:
                sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1)
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
                valid_lengths = torch.clamp(sequence_lengths, min=0, max=logits.size(1) - 1)
                return torch.stack([logits[i, :valid_lengths[i]].mean(dim=0) for i in range(batch_size)])

        elif self.reward_token == "special":
            special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for special_token_id in self.special_token_ids:
                special_token_mask = special_token_mask | (input_ids == special_token_id)
            pooled = logits[special_token_mask, ...]
            return pooled.view(batch_size, -1)

        else:
            raise ValueError(f"Invalid reward_token: {self.reward_token}")

    def _run_single_batch_through_model_and_head(self, batch_dict, head_module):
        input_ids = batch_dict.get("input_ids", None)
        attention_mask = batch_dict.get("attention_mask", None)
        position_ids = batch_dict.get("position_ids", None)
        past_key_values = batch_dict.get("past_key_values", None)
        inputs_embeds = batch_dict.get("inputs_embeds", None)
        pixel_values = batch_dict.get("pixel_values", None)
        pixel_values_videos = batch_dict.get("pixel_values_videos", None)
        image_grid_thw = batch_dict.get("image_grid_thw", None)
        video_grid_thw = batch_dict.get("video_grid_thw", None)
        use_cache = batch_dict.get("use_cache", None)
        output_attentions = batch_dict.get("output_attentions", None)
        output_hidden_states = batch_dict.get("output_hidden_states", None)
        return_dict = batch_dict.get("return_dict", True)

        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds must be provided")
            inputs_embeds = self.get_input_embeddings()(input_ids)

            if pixel_values is not None:
                if image_grid_thw is not None:
                    image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                else:
                    image_embeds = self.visual(pixel_values)
                image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                if video_grid_thw is not None:
                    video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                else:
                    video_embeds = self.visual(pixel_values_videos)
                video_mask = (input_ids == self.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
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
            output_attentions=output_attentions if output_attentions is not None else self.config.output_attentions,
            output_hidden_states=output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        batch_size = input_ids.shape[0]

        with torch.autocast(device_type="cuda", dtype=torch.float32):
            logits = head_module(hidden_states)

        return self._pool_logits(logits, input_ids, batch_size)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        rope_deltas=None,
        batch_dim1=None,
        batch_dim2=None,
        **kwargs,
    ):
        # Single head
        if self.rm_head_type == "ranknet":
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
                if pixel_values is not None:
                    image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw) \
                        if image_grid_thw is not None else self.visual(pixel_values)
                    image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds.to(inputs_embeds))
                if pixel_values_videos is not None:
                    video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw) \
                        if video_grid_thw is not None else self.visual(pixel_values_videos)
                    video_mask = (input_ids == self.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                    inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds.to(inputs_embeds))
                if attention_mask is not None:
                    attention_mask = attention_mask.to(inputs_embeds.device)

            outputs = self.model(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions if output_attentions is not None else self.config.output_attentions,
                output_hidden_states=output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states,
                return_dict=return_dict if return_dict is not None else self.config.use_return_dict,
            )
            hidden_states = outputs[0]
            batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]

            with torch.autocast(device_type="cuda", dtype=torch.float32):
                logits = self.rm_head(hidden_states)

            pooled = self._pool_logits(logits, input_ids, batch_size)
            return {"logits": pooled}

        # Multi head
        elif self.rm_head_type in ("ranknet_multi_head", "ranknet_multi_head_regression"):
            num_heads = len(self.rm_heads)
            head_batches = [None] * num_heads

            provided_batches = {**kwargs}
            if batch_dim1 is not None:
                provided_batches["batch_dim1"] = batch_dim1
            if batch_dim2 is not None:
                provided_batches["batch_dim2"] = batch_dim2

            for i in range(num_heads):
                key = f"batch_dim{i+1}"
                if key in provided_batches:
                    head_batches[i] = provided_batches[key]

            if all(b is None for b in head_batches):
                raise ValueError(
                    f"No per-head batches found. Expected keys: {[f'batch_dim{j+1}' for j in range(num_heads)]}"
                )

            logits_per_head = []
            for i, b in enumerate(head_batches):
                if b is not None:
                    logits = self._run_single_batch_through_model_and_head(b, self.rm_heads[f"head_{i}"])
                    logits_per_head.append(logits)

            if self.pooling_strategy is None:
                return logits_per_head

            stacked = torch.stack(logits_per_head, dim=0)
            if self.pooling_strategy == "min":
                final_logits = stacked.min(dim=0).values
            elif self.pooling_strategy == "mean":
                final_logits = stacked.mean(dim=0)
            elif self.pooling_strategy == "sum":
                means = stacked[:, :, 0]
                sigmas = torch.exp(stacked[:, :, 1])
                final_mean = means.sum(dim=0)
                final_var = (sigmas ** 2).sum(dim=0)
                final_sigma = torch.sqrt(final_var)
                final_logits = torch.stack([final_mean, torch.log(final_sigma)], dim=-1)
            else:
                final_logits = stacked.mean(dim=0)

            return {"logits": final_logits}

        # Shared head
        elif self.rm_head_type == "ranknet_share_head":
            num_heads = 2
            head_batches = [None] * num_heads

            provided_batches = {**kwargs}
            if batch_dim1 is not None:
                provided_batches["batch_dim1"] = batch_dim1
            if batch_dim2 is not None:
                provided_batches["batch_dim2"] = batch_dim2

            for i in range(num_heads):
                key = f"batch_dim{i+1}"
                if key in provided_batches:
                    head_batches[i] = provided_batches[key]

            if all(b is None for b in head_batches):
                raise ValueError(
                    f"No per-head batches found. Expected keys: {[f'batch_dim{j+1}' for j in range(num_heads)]}"
                )

            logits_per_head = []
            for i, b in enumerate(head_batches):
                if b is not None:
                    logits = self._run_single_batch_through_model_and_head(b, self.rm_head)
                    logits_per_head.append(logits)

            if self.pooling_strategy is None:
                return logits_per_head

            stacked = torch.stack(logits_per_head, dim=0)
            if self.pooling_strategy == "min":
                final_logits = stacked.min(dim=0).values
            elif self.pooling_strategy == "mean":
                final_logits = stacked.mean(dim=0)
            else:
                final_logits = stacked.mean(dim=0)

            return {"logits": final_logits}
