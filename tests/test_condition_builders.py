from __future__ import annotations

from typing import Type

import pytest
import torch

from unirl.models.qwen_image.pipeline import QwenImagePipeline
from unirl.models.sd3.pipeline import SD3Pipeline
from unirl.models.types.pipeline import Pipeline
from unirl.types.conditions import TextEmbedCondition
from unirl.types.primitives import Texts


class _RecordingTextEmbed:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: Texts) -> TextEmbedCondition:
        values = list(texts.texts)
        self.calls.append(values)
        return TextEmbedCondition(
            embeds=torch.zeros(len(values), 1, 2),
            attn_mask=torch.ones(len(values), 1),
        )


@pytest.mark.parametrize(
    ("pipeline_cls", "default_negative"),
    [
        (SD3Pipeline, ""),
        (QwenImagePipeline, " "),
    ],
)
def test_public_condition_builders_handle_cfg_and_explicit_negatives(
    pipeline_cls: Type[Pipeline],
    default_negative: str,
) -> None:
    pipeline = pipeline_cls.__new__(pipeline_cls)
    text_embed = _RecordingTextEmbed()
    pipeline.text_embed = text_embed
    prompts = Texts(texts=["first", "second"])

    cfg_off = pipeline.build_conditions(prompts, guidance_scale=1.0)
    assert cfg_off.negative_text is None
    assert text_embed.calls == [["first", "second"]]

    text_embed.calls.clear()
    cfg_on = pipeline.build_conditions(prompts, guidance_scale=2.0)
    assert cfg_on.negative_text is not None
    assert text_embed.calls == [["first", "second"], [default_negative, default_negative]]

    text_embed.calls.clear()
    explicit = Texts(texts=["no blur", "no watermark"])
    with_explicit = pipeline.build_conditions(prompts, negatives=explicit, guidance_scale=1.0)
    assert with_explicit.negative_text is not None
    assert text_embed.calls == [["first", "second"], ["no blur", "no watermark"]]

    with pytest.raises(ValueError, match="negative_text length"):
        pipeline.build_conditions(prompts, negatives=Texts(texts=["too short"]), guidance_scale=2.0)
