"""CPU contract tests for Edit-Plus source-image prompt conditioning."""

from __future__ import annotations

from types import SimpleNamespace

from unirl.models.qwen_image_edit_plus.pipeline import QwenImageEditPlusPipeline
from unirl.types.primitives import Images, Texts


class _TextEmbed:
    def __init__(self) -> None:
        self.calls = []

    def embed(self, texts, images=None):
        self.calls.append((list(texts.texts), images))
        return SimpleNamespace(texts=list(texts.texts), images=images)


def _pipeline(*, use_condition_image_prompt: bool):
    pipeline = object.__new__(QwenImageEditPlusPipeline)
    pipeline.text_embed = _TextEmbed()
    pipeline.use_condition_image_prompt = use_condition_image_prompt
    return pipeline


def test_edit_plus_conditions_both_cfg_branches_on_source_images() -> None:
    pipeline = _pipeline(use_condition_image_prompt=True)
    texts = Texts(texts=["edit"])
    images = Images(pixels=None)

    conditions = pipeline.build_conditions(
        texts,
        images=images,
        guidance_scale=2.0,
    )

    assert [call[0] for call in pipeline.text_embed.calls] == [["edit"], [" "]]
    assert all(call[1] is images for call in pipeline.text_embed.calls)
    assert conditions.text.images is images
    assert conditions.negative_text.images is images


def test_edit_plus_toggle_only_removes_images_from_text_encoder() -> None:
    pipeline = _pipeline(use_condition_image_prompt=False)
    texts = Texts(texts=["edit"])
    images = Images(pixels=None)

    conditions = pipeline.build_conditions(
        texts,
        negatives=Texts(texts=["do not change"]),
        images=images,
        guidance_scale=2.0,
    )

    assert [call[1] for call in pipeline.text_embed.calls] == [None, None]
    assert conditions.text.images is None
    assert conditions.negative_text.images is None
