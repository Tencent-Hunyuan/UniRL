from types import SimpleNamespace

import pytest
import torch

from unirl.models.hunyuan_video.config import HunyuanVideoPipelineConfig
from unirl.models.hunyuan_video.text_embed import HunyuanVideoTextEmbedStage


class _Tokenizer:
    def __call__(self, prompts, *, max_length, **kwargs):
        shape = (len(prompts), max_length)
        return SimpleNamespace(
            input_ids=torch.zeros(shape, dtype=torch.long),
            attention_mask=torch.ones(shape, dtype=torch.long),
        )


class _TextEncoder:
    def parameters(self):
        yield torch.zeros((), dtype=torch.float32)

    def __call__(self, *, input_ids, **kwargs):
        shape = (*input_ids.shape, 1)
        return SimpleNamespace(hidden_states=tuple(torch.full(shape, float(i)) for i in range(4)))


def _bundle():
    return SimpleNamespace(
        tokenizer=_Tokenizer(),
        text_encoder=_TextEncoder(),
        device=torch.device("cpu"),
    )


@pytest.mark.parametrize(("skip", "expected"), [(0, 3.0), (2, 1.0)])
def test_llama_hidden_state_skip_selects_from_end(skip, expected):
    stage = HunyuanVideoTextEmbedStage(_bundle(), crop_start=0, hidden_state_skip_layer=skip)

    embeds, _ = stage._encode_llama(["prompt"])

    assert torch.all(embeds == expected)


def test_hidden_state_skip_rejects_negative_values():
    with pytest.raises(ValueError, match="hidden_state_skip_layer must be >= 0"):
        HunyuanVideoPipelineConfig(
            pretrained_model_ckpt_path="unused",
            hidden_state_skip_layer=-1,
        )

    with pytest.raises(ValueError, match="hidden_state_skip_layer must be >= 0"):
        HunyuanVideoTextEmbedStage(_bundle(), hidden_state_skip_layer=-1)


def test_llama_hidden_state_skip_rejects_missing_layer():
    stage = HunyuanVideoTextEmbedStage(_bundle(), crop_start=0, hidden_state_skip_layer=4)

    with pytest.raises(ValueError, match="encoder returned 4 hidden states"):
        stage._encode_llama(["prompt"])
