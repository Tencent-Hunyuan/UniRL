from types import SimpleNamespace

import pytest
import torch

from unirl.rollout.engine.sglang_diffusion.adapters.video import (
    HunyuanVideoAdapter,
    VideoAdapter,
)
from unirl.types.conditions.text import TextEmbedCondition


def _raw_result(*, seq_len: int, batch_size: int = 1):
    return SimpleNamespace(
        prompt_embeds=[
            torch.randn(batch_size, seq_len, 4096),
            torch.randn(batch_size, 1, 768),
        ],
        encoder_attention_mask=[
            torch.ones(batch_size, seq_len, dtype=torch.long),
            torch.ones(batch_size, 1, dtype=torch.long),
        ],
        samples=torch.rand(3, 5, 8, 8),
    )


def _adapter():
    config = SimpleNamespace(populate_conditions=True, target_modules=None)
    model_config = SimpleNamespace(
        pretrained_model_ckpt_path="unused",
        shift=5.0,
        weight_sync_param_name_prefix="transformer.",
    )
    return HunyuanVideoAdapter(config, model_config)


def test_hunyuan_adapter_emits_video_and_separate_dual_text_conditions():
    adapter = _adapter()
    assert isinstance(adapter, VideoAdapter)

    conditions = adapter.build_condition(
        [
            _raw_result(seq_len=3),
            _raw_result(seq_len=5),
        ]
    )

    assert set(conditions) == {"text_llama", "pooled_clip"}
    assert isinstance(conditions["text_llama"], TextEmbedCondition)
    assert conditions["text_llama"].embeds.shape == (2, 5, 4096)
    assert conditions["text_llama"].attn_mask.shape == (2, 5)
    assert torch.count_nonzero(conditions["text_llama"].attn_mask[0, 3:]) == 0
    assert conditions["pooled_clip"].embeds.shape == (2, 768)

    videos = adapter.build_decoded(None, [_raw_result(seq_len=3)])
    assert videos.frames.shape == (5, 3, 8, 8)
    assert videos.cu_frames.tolist() == [0, 5]


def test_hunyuan_adapter_rejects_fused_or_missing_text_stream():
    result = _raw_result(seq_len=3)
    result.prompt_embeds = torch.randn(1, 3, 4096)

    with pytest.raises(ValueError, match=r"prompt_embeds=\[LLaMA, CLIP-pooled\]"):
        _adapter().build_condition([result])
