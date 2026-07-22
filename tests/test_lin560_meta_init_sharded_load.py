from __future__ import annotations

import torch
from accelerate import init_empty_weights
from torch import nn

from unirl.models.types.meta_init import build_meta_init_transformer, restore_init_state
from unirl.train.backend.sharded_load import _remap_hf_checkpoint_keys


def _tiny_qwen_vl():
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

    config = Qwen2_5_VLConfig(
        text_config={
            "vocab_size": 64,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "max_position_embeddings": 64,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10_000.0},
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "out_hidden_size": 16,
            "patch_size": 2,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "fullatt_block_indexes": [0],
            "window_size": 8,
        },
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=62,
        vision_end_token_id=63,
    )
    with init_empty_weights(include_buffers=False):
        return Qwen2_5_VLForConditionalGeneration(config)


def test_hf_checkpoint_renaming_uses_model_type_for_plain_and_fsdp_classes() -> None:
    model = _tiny_qwen_vl()
    model_keys = set(model.state_dict())
    language_key = next(key for key in model_keys if key.startswith("model.language_model."))
    vision_key = next(key for key in model_keys if key.startswith("model.visual."))
    old_language_key = language_key.replace("model.language_model.", "model.", 1)
    old_vision_key = vision_key.replace("model.visual.", "visual.", 1)
    stale_state = {old_language_key: object(), old_vision_key: object()}

    renamed = _remap_hf_checkpoint_keys(stale_state, model)
    assert set(renamed) == {language_key, vision_key}

    from torch.distributed.fsdp import FSDPModule

    original_cls = type(model)
    model.__class__ = type(f"FSDP{original_cls.__name__}", (FSDPModule, original_cls), {})
    renamed = _remap_hf_checkpoint_keys(stale_state, model)
    assert set(renamed) == {language_key, vision_key}


def test_meta_init_capture_survives_materialization() -> None:
    class ToyTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(2, 2, bias=False)
            self.register_buffer("rope", torch.tensor([1.25, 2.5]), persistent=False)
            self.register_buffer("persistent", torch.tensor([7.0]))
            self.plain_rope = torch.tensor([3 + 4j])

    model, captured = build_meta_init_transformer(ToyTransformer, dtype=torch.float16)
    assert model.proj.weight.is_meta
    assert set(captured["buffers"]) == {"rope"}
    assert set(captured["attrs"]) == {("", "plain_rope")}

    model.to_empty(device="cpu")
    model.rope.zero_()
    model.plain_rope.zero_()
    assert restore_init_state(model, captured) == 2
    torch.testing.assert_close(model.rope, torch.tensor([1.25, 2.5], dtype=torch.float16))
    torch.testing.assert_close(model.plain_rope, torch.tensor([3 + 4j]))
