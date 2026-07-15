from types import SimpleNamespace

import pytest
import torch
from torch import nn

from unirl.models.bagel.conditions import (
    BAGEL_T2TI_REPLAY_CUSTOM_OUTPUT,
    BagelT2TIDiffusionConditions,
    BagelThinkKVReplaySpec,
)
from unirl.models.bagel.rl_ops import rebuild_text_context_from_chunks


def _payload(**overrides):
    value = {
        "cache_input_ids": [11, 12, 13, 14],
        "chunk_offsets": [0, 2, 3, 4],
        "kv_length": 4,
        "ropes": [4],
        "received_kv_length": 4,
        "received_ropes": [4],
        "image_shape": [512, 384],
    }
    value.update(overrides)
    return {BAGEL_T2TI_REPLAY_CUSTOM_OUTPUT: value}


def test_replay_spec_parses_wire_payload_and_preserves_chunks():
    spec = BagelThinkKVReplaySpec.from_custom_output(_payload())

    assert spec.chunks() == ((11, 12), (13,), (14,))
    assert spec.image_shape == (512, 384)

    conditions = BagelT2TIDiffusionConditions.for_sample(spec)
    restored = BagelT2TIDiffusionConditions.from_dict(conditions.to_dict())
    assert restored.single_spec() is spec


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"received_kv_length": 3}, "received KV length"),
        ({"received_ropes": [3]}, "received ropes"),
        ({"chunk_offsets": [0, 4, 3]}, "strictly increasing"),
    ],
)
def test_replay_spec_rejects_inconsistent_transfer_metadata(overrides, message):
    with pytest.raises(ValueError, match=message):
        BagelThinkKVReplaySpec.from_custom_output(_payload(**overrides))


class NaiveCache:
    def __init__(self, num_layers):
        self.key_cache = {index: None for index in range(num_layers)}
        self.value_cache = {index: None for index in range(num_layers)}

    def fork(self):
        cache = type(self)(len(self.key_cache))
        cache.key_cache = self.key_cache.copy()
        cache.value_cache = self.value_cache.copy()
        return cache


class _FakeLMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(32, 4)
        self.cache_update = _MutatingCacheUpdate()


class _MutatingCacheUpdate(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, values, past_key_values, expected_length):
        self.calls += 1
        updated_cache = past_key_values.fork()
        values = torch.sin(values)
        previous = updated_cache.key_cache[0]
        actual_length = 0 if previous is None else int(previous.shape[0])
        if actual_length != int(expected_length):
            raise RuntimeError(f"cache advanced before recompute: {actual_length} != {expected_length}")
        merged = values if previous is None else torch.cat((previous, values), dim=0)
        updated_cache.key_cache[0] = merged
        updated_cache.value_cache[0] = merged
        return merged, updated_cache


class _FakeLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _FakeLMModel()


class _FakeBagel(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = _FakeLanguageModel()
        self.config = SimpleNamespace(llm_config=SimpleNamespace(num_hidden_layers=1))
        self.calls = []

    @torch.no_grad()
    def forward_cache_update_text(
        self,
        past_key_values,
        packed_text_ids,
        packed_text_position_ids,
        text_token_lens,
        packed_text_indexes,
        packed_key_value_indexes,
        key_values_lens,
    ):
        del text_token_lens, packed_text_indexes, packed_key_value_indexes
        self.calls.append(
            (
                packed_text_ids.detach().cpu().tolist(),
                packed_text_position_ids.detach().cpu().tolist(),
            )
        )
        values = self.language_model.model.embed_tokens(packed_text_ids)
        _, past_key_values = self.language_model.model.cache_update(
            values,
            past_key_values,
            int(key_values_lens[0]),
        )
        return past_key_values


def test_rebuild_text_context_replays_exact_chunks_with_gradients():
    from torch.distributed._composable import checkpoint

    model = _FakeBagel()
    model.language_model.train()
    checkpoint(model.language_model.model.cache_update)

    context = rebuild_text_context_from_chunks(
        model,
        chunks=((11, 12), (13,), (14,)),
        expected_kv_length=4,
        expected_ropes=(4,),
        device=torch.device("cpu"),
    )

    assert model.calls == [([11, 12], [0, 1]), ([13], [2]), ([14], [3])]
    assert context["kv_lens"] == [4]
    assert context["ropes"] == [4]
    assert context["past_key_values"].key_cache[0].requires_grad
    context["past_key_values"].key_cache[0].sum().backward()
    assert model.language_model.model.embed_tokens.weight.grad is not None
    assert model.language_model.model.cache_update.calls == 6
    assert checkpoint.state(model.language_model.model.cache_update).enable_hook
    # Grad-enabled inference replay intentionally remains in eval dispatch
    # through the later checkpoint recomputation/backward.
    assert not model.language_model.training
