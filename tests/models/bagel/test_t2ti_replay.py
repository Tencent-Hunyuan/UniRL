from types import SimpleNamespace

import pytest
import torch
from torch import nn

from unirl.models.bagel.conditions import (
    BAGEL_T2TI_REPLAY_CUSTOM_OUTPUT,
    BagelT2TIDiffusionConditions,
    BagelThinkKVReplaySpec,
)
from unirl.models.bagel.diffusion import BagelDiffusionStage
from unirl.models.bagel.rl_ops import rebuild_text_context_from_chunks, validate_t2ti_replay_chunk_mode


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
        updated_cache = past_key_values.fork() if torch.is_grad_enabled() else past_key_values
        values = torch.sin(values)
        previous = updated_cache.key_cache[0]
        actual_length = 0 if previous is None else int(previous.shape[0])
        if actual_length != int(expected_length):
            raise RuntimeError(f"cache advanced before recompute: {actual_length} != {expected_length}")
        merged = values if previous is None else torch.cat((previous, values), dim=0)
        updated_cache.key_cache[0] = merged
        updated_cache.value_cache[0] = merged
        return merged, updated_cache


def test_cache_update_preserves_no_grad_in_place_contract():
    update = _MutatingCacheUpdate()
    cache = NaiveCache(1)

    with torch.no_grad():
        _, updated = update(torch.ones(1, 4), cache, 0)

    assert updated is cache
    assert cache.key_cache[0] is not None


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


def test_collapsed_replay_matches_exact_cache_and_gradients_with_one_prefill():
    exact = _FakeBagel()
    collapsed = _FakeBagel()
    collapsed.load_state_dict(exact.state_dict())

    exact_context = rebuild_text_context_from_chunks(
        exact,
        chunks=((11, 12), (13,), (14,)),
        expected_kv_length=4,
        expected_ropes=(4,),
        device=torch.device("cpu"),
    )
    collapsed_context = rebuild_text_context_from_chunks(
        collapsed,
        chunks=((11, 12), (13,), (14,)),
        expected_kv_length=4,
        expected_ropes=(4,),
        device=torch.device("cpu"),
        chunk_mode="collapsed",
    )

    assert exact.calls == [([11, 12], [0, 1]), ([13], [2]), ([14], [3])]
    assert collapsed.calls == [([11, 12, 13, 14], [0, 1, 2, 3])]
    exact_cache = exact_context["past_key_values"].key_cache[0]
    collapsed_cache = collapsed_context["past_key_values"].key_cache[0]
    torch.testing.assert_close(collapsed_cache, exact_cache)
    assert exact_context["kv_lens"] == collapsed_context["kv_lens"] == [4]
    assert exact_context["ropes"] == collapsed_context["ropes"] == [4]

    exact_cache.square().sum().backward()
    collapsed_cache.square().sum().backward()
    torch.testing.assert_close(
        collapsed.language_model.model.embed_tokens.weight.grad,
        exact.language_model.model.embed_tokens.weight.grad,
    )


def test_diffusion_stage_applies_collapsed_replay_mode():
    model = _FakeBagel()
    stage = BagelDiffusionStage(
        model=SimpleNamespace(model=model, device="cpu"),
        t2ti_replay_chunk_mode="collapsed",
    )
    conditions = BagelT2TIDiffusionConditions.for_sample(BagelThinkKVReplaySpec.from_custom_output(_payload()))

    gen, cfg_text, cfg_img, image_shape = stage._build_contexts_from_replay(conditions)

    assert stage.t2ti_replay_chunk_mode == "collapsed"
    assert model.calls == [([11, 12, 13, 14], [0, 1, 2, 3])]
    assert gen["kv_lens"] == [4]
    assert cfg_text["kv_lens"] == [0]
    assert cfg_img is gen
    assert image_shape == (512, 384)


def test_replay_chunk_mode_normalizes_explicit_opt_in():
    assert validate_t2ti_replay_chunk_mode(" COLLAPSED ") == "collapsed"


@pytest.mark.parametrize("mode", ["unknown", "", None])
def test_replay_chunk_mode_rejects_unknown_values(mode):
    with pytest.raises(ValueError, match="must be one of"):
        validate_t2ti_replay_chunk_mode(mode)


@pytest.mark.parametrize("chunk_mode", ["exact", "collapsed"])
@pytest.mark.parametrize(
    ("expected_kv_length", "expected_ropes", "message"),
    [
        (3, (4,), "KV length"),
        (4, (3,), "ropes"),
    ],
)
def test_replay_modes_preserve_geometry_validation(chunk_mode, expected_kv_length, expected_ropes, message):
    with pytest.raises(ValueError, match=message):
        rebuild_text_context_from_chunks(
            _FakeBagel(),
            chunks=((11, 12), (13,), (14,)),
            expected_kv_length=expected_kv_length,
            expected_ropes=expected_ropes,
            device=torch.device("cpu"),
            chunk_mode=chunk_mode,
        )


@pytest.mark.parametrize("chunk_mode", ["exact", "collapsed"])
def test_replay_modes_reject_empty_captured_chunks(chunk_mode):
    with pytest.raises(ValueError, match="chunk 1 is empty"):
        rebuild_text_context_from_chunks(
            _FakeBagel(),
            chunks=((11, 12), (), (14,)),
            expected_kv_length=3,
            expected_ropes=(3,),
            device=torch.device("cpu"),
            chunk_mode=chunk_mode,
        )
