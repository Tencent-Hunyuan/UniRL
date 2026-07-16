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
from unirl.models.bagel.rl_ops import (
    detach_replay_tree,
    move_replay_tree,
    rebuild_text_context_from_chunks,
    validate_t2ti_replay_chunk_mode,
    validate_t2ti_replay_execution_order,
)
from unirl.models.types.replay_result import ReplayResult


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


def test_detach_replay_tree_preserves_cache_aliases_without_cloning_storage():
    source = torch.tensor([2.0], requires_grad=True)
    cache = NaiveCache(1)
    cache.key_cache[0] = source * 3.0
    cache.value_cache[0] = source * 5.0
    tree = {"gen": cache, "cfg_img": cache, "packed": source * 7.0}

    detached = detach_replay_tree(tree)

    assert detached is not tree
    assert detached["gen"] is detached["cfg_img"]
    assert detached["gen"] is not cache
    assert detached["gen"].key_cache[0].data_ptr() == cache.key_cache[0].data_ptr()
    assert detached["gen"].value_cache[0].data_ptr() == cache.value_cache[0].data_ptr()
    assert not detached["gen"].key_cache[0].requires_grad
    assert not detached["gen"].value_cache[0].requires_grad
    assert not detached["packed"].requires_grad
    assert cache.key_cache[0].requires_grad


def test_move_replay_tree_preserves_aliases_dtype_and_owns_storage():
    source = torch.tensor([2.0], dtype=torch.bfloat16, requires_grad=True)
    shared = source * 3.0
    cache = NaiveCache(1)
    cache.key_cache[0] = shared
    cache.value_cache[0] = shared
    tree = {
        "gen": cache,
        "cfg_img": cache,
        "packed": shared,
        "nested": [shared, {"again": shared}],
    }

    moved = move_replay_tree(tree, torch.device("cpu"))

    assert moved is not tree
    assert moved["gen"] is moved["cfg_img"]
    assert moved["gen"] is not cache
    assert moved["gen"].key_cache[0] is moved["gen"].value_cache[0]
    assert moved["gen"].key_cache[0] is moved["packed"]
    assert moved["nested"][0] is moved["packed"]
    assert moved["nested"][1]["again"] is moved["packed"]
    assert moved["packed"].dtype == torch.bfloat16
    assert moved["packed"].device.type == "cpu"
    assert not moved["packed"].requires_grad
    assert moved["packed"].data_ptr() != shared.data_ptr()

    moved_value = moved["packed"].clone()
    with torch.no_grad():
        shared.add_(4.0)
    assert torch.equal(moved["packed"], moved_value)
    source_value = shared.clone()
    moved["packed"].add_(7.0)
    assert torch.equal(shared, source_value)


def test_replay_exposes_a_detached_copy_of_its_exact_forward_kwargs():
    source = torch.tensor([2.0], requires_grad=True)
    cache = NaiveCache(1)
    cache.key_cache[0] = source * 3.0
    cache.value_cache[0] = source * 5.0
    result = ReplayResult(log_probs=source.reshape(1, 1), prev_sample_means=source.reshape(1, 1, 1, 1))
    stage = object.__new__(BagelDiffusionStage)
    stage._replay_impl = lambda *_args, **_kwargs: (result, {"past_key_values": cache})

    replay, forward_kwargs = stage.replay_with_detached_forward_kwargs(object(), segment=object(), params=object())

    assert replay is result
    assert not forward_kwargs["past_key_values"].key_cache[0].requires_grad
    assert forward_kwargs["past_key_values"].key_cache[0].data_ptr() == cache.key_cache[0].data_ptr()
    replay.log_probs.sum().backward()
    assert torch.equal(source.grad, torch.ones_like(source))


class _FakeLMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(32, 4)
        self.cache_update = _MutatingCacheUpdate()


class _MutatingCacheUpdate(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.input_lengths = []
        self.output_lengths = []

    def forward(self, values, past_key_values, expected_length):
        self.calls += 1
        updated_cache = past_key_values.fork() if torch.is_grad_enabled() else past_key_values
        input_lengths = []
        output_lengths = []
        output = None
        for layer_idx in sorted(updated_cache.key_cache):
            key_values = torch.sin(values + layer_idx)
            value_values = torch.cos(values + layer_idx)
            previous_key = updated_cache.key_cache[layer_idx]
            previous_value = updated_cache.value_cache[layer_idx]
            actual_length = 0 if previous_key is None else int(previous_key.shape[0])
            if actual_length != int(expected_length):
                raise RuntimeError(f"cache advanced before recompute: {actual_length} != {expected_length}")
            input_lengths.append(actual_length)
            merged_key = key_values if previous_key is None else torch.cat((previous_key, key_values), dim=0)
            merged_value = value_values if previous_value is None else torch.cat((previous_value, value_values), dim=0)
            updated_cache.key_cache[layer_idx] = merged_key
            updated_cache.value_cache[layer_idx] = merged_value
            output_lengths.append(int(merged_key.shape[0]))
            output = merged_key
        self.input_lengths.append(tuple(input_lengths))
        self.output_lengths.append(tuple(output_lengths))
        return output, updated_cache


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
        self.config = SimpleNamespace(llm_config=SimpleNamespace(num_hidden_layers=2))
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


def test_exact_replay_pads_collective_depth_without_advancing_real_cache():
    model = _FakeBagel()
    reference = _FakeBagel()
    reference.load_state_dict(model.state_dict())

    context = rebuild_text_context_from_chunks(
        model,
        chunks=((11, 12), (13,), (14,)),
        expected_kv_length=4,
        expected_ropes=(4,),
        device=torch.device("cpu"),
        collective_target_chunks=5,
    )
    reference_context = rebuild_text_context_from_chunks(
        reference,
        chunks=((11, 12), (13,), (14,)),
        expected_kv_length=4,
        expected_ropes=(4,),
        device=torch.device("cpu"),
        collective_target_chunks=3,
    )

    assert model.calls == [
        ([11, 12], [0, 1]),
        ([13], [2]),
        ([14], [3]),
        ([14], [4]),
        ([14], [5]),
    ]
    assert model.language_model.model.cache_update.input_lengths[:5] == [
        (0, 0),
        (2, 2),
        (3, 3),
        (1, 1),
        (1, 1),
    ]
    assert model.language_model.model.cache_update.output_lengths[:5] == [
        (2, 2),
        (3, 3),
        (4, 4),
        (2, 2),
        (2, 2),
    ]
    assert context["kv_lens"] == [4]
    assert context["ropes"] == [4]
    assert context["collective_pad_zero"].item() == 0.0
    for store_name in ("key_cache", "value_cache"):
        cache_store = getattr(context["past_key_values"], store_name)
        reference_store = getattr(reference_context["past_key_values"], store_name)
        for layer_idx in cache_store:
            assert cache_store[layer_idx].shape[0] == 4
            torch.testing.assert_close(cache_store[layer_idx], reference_store[layer_idx])

    (context["past_key_values"].key_cache[0].sum() + context["collective_pad_zero"]).backward()
    reference_context["past_key_values"].key_cache[0].sum().backward()
    assert model.language_model.model.embed_tokens.weight.grad is not None
    torch.testing.assert_close(
        model.language_model.model.embed_tokens.weight.grad,
        reference.language_model.model.embed_tokens.weight.grad,
    )


def test_exact_replay_pads_no_grad_collective_depth_without_zero_graph():
    model = _FakeBagel()

    with torch.no_grad():
        context = rebuild_text_context_from_chunks(
            model,
            chunks=((11, 12), (13,), (14,)),
            expected_kv_length=4,
            expected_ropes=(4,),
            device=torch.device("cpu"),
            collective_target_chunks=5,
        )

    assert model.calls == [
        ([11, 12], [0, 1]),
        ([13], [2]),
        ([14], [3]),
        ([14], [4]),
        ([14], [5]),
    ]
    assert model.language_model.model.cache_update.input_lengths == [
        (0, 0),
        (2, 2),
        (3, 3),
        (1, 1),
        (1, 1),
    ]
    assert model.language_model.model.cache_update.output_lengths == [
        (2, 2),
        (3, 3),
        (4, 4),
        (2, 2),
        (2, 2),
    ]
    assert context["kv_lens"] == [4]
    assert context["ropes"] == [4]
    for store in (context["past_key_values"].key_cache, context["past_key_values"].value_cache):
        assert all(value.shape[0] == 4 for value in store.values())
    assert "collective_pad_zero" not in context


def test_exact_replay_rejects_collective_target_shorter_than_trace():
    with pytest.raises(ValueError, match="collective target cannot be shorter"):
        rebuild_text_context_from_chunks(
            _FakeBagel(),
            chunks=((11, 12), (13,), (14,)),
            expected_kv_length=4,
            expected_ropes=(4,),
            device=torch.device("cpu"),
            collective_target_chunks=2,
        )


def test_collapsed_replay_matches_exact_cache_and_gradients_with_prefill_boundary():
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
    assert collapsed.calls == [([11, 12], [0, 1]), ([13, 14], [2, 3])]
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
    assert model.calls == [([11, 12], [0, 1]), ([13, 14], [2, 3])]
    assert gen["kv_lens"] == [4]
    assert cfg_text["kv_lens"] == [0]
    assert cfg_img is gen
    assert image_shape == (512, 384)


def test_replay_chunk_mode_normalizes_explicit_opt_in():
    assert validate_t2ti_replay_chunk_mode(" COLLAPSED ") == "collapsed"


def test_replay_execution_order_normalizes_layer_major():
    assert validate_t2ti_replay_execution_order(" LAYER_MAJOR ") == "layer_major"


@pytest.mark.parametrize("mode", ["unknown", "", None])
def test_replay_chunk_mode_rejects_unknown_values(mode):
    with pytest.raises(ValueError, match="must be one of"):
        validate_t2ti_replay_chunk_mode(mode)


@pytest.mark.parametrize("order", ["unknown", "", None])
def test_replay_execution_order_rejects_unknown_values(order):
    with pytest.raises(ValueError, match="must be one of"):
        validate_t2ti_replay_execution_order(order)


def test_collapsed_replay_rejects_layer_major_execution():
    with pytest.raises(ValueError, match="collapsed replay only supports chunk_major"):
        rebuild_text_context_from_chunks(
            _FakeBagel(),
            chunks=((11, 12), (13,)),
            expected_kv_length=3,
            expected_ropes=(3,),
            device=torch.device("cpu"),
            chunk_mode="collapsed",
            execution_order="layer_major",
        )


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
