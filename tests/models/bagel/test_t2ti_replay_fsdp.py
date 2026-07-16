from __future__ import annotations

import datetime
import multiprocessing as mp
import queue
import tempfile
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest
import torch
from torch import nn

from unirl.models.bagel.rl_ops import rebuild_text_context_from_chunks


class NaiveCache:
    """Minimal BAGEL cache exposed from the model module for ``rl_ops``."""

    def __init__(self, num_layers: int) -> None:
        self.key_cache = {index: None for index in range(num_layers)}
        self.value_cache = {index: None for index in range(num_layers)}

    def fork(self) -> NaiveCache:
        cache = type(self)(len(self.key_cache))
        cache.key_cache = self.key_cache.copy()
        cache.value_cache = self.value_cache.copy()
        return cache


def _merge_cache(past: Optional[torch.Tensor], current: torch.Tensor) -> torch.Tensor:
    if past is None:
        return current

    # Match BAGEL's cached branch: allocate a fresh merged buffer and write the
    # old cache and fresh query projections into it. Its CopySlices graph is the
    # topology that the old no-cache padding failed to reproduce.
    merged = current.new_zeros((past.shape[0] + current.shape[0], current.shape[1]))
    merged[: past.shape[0]] = past
    merged[past.shape[0] :] = current
    return merged


class _ReplayCacheBlock(nn.Module):
    def __init__(self, width: int, layer_index: int) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.norm = nn.LayerNorm(width)
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.SiLU(),
            nn.Linear(width * 2, width),
        )
        self.past_lengths: list[int] = []

    def forward(
        self,
        hidden: torch.Tensor,
        past_key: Optional[torch.Tensor],
        past_value: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.past_lengths.append(0 if past_key is None else int(past_key.shape[0]))

        residual = hidden
        hidden = self.norm(hidden)
        query = self.query(hidden)
        merged_key = _merge_cache(past_key, self.key(hidden))
        merged_value = _merge_cache(past_value, self.value(hidden))

        attended = query + merged_key.mean(dim=0, keepdim=True) + merged_value.mean(dim=0, keepdim=True)
        hidden = residual + self.output(torch.tanh(attended))
        hidden = hidden + self.mlp(self.norm(hidden))
        return hidden, merged_key, merged_value


class _ReplayLMModel(nn.Module):
    def __init__(self, *, width: int = 4, num_layers: int = 3) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(64, width)
        self.embed_tokens.weight.requires_grad_(False)
        self.layers = nn.ModuleList(_ReplayCacheBlock(width, layer_index) for layer_index in range(num_layers))


class _ReplayLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _ReplayLMModel()

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_ids: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values: bool = True,
        is_causal: bool = True,
        mode: str = "und",
    ) -> SimpleNamespace:
        del query_lens, packed_query_position_ids, packed_query_indexes
        del key_values_lens, packed_key_value_indexes, is_causal, mode

        if update_past_key_values:
            assert past_key_values is not None
            updated_cache = past_key_values.fork()
        else:
            updated_cache = past_key_values

        hidden = packed_query_sequence
        for layer_index, layer in enumerate(self.model.layers):
            past_key = None if past_key_values is None else past_key_values.key_cache[layer_index]
            past_value = None if past_key_values is None else past_key_values.value_cache[layer_index]
            hidden, merged_key, merged_value = layer(hidden, past_key, past_value)
            if update_past_key_values:
                assert updated_cache is not None
                updated_cache.key_cache[layer_index] = merged_key
                updated_cache.value_cache[layer_index] = merged_value

        return SimpleNamespace(
            packed_query_sequence=hidden,
            past_key_values=updated_cache,
        )


class _ReplayBagel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = _ReplayLanguageModel()
        self.config = SimpleNamespace(llm_config=SimpleNamespace(num_hidden_layers=3))
        self.cache_input_calls: list[tuple[int, ...]] = []

    @torch.no_grad()
    def forward_cache_update_text(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.Tensor,
        packed_text_position_ids: torch.Tensor,
        text_token_lens: torch.Tensor,
        packed_text_indexes: torch.Tensor,
        packed_key_value_indexes: torch.Tensor,
        key_values_lens: torch.Tensor,
    ) -> NaiveCache:
        self.cache_input_calls.append(tuple(int(token) for token in packed_text_ids.tolist()))
        hidden = self.language_model.model.embed_tokens(packed_text_ids)
        hidden = hidden + packed_text_position_ids.to(dtype=hidden.dtype).unsqueeze(-1) * 0.01
        output = self.language_model.forward_inference(
            packed_query_sequence=hidden,
            query_lens=text_token_lens,
            packed_query_position_ids=packed_text_position_ids,
            packed_query_indexes=packed_text_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=True,
            is_causal=True,
            mode="und",
        )
        assert output.past_key_values is not None
        return output.past_key_values


def _rank_chunks(rank: int) -> tuple[tuple[int, ...], ...]:
    if rank == 0:
        return ((1, 2), (3,))
    return ((7, 8), (9,), (10,), (11,), (12,))


def _rebuild_context(model: _ReplayBagel, *, rank: int, target_chunks: int) -> dict[str, Any]:
    chunks = _rank_chunks(rank)
    token_count = sum(len(chunk) for chunk in chunks)
    return rebuild_text_context_from_chunks(
        model,
        chunks=chunks,
        expected_kv_length=token_count,
        expected_ropes=(token_count,),
        device=torch.device("cpu"),
        collective_target_chunks=target_chunks,
    )


def _semantic_image_loss(model: _ReplayBagel, context: dict[str, Any], *, rank: int) -> torch.Tensor:
    width = model.language_model.model.embed_tokens.embedding_dim
    image = torch.linspace(0.1, 0.4, width).reshape(1, width) + rank * 0.05
    query_lens = torch.ones(1, dtype=torch.int)
    real_cache_length = int(context["kv_lens"][0])
    output = model.language_model.forward_inference(
        packed_query_sequence=image,
        query_lens=query_lens,
        packed_query_position_ids=torch.zeros(1, dtype=torch.long),
        packed_query_indexes=torch.tensor([real_cache_length], dtype=torch.long),
        past_key_values=context["past_key_values"],
        key_values_lens=torch.tensor([real_cache_length], dtype=torch.int),
        packed_key_value_indexes=torch.arange(real_cache_length, dtype=torch.long),
        update_past_key_values=False,
        is_causal=True,
        mode="und",
    )
    padding_zero = context.get("collective_pad_zero", image.new_zeros(()))
    return output.packed_query_sequence.float().square().mean() + padding_zero


def _layer_parameters(model: _ReplayBagel):
    return model.language_model.model.layers.parameters()


def _run_fsdp_replay_worker(rank: int, store_path: str, result_queue: Any) -> None:
    import torch.distributed as dist

    try:
        from torch.distributed._composable import checkpoint
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard

        dist.init_process_group(
            "gloo",
            init_method=f"file://{store_path}",
            rank=rank,
            world_size=2,
            timeout=datetime.timedelta(seconds=15),
        )
        mesh = init_device_mesh("cpu", (2,))

        torch.manual_seed(1234)
        reference = _ReplayBagel()
        model = _ReplayBagel()
        model.load_state_dict(reference.state_dict())

        # Padding is zero-valued, so the sharded result must equal the averaged
        # gradients of independent, rank-local, unpadded semantic traces.
        reference_context = _rebuild_context(
            reference,
            rank=rank,
            target_chunks=len(_rank_chunks(rank)),
        )
        _semantic_image_loss(reference, reference_context, rank=rank).backward()
        expected_gradients = []
        for parameter in _layer_parameters(reference):
            assert parameter.grad is not None
            gradient = parameter.grad.detach().clone()
            dist.all_reduce(gradient)
            expected_gradients.append((gradient / 2).reshape(-1))

        for layer in model.language_model.model.layers:
            checkpoint(layer)
        for layer in model.language_model.model.layers:
            fully_shard(layer, mesh=mesh, reshard_after_forward=True)

        target_chunks = 5
        context = _rebuild_context(model, rank=rank, target_chunks=target_chunks)
        real_cache_length = sum(len(chunk) for chunk in _rank_chunks(rank))
        expected_replay_history = (0, 2, 1, 1, 1) if rank == 0 else (0, 2, 3, 4, 5)
        expected_history = expected_replay_history + (real_cache_length,)

        loss = _semantic_image_loss(model, context, rank=rank)
        forward_histories = tuple(tuple(layer.past_lengths) for layer in model.language_model.model.layers)
        real_cache_lengths = tuple(
            int(context["past_key_values"].key_cache[index].shape[0])
            for index in range(len(model.language_model.model.layers))
        )
        real_calls = tuple(tuple(chunk) for chunk in _rank_chunks(rank))
        padding_calls = tuple(model.cache_input_calls[len(real_calls) :])
        topology_ok = all(history == expected_history for history in forward_histories)
        real_inputs_ok = tuple(model.cache_input_calls[: len(real_calls)]) == real_calls
        bounded_padding_ok = (
            not padding_calls
            if rank == 1
            else len(padding_calls) == target_chunks - len(real_calls)
            and all(len(call) == 1 for call in padding_calls)
            and expected_replay_history[len(real_calls) :] == (1, 1, 1)
        )
        real_cache_untouched = real_cache_lengths == (real_cache_length,) * len(real_cache_lengths)

        loss.backward()

        actual_gradients = []
        for parameter in _layer_parameters(model):
            assert parameter.grad is not None
            actual_gradients.append(parameter.grad.full_tensor().detach().reshape(-1))

        actual = torch.cat(actual_gradients)
        expected = torch.cat(expected_gradients)
        dist.barrier()
        result_queue.put(
            (
                rank,
                "ok",
                float((actual - expected).abs().max().item()),
                float(actual.norm().item()),
                bool(torch.isfinite(actual).all().item()),
                topology_ok,
                real_inputs_ok,
                bounded_padding_ok,
                real_cache_untouched,
                forward_histories,
                padding_calls,
            )
        )
    except Exception as error:
        result_queue.put((rank, "error", type(error).__name__, str(error), traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_cache_faithful_padding_preserves_fsdp2_collectives_and_gradients() -> None:
    dist = torch.distributed
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("requires torch.distributed with Gloo")
    try:
        from torch.distributed._composable import checkpoint as _checkpoint  # noqa: F401
        from torch.distributed.fsdp import fully_shard as _fully_shard  # noqa: F401
    except ImportError:
        pytest.skip("requires composable checkpointing and FSDP2")

    context = mp.get_context("spawn")
    result_queue = context.Queue()
    with tempfile.TemporaryDirectory() as temp_dir:
        store_path = str(Path(temp_dir) / "store")
        processes = [
            context.Process(target=_run_fsdp_replay_worker, args=(rank, store_path, result_queue)) for rank in range(2)
        ]
        for process in processes:
            process.start()

        deadline = time.monotonic() + 30
        for process in processes:
            process.join(max(0.0, deadline - time.monotonic()))

        hanging = [process for process in processes if process.is_alive()]
        for process in hanging:
            process.kill()
        for process in hanging:
            process.join(5)
        if hanging:
            pytest.fail("two-rank cache-faithful FSDP2 replay exceeded the 30-second deadline")

        results = []
        for _ in processes:
            try:
                results.append(result_queue.get(timeout=2))
            except queue.Empty:
                break

    assert len(results) == 2, f"missing worker result; exit codes={[p.exitcode for p in processes]}"
    errors = [result for result in results if result[1] != "ok"]
    assert not errors, "\n".join(str(error) for error in errors)

    results.sort()
    for result in results:
        (
            _rank,
            _status,
            max_error,
            gradient_norm,
            finite,
            topology_ok,
            real_inputs_ok,
            bounded_padding_ok,
            real_cache_untouched,
            forward_histories,
            padding_calls,
        ) = result
        assert finite
        assert gradient_norm > 0.0
        assert max_error <= 1e-6
        assert topology_ok, forward_histories
        assert real_inputs_ok
        assert bounded_padding_ok, padding_calls
        assert real_cache_untouched
    assert results[0][3] == pytest.approx(results[1][3], rel=1e-6, abs=1e-7)
