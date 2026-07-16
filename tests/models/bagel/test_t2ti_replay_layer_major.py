from __future__ import annotations

import datetime
import multiprocessing as mp
import os
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

from unirl.models.bagel.rl_ops import (
    install_layer_major_replay_dispatch,
    rebuild_text_context_from_chunks,
)


class NaiveCache:
    """Minimal BAGEL cache discoverable by ``init_und_context``."""

    def __init__(self, num_layers: int) -> None:
        self.key_cache: dict[int, Optional[torch.Tensor]] = {index: None for index in range(num_layers)}
        self.value_cache: dict[int, Optional[torch.Tensor]] = {index: None for index in range(num_layers)}

    def fork(self) -> NaiveCache:
        cache = type(self)(len(self.key_cache))
        cache.key_cache = self.key_cache.copy()
        cache.value_cache = self.value_cache.copy()
        return cache


def _merge_cache(past: Optional[torch.Tensor], current: torch.Tensor) -> torch.Tensor:
    if past is None:
        return current
    merged = current.new_zeros((past.shape[0] + current.shape[0], current.shape[1]))
    merged[: past.shape[0]] = past
    merged[past.shape[0] :] = current
    return merged


def _record_outer_call(module: nn.Module, _args: Any, _kwargs: Any) -> None:
    module.outer_calls += 1


class _ReplayCacheBlock(nn.Module):
    """Cached decoder block with BAGEL's inference-call contract."""

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
        self.outer_calls = 0
        self.forward_inference_calls = 0
        self.past_lengths: list[int] = []
        self.register_forward_pre_hook(_record_outer_call, with_kwargs=True)

    def forward(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, NaiveCache]:
        return self.forward_inference(*args, **kwargs)

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        packed_query_indexes: torch.Tensor,
        past_key_values: NaiveCache,
        key_values_lens: torch.Tensor,
        packed_key_value_indexes: torch.Tensor,
        update_past_key_values: bool = True,
        is_causal: bool = True,
        mode: str = "und",
    ) -> tuple[torch.Tensor, NaiveCache]:
        assert is_causal
        assert mode == "und"
        self.forward_inference_calls += 1

        past_key = past_key_values.key_cache[self.layer_index]
        past_value = past_key_values.value_cache[self.layer_index]
        past_length = 0 if past_key is None else int(past_key.shape[0])
        query_length = int(packed_query_sequence.shape[0])
        self.past_lengths.append(past_length)

        assert query_lens.tolist() == [query_length]
        assert key_values_lens.tolist() == [past_length]
        assert packed_key_value_indexes.tolist() == list(range(past_length))
        assert packed_query_indexes.tolist() == list(range(past_length, past_length + query_length))

        cos, sin = packed_query_position_embeddings
        residual = packed_query_sequence
        hidden = self.norm(packed_query_sequence + (cos + sin) * 0.01)
        query = self.query(hidden)
        merged_key = _merge_cache(past_key, self.key(hidden))
        merged_value = _merge_cache(past_value, self.value(hidden))

        # Preserve causal within-chunk geometry instead of reducing every query
        # against the completed current chunk.
        contexts = []
        for query_index in range(query_length):
            visible = past_length + query_index + 1
            contexts.append(merged_key[:visible].mean(dim=0) + merged_value[:visible].mean(dim=0))
        attended = query + torch.stack(contexts, dim=0)
        hidden = residual + self.output(torch.tanh(attended))
        hidden = hidden + self.mlp(self.norm(hidden))

        if not update_past_key_values:
            return hidden, past_key_values
        # Match BAGEL's checkpoint-safe grad path: never mutate a cache object
        # captured as an activation-checkpoint input.
        updated_cache = past_key_values.fork() if torch.is_grad_enabled() else past_key_values
        updated_cache.key_cache[self.layer_index] = merged_key
        updated_cache.value_cache[self.layer_index] = merged_value
        return hidden, updated_cache


class _ReplayRotaryEmbedding(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.register_buffer("frequencies", torch.linspace(0.05, 0.4, width), persistent=False)

    def forward(
        self,
        hidden: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        phase = position_ids.to(dtype=torch.float32).unsqueeze(-1) * self.frequencies
        return phase.cos().to(dtype=hidden.dtype), phase.sin().to(dtype=hidden.dtype)


class _ReplayLMModel(nn.Module):
    def __init__(self, *, width: int = 8, num_layers: int = 3) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(128, width)
        self.embed_tokens.weight.requires_grad_(False)
        self.rotary_emb = _ReplayRotaryEmbedding(width)
        self.layers = nn.ModuleList(_ReplayCacheBlock(width, index) for index in range(num_layers))
        self.enable_taylorseer = False

    def forward(self, *args: Any, **kwargs: Any) -> SimpleNamespace:
        return self.forward_inference(*args, **kwargs)

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_ids: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: NaiveCache,
        key_values_lens: torch.Tensor,
        packed_key_value_indexes: torch.Tensor,
        update_past_key_values: bool = True,
        is_causal: bool = True,
        mode: str = "und",
    ) -> SimpleNamespace:
        cos, sin = self.rotary_emb(
            packed_query_sequence,
            packed_query_position_ids.unsqueeze(0),
        )
        position_embeddings = (cos.squeeze(0), sin.squeeze(0))
        hidden = packed_query_sequence
        cache = past_key_values
        for layer in self.layers:
            hidden, cache = layer(
                packed_query_sequence=hidden,
                query_lens=query_lens,
                packed_query_position_embeddings=position_embeddings,
                packed_query_indexes=packed_query_indexes,
                past_key_values=cache,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=update_past_key_values,
                is_causal=is_causal,
                mode=mode,
            )
        return SimpleNamespace(packed_query_sequence=hidden, past_key_values=cache)


class _ReplayLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _ReplayLMModel()

    def forward_inference(self, **kwargs: Any) -> SimpleNamespace:
        return self.model(**kwargs)


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
        return output.past_key_values


def _rebuild(
    model: _ReplayBagel,
    chunks: tuple[tuple[int, ...], ...],
    *,
    execution_order: str,
) -> dict[str, Any]:
    token_count = sum(len(chunk) for chunk in chunks)
    device = next(model.parameters()).device
    return rebuild_text_context_from_chunks(
        model,
        chunks=chunks,
        expected_kv_length=token_count,
        expected_ropes=(token_count,),
        device=device,
        execution_order=execution_order,
        collective_target_chunks=len(chunks) if execution_order == "chunk_major" else None,
    )


def _semantic_image_output(
    model: _ReplayBagel,
    context: dict[str, Any],
    *,
    rank: int = 0,
) -> torch.Tensor:
    width = model.language_model.model.embed_tokens.embedding_dim
    kv_length = int(context["kv_lens"][0])
    device = next(model.parameters()).device
    dtype = model.language_model.model.embed_tokens.weight.dtype
    hidden = torch.linspace(-0.4, 0.5, width, device=device, dtype=dtype).reshape(1, width) + rank * 0.03
    output = model.language_model.forward_inference(
        packed_query_sequence=hidden,
        query_lens=torch.ones(1, dtype=torch.int, device=device),
        packed_query_position_ids=torch.tensor([kv_length], dtype=torch.long, device=device),
        packed_query_indexes=torch.tensor([kv_length], dtype=torch.long, device=device),
        past_key_values=context["past_key_values"],
        key_values_lens=torch.tensor([kv_length], dtype=torch.int, device=device),
        packed_key_value_indexes=torch.arange(kv_length, dtype=torch.long, device=device),
        update_past_key_values=False,
        is_causal=True,
        mode="und",
    )
    return output.packed_query_sequence


def _layer_parameters(model: _ReplayBagel):
    return model.language_model.model.layers.parameters()


def _assert_cache_equal(left: NaiveCache, right: NaiveCache) -> None:
    for store_name in ("key_cache", "value_cache"):
        left_store = getattr(left, store_name)
        right_store = getattr(right, store_name)
        assert left_store.keys() == right_store.keys()
        for layer_index in left_store:
            torch.testing.assert_close(
                left_store[layer_index],
                right_store[layer_index],
                rtol=0,
                atol=0,
            )


def test_layer_major_matches_chunk_major_cache_loss_and_gradients() -> None:
    torch.manual_seed(1234)
    chunk_major = _ReplayBagel().eval()
    layer_major = _ReplayBagel().eval()
    layer_major.load_state_dict(chunk_major.state_dict())
    install_layer_major_replay_dispatch(layer_major)
    chunks = ((1, 2, 3), (4,), (5, 6), (7,), (8, 9, 10, 11))

    chunk_context = _rebuild(chunk_major, chunks, execution_order="chunk_major")
    layer_context = _rebuild(layer_major, chunks, execution_order="layer_major")

    assert chunk_major.cache_input_calls == list(chunks)
    assert layer_major.cache_input_calls == []
    assert "collective_pad_zero" not in layer_context
    _assert_cache_equal(chunk_context["past_key_values"], layer_context["past_key_values"])

    chunk_output = _semantic_image_output(chunk_major, chunk_context)
    layer_output = _semantic_image_output(layer_major, layer_context)
    torch.testing.assert_close(chunk_output, layer_output, rtol=0, atol=0)
    chunk_loss = chunk_output.float().square().mean()
    layer_loss = layer_output.float().square().mean()
    torch.testing.assert_close(chunk_loss, layer_loss, rtol=0, atol=0)

    chunk_loss.backward()
    layer_loss.backward()
    for chunk_parameter, layer_parameter in zip(
        _layer_parameters(chunk_major),
        _layer_parameters(layer_major),
    ):
        assert chunk_parameter.grad is not None
        assert layer_parameter.grad is not None
        torch.testing.assert_close(chunk_parameter.grad, layer_parameter.grad, rtol=0, atol=0)

    assert [layer.outer_calls for layer in chunk_major.language_model.model.layers] == [len(chunks) + 1] * 3
    assert [layer.outer_calls for layer in layer_major.language_model.model.layers] == [2, 2, 2]
    assert [layer.forward_inference_calls for layer in layer_major.language_model.model.layers] == [len(chunks) + 1] * 3


def test_layer_major_dispatch_survives_accelerate_hook_removal() -> None:
    hooks = pytest.importorskip("accelerate.hooks")
    from unirl.train.backend.fsdp.backend import _remove_accelerate_hooks_before_fsdp

    model = _ReplayBagel().eval()
    state_dict_keys = tuple(model.state_dict())
    install_layer_major_replay_dispatch(model)
    layers = tuple(model.language_model.model.layers)

    # Mirror load_checkpoint_and_dispatch(force_hooks=True): Accelerate saves
    # the installed dispatch as each module's _old_forward before wrapping it.
    for module in model.modules():
        hooks.add_hook_to_module(
            module,
            hooks.AlignDevicesHook(execution_device=torch.device("cpu")),
        )
    assert all(hasattr(layer, "_hf_hook") for layer in layers)
    assert all(
        getattr(getattr(layer, "_old_forward"), "__func__", None).__name__ == "_layer_major_replay_forward_dispatch"
        for layer in layers
    )

    removed = _remove_accelerate_hooks_before_fsdp(model.language_model)

    assert removed > 0
    assert all(not hasattr(layer, "_hf_hook") for layer in layers)
    assert all(not hasattr(layer, "_old_forward") for layer in layers)
    assert all(
        getattr(layer.forward, "__func__", None).__name__ == "_layer_major_replay_forward_dispatch" for layer in layers
    )

    chunks = ((1, 2), (3,), (4, 5))
    context = _rebuild(model, chunks, execution_order="layer_major")
    output = _semantic_image_output(model, context)

    assert output.shape == (1, 8)
    assert [layer.outer_calls for layer in layers] == [2, 2, 2]
    assert [layer.forward_inference_calls for layer in layers] == [len(chunks) + 1] * 3
    assert tuple(model.state_dict()) == state_dict_keys


def _rank_chunks(rank: int) -> tuple[tuple[int, ...], ...]:
    if rank == 0:
        return ((1, 2), (3,))
    return ((7, 8), (9,), (10,), (11,), (12, 13))


def _run_layer_major_fsdp_worker(rank: int, store_path: str, result_queue: Any, device_type: str) -> None:
    import torch.distributed as dist

    try:
        from torch.distributed._composable import checkpoint
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        backend = "nccl" if device_type == "cuda" else "gloo"
        device = torch.device("cuda", rank) if device_type == "cuda" else torch.device("cpu")
        if device_type == "cuda":
            torch.cuda.set_device(device)
            torch.backends.cuda.matmul.allow_tf32 = False

        dist.init_process_group(
            backend,
            init_method=f"file://{store_path}",
            rank=rank,
            world_size=2,
            timeout=datetime.timedelta(seconds=60 if device_type == "cuda" else 20),
        )
        mesh = init_device_mesh(device_type, (2,))

        torch.manual_seed(4321)
        compute_dtype = torch.bfloat16 if device_type == "cuda" else torch.float32
        reference = _ReplayBagel().eval().to(device=device, dtype=compute_dtype)
        sharded = _ReplayBagel().eval().to(device=device, dtype=compute_dtype)
        sharded.load_state_dict(reference.state_dict())
        if device_type == "cuda":
            # Match the production BAGEL policy: bf16-loaded values, fp32
            # sharded masters, and bf16 all-gathered compute parameters.
            for parameter in _layer_parameters(sharded):
                parameter.data = parameter.data.float()
        install_layer_major_replay_dispatch(reference)
        install_layer_major_replay_dispatch(sharded)
        chunks = _rank_chunks(rank)

        reference_context = _rebuild(reference, chunks, execution_order="layer_major")
        reference_output = _semantic_image_output(reference, reference_context, rank=rank)
        reference_output.float().square().mean().backward()
        expected_gradients = []
        for parameter in _layer_parameters(reference):
            assert parameter.grad is not None
            gradient = parameter.grad.detach().float().clone()
            dist.all_reduce(gradient)
            expected_gradients.append(gradient / 2)

        for layer in sharded.language_model.model.layers:
            checkpoint(layer)
        fsdp_kwargs = {}
        if device_type == "cuda":
            fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
            )
        for layer in sharded.language_model.model.layers:
            fully_shard(
                layer,
                mesh=mesh,
                reshard_after_forward=True,
                **fsdp_kwargs,
            )

        context = _rebuild(sharded, chunks, execution_order="layer_major")
        assert "collective_pad_zero" not in context
        output = _semantic_image_output(sharded, context, rank=rank)
        forward_outer_calls = [layer.outer_calls for layer in sharded.language_model.model.layers]
        forward_inner_calls = [layer.forward_inference_calls for layer in sharded.language_model.model.layers]
        output.float().square().mean().backward()

        actual_gradients = []
        for parameter in _layer_parameters(sharded):
            assert parameter.grad is not None
            actual_gradients.append(parameter.grad.full_tensor().detach().float())
        max_error = max(
            float((actual - expected).abs().max().item())
            for actual, expected in zip(actual_gradients, expected_gradients)
        )
        cache_lengths = tuple(
            int(context["past_key_values"].key_cache[index].shape[0])
            for index in range(len(sharded.language_model.model.layers))
        )
        dist.barrier()
        result_queue.put(
            (
                rank,
                "ok",
                max_error,
                forward_outer_calls,
                forward_inner_calls,
                [layer.outer_calls for layer in sharded.language_model.model.layers],
                [layer.forward_inference_calls for layer in sharded.language_model.model.layers],
                cache_lengths,
            )
        )
    except Exception as error:
        result_queue.put((rank, "error", type(error).__name__, str(error), traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_layer_major_unequal_depths_preserve_fsdp2_checkpoint_collectives() -> None:
    dist = torch.distributed
    device_type = os.environ.get("UNIRL_BAGEL_FSDP_TEST_DEVICE", "cpu").strip().lower()
    if device_type not in {"cpu", "cuda"}:
        pytest.fail(f"unknown UNIRL_BAGEL_FSDP_TEST_DEVICE={device_type!r}")
    if not dist.is_available():
        pytest.skip("requires torch.distributed")
    if device_type == "cpu" and not dist.is_gloo_available():
        pytest.skip("requires torch.distributed with Gloo")
    if device_type == "cuda" and (
        not torch.cuda.is_available() or torch.cuda.device_count() < 2 or not dist.is_nccl_available()
    ):
        pytest.skip("requires two CUDA devices and torch.distributed with NCCL")
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
            context.Process(
                target=_run_layer_major_fsdp_worker,
                args=(rank, store_path, result_queue, device_type),
            )
            for rank in range(2)
        ]
        for process in processes:
            process.start()

        deadline = time.monotonic() + (120 if device_type == "cuda" else 45)
        for process in processes:
            process.join(max(0.0, deadline - time.monotonic()))

        hanging = [process for process in processes if process.is_alive()]
        for process in hanging:
            process.kill()
        for process in hanging:
            process.join(5)
        if hanging:
            pytest.fail(
                f"two-rank layer-major FSDP2 replay exceeded the {120 if device_type == 'cuda' else 45}-second deadline"
            )

        results = []
        for _ in processes:
            try:
                results.append(result_queue.get(timeout=2))
            except queue.Empty:
                break

    assert len(results) == 2, f"missing worker result; exit codes={[process.exitcode for process in processes]}"
    errors = [result for result in results if result[1] != "ok"]
    assert not errors, "\n".join(str(error) for error in errors)

    results.sort()
    for result in results:
        (
            rank,
            _status,
            max_error,
            forward_outer_calls,
            forward_inner_calls,
            total_outer_calls,
            total_inner_calls,
            cache_lengths,
        ) = result
        expected_inner_calls = len(_rank_chunks(rank)) + 1
        expected_cache_length = sum(len(chunk) for chunk in _rank_chunks(rank))
        assert max_error <= (1e-5 if device_type == "cuda" else 1e-6)
        assert forward_outer_calls == [2, 2, 2]
        assert forward_inner_calls == [expected_inner_calls] * 3
        assert total_outer_calls == [4, 4, 4]
        assert total_inner_calls == [expected_inner_calls * 2] * 3
        assert cache_lengths == (expected_cache_length,) * 3
