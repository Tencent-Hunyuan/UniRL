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

from unirl.models.bagel.diffusion import BagelDiffusionParams, BagelDiffusionStage, BagelDiffusionStep
from unirl.models.bagel.rl_ops import forward_flow, forward_flow_many, install_layer_major_replay_dispatch
from unirl.sde.kernels import FlowSDEStrategy
from unirl.types.segments import make_image_segment


class _Cache:
    def __init__(self, num_layers: int) -> None:
        self.key_cache: dict[int, Optional[torch.Tensor]] = {index: None for index in range(num_layers)}
        self.value_cache: dict[int, Optional[torch.Tensor]] = {index: None for index in range(num_layers)}

    def fork(self) -> _Cache:
        cache = type(self)(len(self.key_cache))
        cache.key_cache = self.key_cache.copy()
        cache.value_cache = self.value_cache.copy()
        return cache


class _Rotary(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.register_buffer("frequencies", torch.linspace(0.07, 0.31, width), persistent=False)

    def forward(self, hidden: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        phase = position_ids.to(dtype=hidden.dtype).unsqueeze(-1) * self.frequencies.to(dtype=hidden.dtype)
        return phase.cos(), phase.sin()


def _record_outer_call(module: nn.Module, _args: Any, _kwargs: Any) -> None:
    module.outer_calls += 1


class _FlowBlock(nn.Module):
    def __init__(self, width: int, layer_index: int) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.norm = nn.LayerNorm(width)
        self.text_proj = nn.Linear(width, width, bias=False)
        self.image_proj = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        self.mlp = nn.Sequential(nn.Linear(width, width * 2), nn.SiLU(), nn.Linear(width * 2, width))
        self.outer_calls = 0
        self.forward_inference_calls = 0
        self.register_forward_pre_hook(_record_outer_call, with_kwargs=True)

    def forward(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, _Cache]:
        return self.forward_inference(*args, **kwargs)

    def forward_inference(
        self,
        *,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        packed_query_indexes: torch.Tensor,
        past_key_values: _Cache,
        key_values_lens: torch.Tensor,
        packed_key_value_indexes: torch.Tensor,
        update_past_key_values: bool,
        is_causal: bool,
        mode: str,
        packed_vae_token_indexes: torch.Tensor,
        packed_text_indexes: torch.Tensor,
    ) -> tuple[torch.Tensor, _Cache]:
        assert mode == "gen"
        assert not update_past_key_values
        assert not is_causal
        assert query_lens.tolist() == [int(packed_query_sequence.shape[0])]
        assert key_values_lens.tolist() == [int(packed_key_value_indexes.numel())]
        assert packed_query_indexes.numel() == packed_query_sequence.shape[0]
        self.forward_inference_calls += 1

        cos, sin = packed_query_position_embeddings
        hidden = self.norm(packed_query_sequence + 0.02 * (cos + sin))
        projected = torch.zeros_like(hidden)
        projected[packed_text_indexes] = self.text_proj(hidden[packed_text_indexes])
        projected[packed_vae_token_indexes] = self.image_proj(hidden[packed_vae_token_indexes])
        key = past_key_values.key_cache[self.layer_index]
        value = past_key_values.value_cache[self.layer_index]
        assert key is not None and value is not None
        context = (key + 0.37 * value).mean(dim=0, keepdim=True)
        hidden = packed_query_sequence + self.output(torch.tanh(projected + context))
        return hidden + self.mlp(self.norm(hidden)), past_key_values


class _FlowLMModel(nn.Module):
    def __init__(self, width: int, num_layers: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(64, width)
        self.rotary_emb = _Rotary(width)
        self.layers = nn.ModuleList(_FlowBlock(width, index) for index in range(num_layers))
        self.norm = nn.LayerNorm(width)
        self.norm_moe_gen = nn.LayerNorm(width)
        self.use_moe = True
        self.enable_taylorseer = False

    def forward(self, *args: Any, **kwargs: Any) -> SimpleNamespace:
        return self.forward_inference(*args, **kwargs)

    def forward_inference(
        self,
        *,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_ids: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: _Cache,
        key_values_lens: torch.Tensor,
        packed_key_value_indexes: torch.Tensor,
        update_past_key_values: bool,
        is_causal: bool,
        mode: str,
        packed_vae_token_indexes: torch.Tensor,
        packed_text_indexes: torch.Tensor,
    ) -> SimpleNamespace:
        cos, sin = self.rotary_emb(packed_query_sequence, packed_query_position_ids.unsqueeze(0))
        positions = (cos.squeeze(0), sin.squeeze(0))
        hidden = packed_query_sequence
        cache = past_key_values
        for layer in self.layers:
            hidden, cache = layer(
                packed_query_sequence=hidden,
                query_lens=query_lens,
                packed_query_position_embeddings=positions,
                packed_query_indexes=packed_query_indexes,
                past_key_values=cache,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=update_past_key_values,
                is_causal=is_causal,
                mode=mode,
                packed_vae_token_indexes=packed_vae_token_indexes,
                packed_text_indexes=packed_text_indexes,
            )
        normalized = torch.zeros_like(hidden)
        normalized[packed_text_indexes] = self.norm(hidden[packed_text_indexes])
        normalized[packed_vae_token_indexes] = self.norm_moe_gen(hidden[packed_vae_token_indexes])
        return SimpleNamespace(packed_query_sequence=normalized, past_key_values=cache)


class _FlowLanguageModel(nn.Module):
    def __init__(self, width: int, num_layers: int) -> None:
        super().__init__()
        self.model = _FlowLMModel(width, num_layers)

    def forward_inference(self, **kwargs: Any) -> SimpleNamespace:
        return self.model(**kwargs)


class _TimeEmbedding(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.proj = nn.Linear(1, width)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        return self.proj(timestep.reshape(-1, 1).to(dtype=self.proj.weight.dtype))


class _FlowBagel(nn.Module):
    def __init__(self, *, width: int = 8, latent_width: int = 4, num_layers: int = 3) -> None:
        super().__init__()
        self.hidden_size = width
        self.use_moe = True
        self.language_model = _FlowLanguageModel(width, num_layers)
        self.latent_pos_embed = nn.Embedding(32, width)
        self.time_embedder = _TimeEmbedding(width)
        self.vae2llm = nn.Linear(latent_width, width)
        self.llm2vae = nn.Linear(width, latent_width)

    @torch.no_grad()
    def _forward_flow(
        self,
        *,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        packed_vae_token_indexes: torch.Tensor,
        packed_vae_position_ids: torch.Tensor,
        packed_text_ids: torch.Tensor,
        packed_text_indexes: torch.Tensor,
        packed_indexes: torch.Tensor,
        packed_position_ids: torch.Tensor,
        packed_seqlens: torch.Tensor,
        key_values_lens: torch.Tensor,
        past_key_values: _Cache,
        packed_key_value_indexes: torch.Tensor,
        cfg_text_scale: float = 1.0,
        cfg_img_scale: float = 1.0,
        **_kwargs: Any,
    ) -> torch.Tensor:
        del cfg_text_scale, cfg_img_scale
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding
        assert timestep.unique().shape[0] == 1
        packed_latent = (
            self.vae2llm(x_t) + self.time_embedder(timestep) + self.latent_pos_embed(packed_vae_position_ids)
        )
        packed_sequence[packed_vae_token_indexes] = packed_latent.to(dtype=packed_sequence.dtype)
        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=False,
            is_causal=False,
            mode="gen",
            packed_vae_token_indexes=packed_vae_token_indexes,
            packed_text_indexes=packed_text_indexes,
        )
        return self.llm2vae(output.packed_query_sequence)[packed_vae_token_indexes]


def _make_forward_kwargs(
    model: _FlowBagel,
    *,
    cache_length: int,
    requires_grad: bool,
    rank: int = 0,
) -> tuple[dict[str, Any], list[torch.Tensor]]:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    cache = _Cache(len(model.language_model.model.layers))
    cache_leaves: list[torch.Tensor] = []
    for layer_index in range(len(model.language_model.model.layers)):
        key = (
            torch.linspace(-0.3, 0.4, cache_length * model.hidden_size, device=device, dtype=dtype)
            .reshape(cache_length, model.hidden_size)
            .add_(0.01 * (rank + layer_index))
            .requires_grad_(requires_grad)
        )
        value = (
            torch.linspace(0.2, -0.5, cache_length * model.hidden_size, device=device, dtype=dtype)
            .reshape(cache_length, model.hidden_size)
            .add_(0.02 * (rank + layer_index))
            .requires_grad_(requires_grad)
        )
        cache.key_cache[layer_index] = key
        cache.value_cache[layer_index] = value
        cache_leaves.extend((key, value))
    return (
        {
            "packed_vae_token_indexes": torch.tensor([1, 2], device=device),
            "packed_vae_position_ids": torch.tensor([1, 2], device=device),
            "packed_text_ids": torch.tensor([5, 9], device=device),
            "packed_text_indexes": torch.tensor([0, 3], device=device),
            "packed_indexes": torch.arange(cache_length, cache_length + 4, device=device),
            "packed_position_ids": torch.arange(4, device=device),
            "packed_seqlens": torch.tensor([4], dtype=torch.int, device=device),
            "key_values_lens": torch.tensor([cache_length], dtype=torch.int, device=device),
            "past_key_values": cache,
            "packed_key_value_indexes": torch.arange(cache_length, device=device),
        },
        cache_leaves,
    )


def _make_streams(
    *,
    device: torch.device,
    dtype: torch.dtype,
    rank: int = 0,
    requires_grad: bool,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    samples = [
        (torch.arange(8, device=device, dtype=dtype).reshape(2, 4) * 0.03 + 0.1 * index + rank * 0.01)
        .clone()
        .requires_grad_(requires_grad)
        for index in range(3)
    ]
    timesteps = [torch.full((2,), value, device=device, dtype=dtype) for value in (0.83, 0.57, 0.29)]
    return samples, timesteps


def _replay_loss(
    velocities: tuple[torch.Tensor, ...],
    samples: list[torch.Tensor],
    timesteps: list[torch.Tensor],
) -> torch.Tensor:
    step = BagelDiffusionStep()
    strategy = FlowSDEStrategy()
    terms = []
    for index, (velocity, sample, timestep) in enumerate(zip(velocities, samples, timesteps)):
        prev_sample = sample.detach() * 0.91 + 0.02 * (index + 1)
        _, log_prob, prev_mean = step.denoise(
            strategy,
            v_t=velocity,
            x_t=sample,
            sigma=timestep[0],
            sigma_next=timestep[0] - 0.08,
            sigma_max=timesteps[0][0],
            eta=0.7,
            prev_sample=prev_sample,
        )
        assert log_prob is not None and prev_mean is not None
        terms.append(-(index + 1) * log_prob + 0.01 * prev_mean.square().mean())
    return torch.stack(terms).mean()


def _layer_parameters(model: _FlowBagel):
    return model.language_model.model.layers.parameters()


def test_forward_flow_many_matches_serial_output_loss_and_full_gradients() -> None:
    torch.manual_seed(7123)
    serial = _FlowBagel().double().eval()
    many = _FlowBagel().double().eval()
    many.load_state_dict(serial.state_dict())
    install_layer_major_replay_dispatch(serial)
    install_layer_major_replay_dispatch(many)
    serial_kwargs, serial_cache_leaves = _make_forward_kwargs(serial, cache_length=5, requires_grad=True)
    many_kwargs, many_cache_leaves = _make_forward_kwargs(many, cache_length=5, requires_grad=True)
    serial_samples, timesteps = _make_streams(device=torch.device("cpu"), dtype=torch.float64, requires_grad=True)
    many_samples = [sample.detach().clone().requires_grad_(True) for sample in serial_samples]

    serial_velocities = tuple(
        forward_flow(
            serial,
            x_t=sample,
            timestep=timestep,
            cfg_text_scale=1.0,
            cfg_img_scale=1.0,
            **serial_kwargs,
        )
        for sample, timestep in zip(serial_samples, timesteps)
    )
    many_velocities = forward_flow_many(
        many,
        x_ts=many_samples,
        timesteps=timesteps,
        forward_kwargs=many_kwargs,
        cfg_text_scales=(1.0, 1.0, 1.0),
        cfg_img_scales=(1.0, 1.0, 1.0),
    )

    for serial_velocity, many_velocity in zip(serial_velocities, many_velocities):
        torch.testing.assert_close(many_velocity, serial_velocity, rtol=0, atol=0)
    serial_loss = _replay_loss(serial_velocities, serial_samples, timesteps)
    many_loss = _replay_loss(many_velocities, many_samples, timesteps)
    torch.testing.assert_close(many_loss, serial_loss, rtol=0, atol=0)
    serial_loss.backward()
    many_loss.backward()

    assert tuple(serial.named_parameters())
    for (serial_name, serial_parameter), (many_name, many_parameter) in zip(
        serial.named_parameters(), many.named_parameters()
    ):
        assert serial_name == many_name
        assert serial_parameter.grad is not None, serial_name
        assert many_parameter.grad is not None, many_name
        torch.testing.assert_close(many_parameter.grad, serial_parameter.grad, rtol=1e-12, atol=1e-12)
    for serial_sample, many_sample in zip(serial_samples, many_samples):
        torch.testing.assert_close(many_sample.grad, serial_sample.grad, rtol=1e-12, atol=1e-12)
    for serial_leaf, many_leaf in zip(serial_cache_leaves, many_cache_leaves):
        torch.testing.assert_close(many_leaf.grad, serial_leaf.grad, rtol=1e-12, atol=1e-12)

    assert [layer.outer_calls for layer in serial.language_model.model.layers] == [3, 3, 3]
    assert [layer.outer_calls for layer in many.language_model.model.layers] == [1, 1, 1]
    assert [layer.forward_inference_calls for layer in many.language_model.model.layers] == [3, 3, 3]


def test_predict_velocities_falls_back_for_single_stream_and_cfg() -> None:
    model = _FlowBagel().eval()
    install_layer_major_replay_dispatch(model)
    forward_kwargs, _ = _make_forward_kwargs(model, cache_length=2, requires_grad=False)
    samples, timesteps = _make_streams(device=torch.device("cpu"), dtype=torch.float32, requires_grad=False)
    step = BagelDiffusionStep()

    single = step.predict_velocities(
        model,
        x_ts=samples[:1],
        t_curs=(timesteps[0][0],),
        cfg_text_scales=(1.0,),
        cfg_img_scales=(1.0,),
        forward_kwargs=forward_kwargs,
        use_layer_major=True,
    )
    assert len(single) == 1
    assert [layer.outer_calls for layer in model.language_model.model.layers] == [1, 1, 1]

    cfg = step.predict_velocities(
        model,
        x_ts=samples[:2],
        t_curs=(timesteps[0][0], timesteps[1][0]),
        cfg_text_scales=(2.0, 2.0),
        cfg_img_scales=(1.0, 1.0),
        forward_kwargs=forward_kwargs,
        use_layer_major=True,
    )
    assert len(cfg) == 2
    assert [layer.outer_calls for layer in model.language_model.model.layers] == [3, 3, 3]


@pytest.mark.parametrize(("enabled", "expected_outer_calls"), [(False, 3), (True, 1)])
def test_diffusion_stage_flow_many_requires_explicit_opt_in(enabled: bool, expected_outer_calls: int) -> None:
    model = _FlowBagel().eval()
    install_layer_major_replay_dispatch(model)
    forward_kwargs, _ = _make_forward_kwargs(model, cache_length=2, requires_grad=False)
    samples, timesteps = _make_streams(device=torch.device("cpu"), dtype=torch.float32, requires_grad=False)
    stage = BagelDiffusionStage(
        model=SimpleNamespace(model=model, device="cpu"),
        t2ti_replay_execution_order="layer_major",
        t2ti_flow_many_enabled=enabled,
    )

    velocities = stage.predict_velocities_at(
        forward_kwargs,
        samples=samples,
        sigmas=[timestep[0] for timestep in timesteps],
        params=BagelDiffusionParams(),
    )

    assert len(velocities) == 3
    assert stage.t2ti_flow_many_enabled is enabled
    assert [layer.outer_calls for layer in model.language_model.model.layers] == [expected_outer_calls] * 3


def test_diffusion_stage_fused_replay_matches_serial() -> None:
    torch.manual_seed(8123)
    serial_model = _FlowBagel().double().eval()
    many_model = _FlowBagel().double().eval()
    many_model.load_state_dict(serial_model.state_dict())
    install_layer_major_replay_dispatch(serial_model)
    install_layer_major_replay_dispatch(many_model)
    serial_kwargs, _ = _make_forward_kwargs(serial_model, cache_length=5, requires_grad=False)
    many_kwargs, _ = _make_forward_kwargs(many_model, cache_length=5, requires_grad=False)
    samples, _ = _make_streams(device=torch.device("cpu"), dtype=torch.float64, requires_grad=False)
    final_sample = samples[-1] * 0.91 + 0.04
    segment = make_image_segment(
        latents=torch.stack((*samples, final_sample), dim=0).unsqueeze(0),
        sigmas=torch.tensor([0.83, 0.57, 0.29, 0.12], dtype=torch.float64),
        indices=torch.arange(4),
        sde_indices=torch.arange(3),
    )
    params = BagelDiffusionParams(eta=0.7)

    def replay(model: _FlowBagel, forward_kwargs: dict[str, Any], *, flow_many: bool):
        stage = BagelDiffusionStage(
            model=SimpleNamespace(model=model, device="cpu"),
            trajectory_precision="fp32",
            logprob_precision="fp32",
            t2ti_replay_execution_order="layer_major",
            t2ti_flow_many_enabled=flow_many,
        )
        return stage._replay_from_forward_kwargs_impl(
            forward_kwargs,
            segment=segment,
            params=params,
            schedule=segment.sigmas,
            target=[0, 1, 2],
            collective_pad_zero=None,
        )

    serial_result, serial_velocities = replay(serial_model, serial_kwargs, flow_many=False)
    many_result, many_velocities = replay(many_model, many_kwargs, flow_many=True)

    torch.testing.assert_close(many_result.log_probs, serial_result.log_probs, rtol=0, atol=0)
    torch.testing.assert_close(many_result.prev_sample_means, serial_result.prev_sample_means, rtol=0, atol=0)
    for serial_velocity, many_velocity in zip(serial_velocities, many_velocities):
        torch.testing.assert_close(many_velocity, serial_velocity, rtol=0, atol=0)
    assert [layer.outer_calls for layer in serial_model.language_model.model.layers] == [3, 3, 3]
    assert [layer.outer_calls for layer in many_model.language_model.model.layers] == [1, 1, 1]


@pytest.mark.parametrize(
    ("cfg_text_scales", "cfg_img_scales"),
    [((2.0, 1.0), (1.0, 1.0)), ((1.0, 1.0), (1.0, 2.0))],
)
def test_forward_flow_many_rejects_cfg(cfg_text_scales, cfg_img_scales) -> None:
    model = _FlowBagel().eval()
    install_layer_major_replay_dispatch(model)
    forward_kwargs, _ = _make_forward_kwargs(model, cache_length=2, requires_grad=False)
    samples, timesteps = _make_streams(device=torch.device("cpu"), dtype=torch.float32, requires_grad=False)

    with pytest.raises(ValueError, match="CFG text/image scales exactly equal to 1"):
        forward_flow_many(
            model,
            x_ts=samples[:2],
            timesteps=timesteps[:2],
            forward_kwargs=forward_kwargs,
            cfg_text_scales=cfg_text_scales,
            cfg_img_scales=cfg_img_scales,
        )


def test_forward_flow_many_rejects_mixed_latent_grad_state() -> None:
    model = _FlowBagel().eval()
    install_layer_major_replay_dispatch(model)
    forward_kwargs, _ = _make_forward_kwargs(model, cache_length=2, requires_grad=False)
    samples, timesteps = _make_streams(device=torch.device("cpu"), dtype=torch.float32, requires_grad=False)
    samples[0].requires_grad_(True)

    with pytest.raises(ValueError, match="mixed latent requires_grad"):
        forward_flow_many(
            model,
            x_ts=samples[:2],
            timesteps=timesteps[:2],
            forward_kwargs=forward_kwargs,
            cfg_text_scales=(1.0, 1.0),
            cfg_img_scales=(1.0, 1.0),
        )


def _rank_cache_length(rank: int) -> int:
    return 2 if rank == 0 else 7


def _run_flow_many_fsdp_worker(rank: int, store_path: str, result_queue: Any, device_type: str) -> None:
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

        torch.manual_seed(9631)
        compute_dtype = torch.bfloat16 if device_type == "cuda" else torch.float32
        reference = _FlowBagel().eval().to(device=device, dtype=compute_dtype)
        sharded = _FlowBagel().eval().to(device=device, dtype=compute_dtype)
        sharded.load_state_dict(reference.state_dict())
        if device_type == "cuda":
            for parameter in _layer_parameters(sharded):
                parameter.data = parameter.data.float()
        install_layer_major_replay_dispatch(reference)
        install_layer_major_replay_dispatch(sharded)
        cache_length = _rank_cache_length(rank)
        reference_kwargs, reference_cache_leaves = _make_forward_kwargs(
            reference, cache_length=cache_length, requires_grad=True, rank=rank
        )
        sharded_kwargs, sharded_cache_leaves = _make_forward_kwargs(
            sharded, cache_length=cache_length, requires_grad=True, rank=rank
        )
        reference_samples, timesteps = _make_streams(device=device, dtype=compute_dtype, rank=rank, requires_grad=True)
        sharded_samples = [sample.detach().clone().requires_grad_(True) for sample in reference_samples]

        reference_velocities = tuple(
            forward_flow(
                reference,
                x_t=sample,
                timestep=timestep,
                cfg_text_scale=1.0,
                cfg_img_scale=1.0,
                **reference_kwargs,
            )
            for sample, timestep in zip(reference_samples, timesteps)
        )
        _replay_loss(reference_velocities, reference_samples, timesteps).float().backward()
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
            fully_shard(layer, mesh=mesh, reshard_after_forward=True, **fsdp_kwargs)

        velocities = forward_flow_many(
            sharded,
            x_ts=sharded_samples,
            timesteps=timesteps,
            forward_kwargs=sharded_kwargs,
            cfg_text_scales=(1.0, 1.0, 1.0),
            cfg_img_scales=(1.0, 1.0, 1.0),
        )
        forward_outer_calls = [layer.outer_calls for layer in sharded.language_model.model.layers]
        forward_inner_calls = [layer.forward_inference_calls for layer in sharded.language_model.model.layers]
        _replay_loss(velocities, sharded_samples, timesteps).float().backward()

        actual_gradients = []
        for parameter in _layer_parameters(sharded):
            assert parameter.grad is not None
            actual_gradients.append(parameter.grad.full_tensor().detach().float())
        max_error = max(
            float((actual - expected).abs().max().item())
            for actual, expected in zip(actual_gradients, expected_gradients)
        )
        input_max_error = max(
            [
                float((actual.grad.float() - expected.grad.float()).abs().max().item())
                for actual, expected in zip(sharded_samples, reference_samples)
            ]
            + [
                float((actual.grad.float() - expected.grad.float()).abs().max().item())
                for actual, expected in zip(sharded_cache_leaves, reference_cache_leaves)
            ]
        )
        dist.barrier()
        result_queue.put(
            (
                rank,
                "ok",
                max_error,
                input_max_error,
                forward_outer_calls,
                forward_inner_calls,
                [layer.outer_calls for layer in sharded.language_model.model.layers],
                [layer.forward_inference_calls for layer in sharded.language_model.model.layers],
            )
        )
    except Exception as error:
        result_queue.put((rank, "error", type(error).__name__, str(error), traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_forward_flow_many_preserves_fsdp2_checkpoint_collectives_with_unequal_cache_lengths() -> None:
    dist = torch.distributed
    device_type = os.environ.get("UNIRL_BAGEL_FLOW_MANY_FSDP_TEST_DEVICE", "cpu").strip().lower()
    if device_type not in {"cpu", "cuda"}:
        pytest.fail(f"unknown UNIRL_BAGEL_FLOW_MANY_FSDP_TEST_DEVICE={device_type!r}")
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
                target=_run_flow_many_fsdp_worker,
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
                f"two-rank flow-many FSDP2 test exceeded the {120 if device_type == 'cuda' else 45}-second deadline"
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
    for result in results:
        (
            _rank,
            _status,
            max_error,
            input_max_error,
            forward_outer_calls,
            forward_inner_calls,
            total_outer_calls,
            total_inner_calls,
        ) = result
        assert max_error <= (5e-2 if device_type == "cuda" else 1e-5)
        assert input_max_error <= (5e-2 if device_type == "cuda" else 1e-5)
        assert forward_outer_calls == [1, 1, 1]
        assert forward_inner_calls == [3, 3, 3]
        assert total_outer_calls == [2, 2, 2]
        assert total_inner_calls == [6, 6, 6]
