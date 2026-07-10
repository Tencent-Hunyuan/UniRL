"""CPU tests for Cosmos3 SFT packing + flow math (no checkpoints, no GPU)."""

import types

import pytest
import torch

from unirl.models.cosmos3.bundle import is_understanding_param
from unirl.models.cosmos3.packing import (
    noise_action_latents,
    noise_vision_latents,
    pack_joint_sequence,
    pad_action_chunk,
    sample_train_sigma,
)


def test_sample_train_sigma_bounds_and_shift_identity():
    gen = torch.Generator().manual_seed(0)
    for dist in ("uniform", "logitnormal"):
        for _ in range(50):
            s = sample_train_sigma(
                time_dist=dist,
                logitnormal_mean=0.0,
                logitnormal_std=1.0,
                shift=5.0,
                generator=gen,
                device=torch.device("cpu"),
            )
            assert 0.0 < float(s) < 1.0
    # shift=1 is the identity warp: sigma' = 1*s / (1 + 0*s) = s.
    gen_a = torch.Generator().manual_seed(7)
    gen_b = torch.Generator().manual_seed(7)
    s1 = sample_train_sigma(
        time_dist="uniform", logitnormal_mean=0.0, logitnormal_std=1.0, shift=1.0, generator=gen_a, device=torch.device("cpu")
    )
    base = torch.rand((), generator=gen_b)
    assert torch.allclose(s1, base.clamp(1e-4, 1 - 1e-4))


def test_noise_vision_latents_conditioning_and_velocity_relation():
    gen = torch.Generator().manual_seed(0)
    x0 = torch.randn(1, 4, 3, 6, 8)
    sigma = torch.tensor(0.7)
    x_t, v = noise_vision_latents(x0, sigma, condition_frame_indexes=[0], generator=gen)
    assert torch.equal(x_t[:, :, 0], x0[:, :, 0])  # clean conditioning frame
    # Flow relation on noisy frames: x_t = x0 + sigma * v  (since v = eps - x0).
    assert torch.allclose(x_t[:, :, 1:], x0[:, :, 1:] + sigma * v[:, :, 1:], atol=1e-5)


def test_action_padding_and_noising():
    actions = torch.randn(16, 7)
    padded = pad_action_chunk(actions, 64)
    assert padded.shape == (16, 64)
    assert torch.equal(padded[:, :7], actions)
    assert padded[:, 7:].abs().sum() == 0
    with pytest.raises(ValueError):
        pad_action_chunk(torch.randn(16, 65), 64)

    gen = torch.Generator().manual_seed(0)
    x_t, v = noise_action_latents(padded, torch.tensor(0.5), raw_action_dim=7, generator=gen)
    assert x_t[:, 7:].abs().sum() == 0
    assert v[:, 7:].abs().sum() == 0
    assert torch.allclose(x_t[:, :7], padded[:, :7] + 0.5 * v[:, :7], atol=1e-5)


def test_understanding_param_patterns():
    frozen = [
        "embed_tokens.weight",
        "lm_head.weight",
        "norm.weight",
        "layers.0.self_attn.to_q.weight",
        "layers.12.self_attn.to_out.weight",
        "layers.3.self_attn.norm_k.weight",
        "layers.7.mlp.gate_proj.weight",
        "layers.7.input_layernorm.weight",
        "layers.7.post_attention_layernorm.weight",
    ]
    trainable = [
        "norm_moe_gen.weight",
        "layers.0.self_attn.add_q_proj.weight",
        "layers.12.self_attn.to_add_out.weight",
        "layers.3.self_attn.norm_added_k.weight",
        "layers.7.mlp_moe_gen.gate_proj.weight",
        "layers.7.input_layernorm_moe_gen.weight",
        "proj_in.weight",
        "proj_out.bias",
        "time_embedder.linear_1.weight",
        "action_proj_in.fc.weight",
        "action_modality_embed",
    ]
    for name in frozen:
        assert is_understanding_param(name), name
    for name in trainable:
        assert not is_understanding_param(name), name


def _duck_pipe():
    """Duck-typed pipeline exposing the real diffusers segment helpers."""
    diffusers = pytest.importorskip("diffusers", minversion="0.39.0")
    del diffusers
    from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import Cosmos3OmniPipeline

    duck = types.SimpleNamespace()
    duck.transformer = types.SimpleNamespace(
        config=types.SimpleNamespace(
            enable_fps_modulation=True,
            unified_3d_mrope_temporal_modality_margin=15000,
            unified_3d_mrope_reset_spatial_ids=True,
            latent_patch_size=2,
            base_fps=24,
        ),
        dtype=torch.float32,
    )
    duck.vae = types.SimpleNamespace(config=types.SimpleNamespace(scale_factor_temporal=4))
    for name in ("_prepare_text_segment", "_prepare_vision_segment", "_prepare_action_segment"):
        setattr(duck, name, types.MethodType(getattr(Cosmos3OmniPipeline, name), duck))
    return duck


def test_pack_joint_sequence_layout():
    pipe = _duck_pipe()
    device = torch.device("cpu")
    latent_t, latent_h, latent_w = 3, 6, 8  # patch grid 3 x 3 x 4
    vision = torch.randn(1, 4, latent_t, latent_h, latent_w)
    input_ids = list(range(11))
    kwargs, meta = pack_joint_sequence(
        pipe,
        input_ids=input_ids,
        vision_tokens=vision,
        condition_frame_indexes=[0],
        vision_fps=15.0,
        device=device,
    )
    und = len(input_ids)
    tokens_per_frame = 3 * 4
    num_vision = latent_t * tokens_per_frame
    assert kwargs["und_len"] == und
    assert kwargs["sequence_length"] == und + num_vision
    assert kwargs["position_ids"].shape == (3, und + num_vision)
    assert kwargs["vision_sequence_indexes"].tolist() == list(range(und, und + num_vision))
    # Conditioned latent frame 0 is excluded from the loss token set.
    assert meta["num_noisy_vision_tokens"] == (latent_t - 1) * tokens_per_frame
    assert kwargs["vision_mse_loss_indexes"].min().item() == und + tokens_per_frame
    assert meta["vision_noisy_frames"].tolist() == [1, 2]

    # With an action chunk appended, its tokens land after the vision tokens.
    actions = torch.randn(5, 64)
    kwargs, meta = pack_joint_sequence(
        pipe,
        input_ids=input_ids,
        vision_tokens=vision,
        condition_frame_indexes=[0],
        vision_fps=15.0,
        device=device,
        action_tokens=actions,
        action_domain_id=torch.tensor([8]),
        action_fps=15.0,
    )
    assert kwargs["sequence_length"] == und + num_vision + 5
    assert kwargs["action_sequence_indexes"].tolist() == list(range(und + num_vision, und + num_vision + 5))
    assert meta["num_noisy_action_tokens"] == 5
    assert kwargs["action_domain_ids"][0].item() == 8
